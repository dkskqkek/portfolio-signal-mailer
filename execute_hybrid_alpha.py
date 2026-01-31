# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys
import time
import requests
from datetime import datetime

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.kr_stock_scanner import KRStockScanner
from signal_mailer.order_executor import OrderExecutor
from signal_mailer.trade_limit_counter import TradeLimitCounter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("LiveExecution")


def send_discord_msg(config, title, message, color=0x00FF00, fields=None):
    """Send enhanced Discord notification with optional structured fields.

    Args:
        config: Configuration dictionary with Discord webhook URL
        title: Notification title
        message: Main message body
        color: Embed color (default: green)
        fields: Optional list of {"name": str, "value": str, "inline": bool} dicts
    """
    webhook_url = config.get("discord", {}).get("webhook_url")
    if not webhook_url:
        return

    embed = {
        "title": title,
        "description": message,
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if fields:
        embed["fields"] = fields

    payload = {"embeds": [embed]}

    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Discord notice failed: {e}")


def run_hybrid_alpha_execution(dry_run=True):
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 1. Initialize API and Scanner
    kis = KISAPIWrapper(config["kis"])
    scanner = KRStockScanner(kis)
    executor = OrderExecutor(kis)
    trade_limiter = TradeLimitCounter(
        limits_file=os.path.join(current_dir, "data", "trade_limits.json"),
        max_daily_trades=10,
    )

    print(f"\n--- [MODE: {'MOCK' if kis.is_mock else 'REAL'}] Hybrid Alpha Engine ---")
    if dry_run:
        print("💡 DRY RUN MODE: No orders will be executed.")

    # Check trade limit
    remaining_trades = trade_limiter.get_remaining("hybrid_alpha")
    print(f"📊 Remaining trades today: {remaining_trades}/10")
    if remaining_trades == 0:
        print("⚠️  DAILY TRADE LIMIT REACHED (10 trades). No more trades allowed today.")
        send_discord_msg(
            config,
            "⚠️ [Hybrid Alpha] Trade Limit Reached",
            "일일 거래 한도 도달 (10건). 오늘은 더 이상 거래하지 않습니다.",
            color=0xFFA500,
        )
        return

    # 2. Get Current Holdings
    holdings = executor.get_balance()
    held_tickers = {h["pdno"]: h for h in holdings}
    print(f"📊 Current Holdings: {len(holdings)} stocks")

    # 3. Market Scan
    print("🔍 Scanning Market for Candidates (Top 200)...")
    candidates = scanner.scan_full_market(limit=200)
    top_5 = candidates[:5]
    top_5_tickers = [c["ticker"] for c in top_5]

    # Calculate Target Allocation (Global Equity * 0.5)
    total_equity = executor.get_total_equity()
    target_kr_equity = total_equity * 0.5
    target_count = 5
    allocation_per_stock = target_kr_equity / target_count

    print(f"📊 Total Global Equity: {total_equity:,.0f}원")
    print(
        f"📊 Target KR Allocation (50%): {target_kr_equity:,.0f}원 (Per Stock: {allocation_per_stock:,.0f}원)"
    )

    cash = executor.get_kr_cash()
    print(f"💰 Available Cash (KR): {cash:,.0f}원")

    # Execute Strategy
    sell_count = 0
    buy_count = 0
    print("\n=== SELL PHASE ===")
    # 4. SELL Logic: Full Sell or Partial Sell for Rebalancing
    for ticker, h in held_tickers.items():
        name = h.get("prdt_name", ticker)
        current_qty = int(h.get("hldg_qty", 0))
        curr_price = float(h.get("prpr", 0))

        # Case A: Ticker no longer in Top 5 -> FULL SELL
        if ticker not in top_5_tickers:
            print(f"📉 [FULL SELL] {name} ({ticker}): Signal lost or ranked out.")
            if not dry_run:
                # Check trade limit before executing
                if not trade_limiter.check_and_increment("hybrid_alpha"):
                    print(f"   ⚠️  Cannot sell {name}: Daily trade limit reached")
                    send_discord_msg(
                        config,
                        "⚠️ [Hybrid Alpha] Trade Limit Blocked",
                        f"{name} 매도 차단: 일일 거래 한도 도달",
                        color=0xFFA500,
                    )
                    continue

                res = executor.create_order(
                    ticker, side="SELL", qty=current_qty, ord_type="01"
                )
                if res and res.get("rt_cd") == "0":
                    sell_amt = current_qty * curr_price
                    msg = f"**{name}** ({ticker})\n수량: {current_qty}주 (전량 청산)\n단가: ~{curr_price:,}원\n매도금액: {sell_amt:,.0f}원"
                    print(f"   ✅ FULL SELL Success: {name} ({current_qty}주)")
                    send_discord_msg(
                        config, "📉 [Hybrid Alpha] 전량 매도", msg, color=0xFF0000
                    )
                    sell_count += 1
            else:
                print(f"   [DRY RUN] Would sell all {current_qty} shares.")
                sell_count += 1

        # Case B: Ticker in Top 5 but exceeds allocation -> PARTIAL SELL
        else:
            target_qty = int(allocation_per_stock // curr_price)
            if current_qty > target_qty * 1.1:  # Allow 10% buffer to avoid micro-trades
                sell_qty = current_qty - target_qty
                print(
                    f"📉 [PARTIAL SELL] {name} ({ticker}): Reducing weight to 50% split. {current_qty} -> {target_qty}"
                )
                if not dry_run:
                    # Check trade limit
                    if not trade_limiter.check_and_increment("hybrid_alpha"):
                        print(
                            f"   ⚠️  Cannot partial sell {name}: Daily trade limit reached"
                        )
                        continue

                    res = executor.create_order(
                        ticker, side="SELL", qty=sell_qty, ord_type="01"
                    )
                    if res and res.get("rt_cd") == "0":
                        msg = f"**{name}** ({ticker})\n수량: {current_qty}주 → {target_qty}주 (🔻{sell_qty}주 매도)\n단가: ~{curr_price:,}원\n매도금액: {sell_qty * curr_price:,.0f}원"
                        print(f"   ✅ PARTIAL SELL Success: {name} ({sell_qty}주)")
                        send_discord_msg(
                            config,
                            "📉 [Hybrid Alpha] 비중 조절 매도",
                            msg,
                            color=0xFFA500,
                        )
                        sell_count += 1
                else:
                    print(f"   [DRY RUN] Would sell {sell_qty} shares to rebalance.")
                    sell_count += 1

    if sell_count > 0:
        time.sleep(1)

    # 5. BUY Logic: Top 5 Stocks
    cash = executor.get_cash()
    print(f"💰 Available Cash (Domestic): {cash:,}원")

    for stock in top_5:
        ticker = stock["ticker"]
        name = stock["name"]
        curr_price = stock["price"]

        target_qty = int(allocation_per_stock // curr_price)
        current_qty = int(held_tickers.get(ticker, {}).get("hldg_qty", 0))

        needed_qty = target_qty - current_qty

        if needed_qty > 0:
            # Check cash limit
            max_qty_by_cash = int(cash // curr_price)
            buy_qty = min(needed_qty, max_qty_by_cash)

            if buy_qty > 0:
                invest_amt = buy_qty * curr_price

                # Position size limit check: max 20% of total equity per order
                max_position_size = total_equity * 0.20
                if invest_amt > max_position_size:
                    print(
                        f"   ⚠️  Order size {invest_amt:,.0f}원 exceeds 20% limit ({max_position_size:,.0f}원)"
                    )
                    print("   Reducing order size to comply with risk limits...")
                    buy_qty = int(max_position_size // curr_price)
                    invest_amt = buy_qty * curr_price

                    if buy_qty <= 0:
                        print(
                            f"   ⚠️  Cannot buy {name}: Even minimum order exceeds 20% limit"
                        )
                        send_discord_msg(
                            config,
                            "⚠️ [Hybrid Alpha] Order Size Blocked",
                            f"{name} 매수 차단: 주문 금액이 계좌의 20%를 초과합니다.",
                            color=0xFFA500,
                        )
                        continue

                print(
                    f"🚀 [BUY/ADD] {name} ({ticker}): {buy_qty} shares @ ~{curr_price:,}원"
                )
                if not dry_run:
                    # Check trade limit
                    if not trade_limiter.check_and_increment("hybrid_alpha"):
                        print(f"   ⚠️  Cannot buy {name}: Daily trade limit reached")
                        break  # Stop buying if limit reached

                    result = executor.create_order(
                        ticker, side="BUY", qty=buy_qty, ord_type="01"
                    )
                    if result and result.get("rt_cd") == "0":
                        cash -= invest_amt
                        msg = f"**{name}** ({ticker})\n수량: {buy_qty}주 @ ~{curr_price:,}원\n매수금액: {invest_amt:,.0f}원\n━━━━━━━━━━━━━━\n💰 잔여 현금: {cash:,.0f}원"
                        print(f"   ✅ BUY Success: {name} ({buy_qty}주)")
                        send_discord_msg(
                            config, "🚀 [Hybrid Alpha] 매수 완료", msg, color=0x00FF00
                        )
                        buy_count += 1
                    else:
                        print(
                            f"   ❌ BUY Failed: {result.get('msg1') if result else 'Error'}"
                        )
                else:
                    print(f"   [DRY RUN] Would buy {buy_qty} shares.")
                time.sleep(1)
            else:
                print(f"⚠️ [BUY SKIP] {name} ({ticker}) - Not enough cash.")
        else:
            print(f"💎 Holding {name} ({ticker}): Allocation reached or exceeded.")

    print("\n--- [Execution Finished] ---")


if __name__ == "__main__":
    # Default to DRY RUN for safety, but we'll call with False for implementation
    run_hybrid_alpha_execution(dry_run=True)
