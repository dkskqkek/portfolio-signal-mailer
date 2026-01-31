# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys
import time
from datetime import datetime

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor
from signal_mailer.mama_lite_predictor import MAMAPredictor
from signal_mailer.trade_limit_counter import TradeLimitCounter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MAMA_Execution")


def send_discord_msg(config, title, message, color=0x00FF00, fields=None):
    """Send enhanced Discord notification with optional structured fields.

    Args:
        config: Configuration dictionary with Discord webhook URL
        title: Notification title
        message: Main message body
        color: Embed color (default: green)
        fields: Optional list of {"name": str, "value": str, "inline": bool} dicts
    """
    import requests

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


def get_exchange_code(ticker):
    """
    Return exchange code for KIS US API.
    Updated for v3.0 9-ETF Universe.
    """
    # MAMA Pro Universe
    # SPY: ARCA(AMS), QQQ: NASDAQ(NAS), IWM: ARCA(AMS)
    # TLT/IEF/SHY: NASDAQ(NAS)
    # GLD: ARCA(AMS), DBC: ARCA(AMS), BIL: ARCA(AMS)
    nas_list = [
        "TLT",
        "QQQ",
        "IEF",
        "SHY",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "NFLX",
        "AVGO",
    ]
    if ticker in nas_list:
        return "NAS"
    return "AMS"


class TransactionCostModel:
    def __init__(self, slippage_bps=5.0, commission_rate=0.0025):
        self.slippage = slippage_bps / 10000.0
        self.comm = commission_rate

    def estimate_cost(self, amount_usd):
        return amount_usd * (self.slippage + self.comm)


def run_mama_lite_execution(dry_run=False):
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 1. Initialize Components
    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)
    predictor = MAMAPredictor(config_path=config_path)

    t_config = config.get("trading", {})
    trade_limiter = TradeLimitCounter(
        limits_file=os.path.join(current_dir, "data", "trade_limits.json"),
        max_daily_trades=t_config.get("max_daily_trades", 15),
    )
    cost_model = TransactionCostModel(
        slippage_bps=t_config.get("slippage_bps", 5.0),
        commission_rate=t_config.get("commission_rate", 0.0025),
    )

    print(
        f"\n--- [MODE: {'MOCK' if kis.is_mock else 'REAL'}] Antigravity v3.0 Engine ---"
    )

    # Real-time FX
    exch_rate = kis.get_exchange_rate()
    print(f"💱 Real-time USD/KRW Rate: {exch_rate:,.2f}")

    if dry_run:
        print("💡 DRY RUN MODE: No orders will be executed.")

    # Check trade limit
    remaining_trades = trade_limiter.get_remaining("mama_lite")
    print(f"📊 Remaining trades today: {remaining_trades}")
    if remaining_trades <= 0:
        print("⚠️ DAILY TRADE LIMIT REACHED. Aborting.")
        return

    # 2. Get Predicted Weights
    target_weights = predictor.predict_portfolio()
    if not target_weights:
        print("❌ Prediction Failed. Aborting.")
        return

    # 3. Get Current Portfolio (US)
    holdings = executor.get_us_balance()
    current_holdings = {}
    current_total_val_usd = 0.0

    print("\n📊 Current US Holdings:")
    for h in holdings:
        ticker = h.get("ovrs_pdno", "")
        qty = float(h.get("ovrs_cblc_qty", 0))
        val_usd = float(h.get("frcr_evlu_amt2", 0))
        if qty > 0:
            current_holdings[ticker] = qty
            current_total_val_usd += val_usd
            print(f"   - {ticker}: {qty} shares (${val_usd:,.2f})")

    # Capital Allocation Rules
    global_total_equity_krw = executor.get_total_equity()
    target_us_equity_krw = global_total_equity_krw * 0.5
    target_us_equity_usd = target_us_equity_krw / exch_rate

    print(f"\n🌍 Net Worth: {global_total_equity_krw:,.0f}원")
    print(f"🎯 Target US Equity: ${target_us_equity_usd:,.2f}")

    total_equity_base = target_us_equity_usd

    # 4. Rebalancing Logic
    trades = []

    # Sell Loop
    for t, qty in current_holdings.items():
        target_w = target_weights.get(t, 0.0)
        curr_price = kis.get_us_current_price(t, exchange=get_exchange_code(t))
        if not curr_price:
            continue

        current_val = qty * curr_price
        target_val = total_equity_base * target_w
        diff_val = target_val - current_val

        if diff_val < -50:  # Threshold $50
            sell_qty = int(abs(diff_val) / curr_price)
            if sell_qty > 0:
                cost = cost_model.estimate_cost(sell_qty * curr_price)
                trades.append(
                    {
                        "ticker": t,
                        "side": "SELL",
                        "qty": sell_qty,
                        "price": curr_price,
                        "cost": cost,
                    }
                )

    # Buy Loop
    for t, w in target_weights.items():
        curr_price = kis.get_us_current_price(t, exchange=get_exchange_code(t))
        if not curr_price:
            continue

        current_qty = current_holdings.get(t, 0)
        current_val = current_qty * curr_price
        target_val = total_equity_base * w
        diff_val = target_val - current_val

        if diff_val > 50:
            buy_qty = int(diff_val / curr_price)
            if buy_qty > 0:
                cost = cost_model.estimate_cost(buy_qty * curr_price)
                # Risk limit: max 20%
                if (buy_qty * curr_price) > (total_equity_base * 0.20):
                    buy_qty = int((total_equity_base * 0.20) / curr_price)

                if buy_qty > 0:
                    trades.append(
                        {
                            "ticker": t,
                            "side": "BUY",
                            "qty": buy_qty,
                            "price": curr_price,
                            "cost": cost,
                        }
                    )

    # 5. Execution
    print(f"\n📝 Generated {len(trades)} Trades:")
    for trade in trades:
        print(
            f"   {trade['side']} {trade['ticker']} {trade['qty']} shares @ ${trade['price']:.2f}"
        )

    if not dry_run:
        # execute
        for trade in trades:
            xc = get_exchange_code(trade["ticker"])

            # Check trade limit
            if not trade_limiter.check_and_increment("mama_lite"):
                print(
                    f"   ⚠️  Cannot execute {trade['side']} for {trade['ticker']}: Daily trade limit reached"
                )
                send_discord_msg(
                    config,
                    "⚠️ [MAMA Lite] Trade Limit Blocked",
                    f"{trade['ticker']} {trade['side']} 차단: 일일 거래 한도 도달",
                    color=0xFFA500,
                )
                continue

            res = executor.create_us_order(
                ticker=trade["ticker"],
                side=trade["side"],
                qty=trade["qty"],
                price=trade["price"],
                exchange=xc,
                ord_type="00",
            )

            if res:
                msg = f"{trade['side']} {trade['ticker']} ({trade['qty']}sh) Executed"
                send_discord_msg(config, "MAMA Lite Trade", msg)
                time.sleep(0.5)

    print("\n[MAMA Lite Execution Finished]")


if __name__ == "__main__":
    # Default to DRY RUN for safety integration
    run_mama_lite_execution(dry_run=True)
