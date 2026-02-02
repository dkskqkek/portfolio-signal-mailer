"""
Script: Daily Strategy Signal to Discord (Premium Dashboard)
Author: Antigravity
Date: 2026-02-01
Description:
    Sends a rich, "At-a-Glance" dashboard to Discord.
    Implements Double-Key V2 Strategy:
    1. Key 1 (Macro): Canary Assets (VWO, BND) 12-month Momentum.
    2. Key 2 (Price): VTI vs 185 MA (Exit < 97%, Entry > MA + 3-day confirm).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import yaml
import requests
import os
import sys
import datetime
import logging
import matplotlib.pyplot as plt
import io
import json

# Setup Logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("DailySignal")


def load_config():
    config = {}
    # Priority: 1. Env Var, 2. Local Config
    config_path = os.path.join(os.getcwd(), "signal_mailer", "config.yaml")
    if not os.path.exists(config_path):
        config_path = "d:/gg/signal_mailer/config.yaml"

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    env_webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if env_webhook:
        if "discord" not in config:
            config["discord"] = {}
        config["discord"]["webhook_url"] = env_webhook

    return config


def fetch_data(tickers):
    logger.info(f"Downloading data for: {tickers}")
    try:
        df = yf.download(tickers, period="2y", progress=False)

        if "Adj Close" in df.columns:
            df = df["Adj Close"]
        elif "Close" in df.columns:
            df = df["Close"]
        else:
            # MultiIndex handling
            if isinstance(df.columns, pd.MultiIndex):
                if "Adj Close" in df.columns.get_level_values(0):
                    df = df.xs("Adj Close", level=0, axis=1)
                elif "Close" in df.columns.get_level_values(0):
                    df = df.xs("Close", level=0, axis=1)

        df = df.ffill()
        return df
    except Exception as e:
        logger.error(f"Data download failed: {e}")
        return pd.DataFrame()


def calculate_double_key_v3(
    df, target="VTI", canary_assets=["VWO", "BND"], risk_asset="^VIX"
):
    """
    Calculates the full state of the Double-Key V3 Strategy.
    Features:
    - Smart Cash (SHY)
    - VIX Panic Filter (>30)
    - Trend Hysteresis (MA185)
    """
    # 1. Macro Key (Canary - VWO/BND)
    # 12-month momentum (252 days)
    canary_mom = df[canary_assets].pct_change(252).iloc[-1]
    vwo_bad = canary_mom["VWO"] <= 0
    bnd_bad = canary_mom["BND"] <= 0

    canary_status = "🟢 SAFE"
    if vwo_bad and bnd_bad:
        canary_status = "🔴 CRISIS (All Bad)"
        risk_level = 2
    elif vwo_bad or bnd_bad:
        canary_status = "🟡 CAUTION (Partial)"
        risk_level = 1
    else:
        risk_level = 0

    # 2. VIX Panic Filter (The "Insurance")
    current_vix = df[risk_asset].iloc[-1]
    vix_panic = current_vix >= 30

    if vix_panic:
        canary_status = f"🔥 PANIC (VIX {current_vix:.1f})"
        risk_level = 3  # Override to Max Risk

    # 3. Price Key (185 MA Tunnel)
    series = df[target]
    ma185 = series.rolling(185).mean()
    lower_band = ma185 * 0.97

    # Hysteresis Logic
    lookback = 400
    subset = df.iloc[-lookback:].copy()
    p_vals = subset[target].values
    ma_vals = subset[target].rolling(185).mean().values
    low_vals = ma_vals * 0.97

    # Init state
    state = 1 if p_vals[0] > low_vals[0] else 0
    days_above = 0

    for i in range(len(subset)):
        if np.isnan(ma_vals[i]):
            continue
        p = p_vals[i]
        ma = ma_vals[i]
        low = low_vals[i]

        # VIX Check roughly? No, VIX filter is instant.
        # But for 'Trend State', we keep simple Price Logic, then Override at the end.
        if state == 1:  # Invested
            if p < low:
                state = 0
                days_above = 0
        else:  # Cash
            if p > ma:
                days_above += 1
            else:
                days_above = 0

            if days_above >= 3:
                state = 1

    # Current Snapshot
    current_price = p_vals[-1]
    current_ma = ma_vals[-1]
    current_low = low_vals[-1]

    dist_ma = (current_price - current_ma) / current_ma
    dist_low = (current_price - current_low) / current_low

    price_key_status = ""
    reentry_info = ""

    if state == 1:
        price_key_status = "🟢 BULL (In Trend)"
        key_details = f"Exit Trigger: ${current_low:.2f} ({dist_low * 100:.1f}%)"
    else:
        price_key_status = "🔴 BEAR (Out)"
        if days_above > 0:
            reentry_info = f"⏳ 확인 중 ({days_above}/3일)"
        else:
            reentry_info = "❌ 돌파 전"
        key_details = (
            f"Target: ${current_ma:.2f} ({dist_ma * 100:.1f}%) | {reentry_info}"
        )

    # 4. Final Allocation Logic (V3)
    # Default: Aggressive (100% Equity)
    # If Risk 1 (Caution): 10% Cash
    # If Risk 2 (Crisis): 25% Cash
    # If State 0 (Bear): 50% Cash (Defensive Base)
    # If VIX Panic (Risk 3): 50% Cash (Force Defense) OR even more?
    # Let's align with V3 backtest: (Trend=0 OR VIX>30) -> 50% Cash

    target_cash = 0
    action = "HOLD"
    reason = "Normal Market"

    # Priority 1: Trend Break or VIX Panic
    if state == 0:
        target_cash = 50
        action = "DEFENSIVE (Bear Trend)"
        reason = "Price < MA185 (Trend Broken)"
    elif vix_panic:
        target_cash = 50
        action = "DEFENSIVE (VIX Panic)"
        reason = f"VIX {current_vix:.1f} > 30 (Crash Insurance)"
    else:
        # Priority 2: Canary Tuning (In Bull Trend)
        if risk_level == 2:  # All Bad
            target_cash = 25
            action = "CAUTION (Macro Bad)"
            reason = "VWO & BND Momentum Negative"
        elif risk_level == 1:  # One Bad
            target_cash = 10
            action = "ALERT (Macro Mixed)"
            reason = "VWO or BND Momentum Negative"
        else:
            target_cash = 0
            action = "AGGRESSIVE (Full Invest)"
            reason = "All Systems Go"

    return {
        "target": target,
        "price": current_price,
        "ma": current_ma,
        "canary_status": canary_status,
        "canary_mom": canary_mom,
        "price_status": price_key_status,
        "price_details": key_details,
        "action": action,
        "target_cash": target_cash,
        "risk_level": risk_level,
        "state": state,
        "reason": reason,
        "vix": current_vix,
    }


def generate_status_chart(df, ticker, ma_window=185, buffer=0.03):
    """
    Generates a trend chart for the last 500 days.
    Returns: BytesIO object of the image.
    """
    # Filter last 2 years (approx 500 days) to keep it readable
    subset = df.iloc[-500:].copy()

    dates = subset.index
    price = subset[ticker]
    ma = subset[ticker].rolling(ma_window).mean()
    lower = ma * (1 - buffer)

    plt.figure(figsize=(10, 5))
    plt.style.use("bmh")  # Clean style

    # 1. Main Lines
    plt.plot(dates, price, label="Price", color="black", alpha=0.7, linewidth=1.5)
    plt.plot(
        dates, ma, label=f"MA{ma_window}", color="orange", linestyle="--", linewidth=1.5
    )
    plt.plot(
        dates, lower, label="Exit Line (-3%)", color="red", linestyle=":", alpha=0.6
    )

    # 2. Fill Areas
    plt.fill_between(dates, lower, ma, color="yellow", alpha=0.1, label="Buffer Zone")

    # 3. Current Point Annotation
    last_date = dates[-1]
    last_price = price.iloc[-1]
    last_ma = ma.iloc[-1]

    # Determine color based on relation to MA
    status_color = "green" if last_price > last_ma else "red"

    plt.scatter(last_date, last_price, color=status_color, s=100, zorder=5)
    plt.annotate(
        f"{last_price:.1f}",
        (last_date, last_price),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color=status_color,
    )

    plt.title(f"{ticker} Trend Check (Price vs MA{ma_window})", fontsize=12)
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return buf


def send_discord_dashboard(
    webhook_url, market_data, portfolio_results, chart_img=None, chart_ticker="VTI"
):
    """
    Sends a premium visual dashboard (V3 Portfolio Edition).
    """
    # 1. Global Market HUD
    vix = portfolio_results[0]["vix"]  # VIX is same for all
    tnx = market_data.get("^TNX", {}).get("price", 0)

    # Global Canary Status (From the first result, as it's shared)
    canary_status = portfolio_results[0]["canary_status"]
    vwo_icon = "✅" if portfolio_results[0]["canary_mom"]["VWO"] > 0 else "❌"
    bnd_icon = "✅" if portfolio_results[0]["canary_mom"]["BND"] > 0 else "❌"
    vix_icon = "🔥" if vix >= 30 else "✅"

    # Header Color based on Global Risk
    # Red if VIX Panic or Canary Crisis
    if vix >= 30 or "CRISIS" in canary_status:
        color = 0xFF0000  # Red
    elif "CAUTION" in canary_status:
        color = 0xFFA500  # Orange
    else:
        color = 0x00FF00  # Green

    hud_text = (
        f"**🌍 Market Health (Key 1)**\n"
        f"• VIX: **{vix_icon} {vix:.1f}**\n"
        f"• Canary: **{canary_status}** (VWO{vwo_icon} BND{bnd_icon})\n"
        f"• 10Y Rate: {tnx:.2f}%"
    )

    # 2. Portfolio Scan (Key 2: Price Trend)
    scan_text = "**📊 Asset Trend (Key 2)**\n"

    for res in portfolio_results:
        ticker = res["target"]
        price = res["price"]
        ma = res["ma"]
        state = res["state"]
        dist_ma = (price - ma) / ma * 100

        # Icon & Status
        if state == 1:
            status_icon = "🟢"
            val_str = f"+{dist_ma:.1f}%"
        else:
            status_icon = "🔴"
            val_str = f"{dist_ma:.1f}%"

        # Re-entry check check
        if state == 0 and "확인 중" in res["price_details"]:
            status_icon = "⏳"

        # Name aliases
        if ticker == "^KS11":
            name = "KOSPI"
        else:
            name = ticker

        scan_text += f"`{name:<6}` {status_icon} ${price:,.2f} ({val_str})\n"

    # 3. Action Recommendation (Global)
    # Based on Global V3 Logic from Proxy (VTI) or Consensus?
    # Usually we follow the Lead Asset (VTI) for "Allocation" advice, or show individual?
    # V3 Logic says: If VIX > 30, Risk Level is Panic.
    # Let's show the recommendation based on VTI (first element) or general macro.

    res0 = portfolio_results[0]
    target_cash = res0["target_cash"]
    action = res0["action"]
    reason = res0["reason"]
    cash_asset = "SHY (Smart Cash)" if target_cash > 0 else "None"

    action_text = (
        f"\n# 📢 Strategy: {action}\n"
        f"> Reason: {reason}\n"
        f"**Rec. Alloc (Based on Market)**:\n"
        f"• Equity: **{100 - target_cash}%**\n"
        f"• Cash ({cash_asset}): **{target_cash}%**"
    )

    embed = {
        "title": f"🛡️ Antigravity V3 Portfolio ({datetime.date.today()})",
        "description": f"{hud_text}\n\n{scan_text}{action_text}",
        "color": color,
        "footer": {"text": "V3 Logic: MA185 Trend + VIX/Canary Macro"},
        "timestamp": datetime.datetime.now().isoformat(),
    }

    if chart_img:
        embed["image"] = {"url": "attachment://chart.png"}

    try:
        if chart_img:
            files = {"file": ("chart.png", chart_img, "image/png")}
            payload = {"username": "Antigravity HQ", "embeds": [embed]}
            requests.post(
                webhook_url, files=files, data={"payload_json": json.dumps(payload)}
            )
        else:
            requests.post(
                webhook_url, json={"username": "Antigravity HQ", "embeds": [embed]}
            )
        logger.info("Dashboard sent.")
    except Exception as e:
        logger.error(f"Failed to send: {e}")


def run_daily_check():
    logger.info("Generating Premium Dashboard (V3 Portfolio)...")
    config = load_config()
    webhook_url = config.get("discord", {}).get("webhook_url")

    if not webhook_url:
        logger.warning("No webhook URL.")
        return

    # Assets
    portfolio = ["^KS11", "VXUS", "GOOGL", "CVX", "NVR", "VTI"]
    # ^KS11 = KOSPI, VXUS = Intl, GOOGL, CVX, NVR

    canaries = ["VWO", "BND"]
    macros = ["^VIX", "^TNX", "SHY"]

    all_tickers = list(set(portfolio + canaries + macros))

    df = fetch_data(all_tickers)
    if df.empty:
        return

    # Sub-data for HUD
    market_data = {}
    for t in macros:
        if t in df.columns:
            market_data[t] = {"price": df[t].iloc[-1]}

    # Calculate V3 for EACH asset
    results = []
    for ticker in portfolio:
        if ticker not in df.columns:
            logger.warning(f"Ticker {ticker} not found in data.")
            continue

        res = calculate_double_key_v3(df, ticker, canaries)
        results.append(res)

    if not results:
        return

    # Generate Chart for the MAIN asset (VTI)
    chart_ticker = "VTI"
    if chart_ticker not in df.columns and results:
        chart_ticker = results[0]["target"]  # Fallback

    logger.info(f"Generating Chart for {chart_ticker}...")
    chart_bytes = generate_status_chart(df, chart_ticker)

    # Send
    send_discord_dashboard(webhook_url, market_data, results, chart_bytes, chart_ticker)


if __name__ == "__main__":
    run_daily_check()
