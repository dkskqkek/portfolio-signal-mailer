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


def calculate_double_key_v2(df, target="VTI", canary_assets=["VWO", "BND"]):
    """
    Calculates the full state of the Double-Key V2 Strategy.
    """
    # 1. Macro Key (Canary)
    # 12-month momentum (252 days)
    # Bad if Mom <= 0
    canary_mom = df[canary_assets].pct_change(252).iloc[-1]
    vwo_bad = canary_mom["VWO"] <= 0
    bnd_bad = canary_mom["BND"] <= 0

    canary_status = "🟢 SAFE"
    if vwo_bad and bnd_bad:
        canary_status = "🔴 CRISIS (All Bad)"
        risk_level = 2  # High Risk
    elif vwo_bad or bnd_bad:
        canary_status = "🟡 CAUTION (Partial)"
        risk_level = 1  # Moderate Risk
    else:
        risk_level = 0  # Low Risk

    # 2. Price Key (185 MA Tunnel)
    # Logic: Exit if < 0.97 * MA. Re-entry if > MA for 3 days.
    series = df[target]
    ma185 = series.rolling(185).mean()
    lower_band = ma185 * 0.97

    # We need history to determine current state (Hysteresis)
    # Let's run a quick loop for the last 30 days to determine current state accurately
    lookback = 400  # Surpass 252 for momentum + buffer
    subset = df.iloc[-lookback:].copy()

    p_vals = subset[target].values
    ma_vals = subset[target].rolling(185).mean().values
    low_vals = ma_vals * 0.97

    # Init state (approx)
    state = 1 if p_vals[0] > low_vals[0] else 0
    days_above = 0

    for i in range(len(subset)):
        if np.isnan(ma_vals[i]):
            continue

        p = p_vals[i]
        ma = ma_vals[i]
        low = low_vals[i]

        if state == 1:  # Invested
            if p < low:
                state = 0  # Exit
                days_above = 0
        else:  # Cash
            if p > ma:
                days_above += 1
            else:
                days_above = 0

            if days_above >= 3:
                state = 1  # Re-entry

    # Current Snapshot
    current_price = p_vals[-1]
    current_ma = ma_vals[-1]
    current_low = low_vals[-1]

    # Distances
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

    # 3. Final Decision
    # Cash Logic:
    # Risk 0 (Safe) + Bull = 0% Cash
    # Risk 1 (Caution) + Bull = 10% Cash
    # Risk 2 (Crisis) + Bull = 25% Cash
    # Bear (State 0) = 50% Cash (Minimum) OR 100%?
    # User said: "Exit -> 50% Cash".
    # Wait, usually Bear means we are OUT.
    # If the strategy is "Asset Alloc", maybe 50% is the defensive postion.
    # Let's assume 50% Cash is the "Bear" stance, and the rest depends on Canary.
    # But Price Key is the "Micro" trigger. If Price Key says OUT, we must be defensive.

    target_cash = 0
    action = "HOLD"

    if state == 0:  # Price Key Broken
        target_cash = 50
        action = "DEFENSIVE (50% Cash)"
    else:  # Price Key Bullish
        if risk_level == 2:
            target_cash = 25
        elif risk_level == 1:
            target_cash = 10
        else:
            target_cash = 0

        action = "AGGRESSIVE (Invest)"

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


def send_discord_dashboard(webhook_url, data_dict, market_data, chart_img=None):
    """
    Sends a premium visual dashboard.
    """
    v = data_dict

    # 1. Header Color
    # Green if State=1 and Risk=0. Yellow if Risk>0. Red if State=0.
    if v["state"] == 0:
        color = 0xFF0000  # Red
    elif v["risk_level"] > 0:
        color = 0xFFA500  # Orange
    else:
        color = 0x00FF00  # Green

    # 2. Market HUD
    vix = market_data.get("^VIX", {}).get("price", 0)
    tnx = market_data.get("^TNX", {}).get("price", 0)

    hud_text = f"**{v['target']}**: ${v['price']:.2f} | **VIX**: {vix:.1f} | **10Y**: {tnx:.2f}%"

    # 3. Logic Breakdown
    # Canary Icons
    vwo_icon = "✅" if v["canary_mom"]["VWO"] > 0 else "❌"
    bnd_icon = "✅" if v["canary_mom"]["BND"] > 0 else "❌"

    # Price Icons
    state_icon = "✅" if v["state"] == 1 else "❌"

    logic_text = (
        f"**🔑 Key 1 (Canary)**\n"
        f"• Status: **{v['canary_status']}**\n"
        f"• VWO({vwo_icon}) | BND({bnd_icon})\n\n"
        f"**🔑 Key 2 (Price Tunnel)**\n"
        f"• Status: **{v['price_status']}**\n"
        f"• Logic: {v['price_details']}\n"
        f"• MA185: ${v['ma']:.2f}"
    )

    # 4. Action Box
    action_text = (
        f"# 📢 {v['action']}\n"
        f"**Target Portfolio**:\n"
        f"• Stock: {100 - v['target_cash']}%\n"
        f"• Cash : {v['target_cash']}%"
    )

    embed = {
        "title": f"🔮 Antigravity V4 Signal ({datetime.date.today()})",
        "description": f"{hud_text}\n\n{logic_text}\n\n{action_text}",
        "color": color,
        "footer": {"text": "Double-Key V2 Strategy | Powered by Gemini"},
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Add Image if provided
    if chart_img:
        embed["image"] = {"url": "attachment://chart.png"}

    try:
        if chart_img:
            # Multipart upload for Image + Embed
            files = {"file": ("chart.png", chart_img, "image/png")}
            # Discord requires 'payload_json' field when sending files with embeds
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
    logger.info("Generating Premium Dashboard...")
    config = load_config()
    webhook_url = config.get("discord", {}).get("webhook_url")

    if not webhook_url:
        logger.warning("No webhook URL.")
        return

    # Assets
    main_target = "VTI"
    canaries = ["VWO", "BND"]
    macros = ["^VIX", "^TNX"]
    watchlist = ["BTC-USD", "GC=F"]

    all_tickers = list(set([main_target] + canaries + macros + watchlist))

    df = fetch_data(all_tickers)
    if df.empty:
        return

    # Sub-data for HUD
    market_data = {}
    for t in macros:
        if t in df.columns:
            market_data[t] = {"price": df[t].iloc[-1]}

    # Main Strategy Calc
    # VTI analysis
    vti_res = calculate_double_key_v2(df, main_target, canaries)

    # Generate Chart
    logger.info("Generating Chart...")
    chart_bytes = generate_status_chart(df, main_target)

    # Send
    send_discord_dashboard(webhook_url, vti_res, market_data, chart_bytes)


if __name__ == "__main__":
    run_daily_check()
