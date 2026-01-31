"""
Script: Daily Strategy Signal to Discord
Author: Antigravity
Date: 2026-01-31

Function:
1. Fetch latest data (VTI, ^TNX, ^IRX).
2. Calculate Antigravity V4 Indicators:
   - Trend: MA185 + 3% Buffer (Hysteresis)
   - Macro: Yield Curve (10Y - 3M)
3. Determine Regime & Allocation:
   - Bull: 100% Stock
   - Bear (Normal): 30% Stock / 70% Defensive (Sortino Opt)
   - Bear (Inverted): 100% Defensive (Crisis)
4. Send Report via Discord Webhook.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import yaml
import requests
import os
import datetime
import logging

# Setup Logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to console (Essential for GitHub Actions)
    ],
)
# Add FileHandler only if possible (Local Dev)
try:
    file_handler = logging.FileHandler(os.path.join(log_dir, "daily_signal.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
except Exception:
    pass  # Skip file logging if permission denied or path issue

logger = logging.getLogger("DailySignal")


def load_config():
    # Priority: 1. Env Var, 2. Config File
    config = {}

    # Check current directory
    config_path = os.path.join(os.getcwd(), "signal_mailer", "config.yaml")

    # If not found, try hardcoded dev path (Local Windows)
    if not os.path.exists(config_path):
        config_path = "d:/gg/signal_mailer/config.yaml"

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")

    # 2. Override/Inject from Env Vars (GitHub Actions)
    env_webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if env_webhook:
        if "discord" not in config:
            config["discord"] = {}
        config["discord"]["webhook_url"] = env_webhook
        logger.info("Loaded Webhook URL from Environment Variable.")

    return config


def send_discord_message(webhook_url, title, color, fields, footer_text):
    if not webhook_url:
        logger.error("No webhook provided.")
        return

    embed = {
        "title": title,
        "color": color,  # Integer color code
        "fields": fields,
        "footer": {"text": footer_text},
        "timestamp": datetime.datetime.now().isoformat(),
    }

    payload = {"username": "Antigravity V4 Bot", "embeds": [embed]}

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Discord notification sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send Discord message: {e}")


def get_trend_state(df, ticker, ma_window=185, buffer=0.03):
    """
    Calculate Trend State using MA + Buffer Hysteresis.
    Returns dict with current state, price, ma, bands, etc.
    """
    # Calculate MA and Bands
    series = df[ticker]
    ma = series.rolling(window=ma_window).mean()
    upper = ma * (1 + buffer)
    lower = ma * (1 - buffer)

    # Hysteresis Loop
    states = np.zeros(len(df))
    # Init based on price vs ma
    curr = 1 if series.iloc[0] > ma.iloc[0] else -1

    vals = series.values
    uppers = upper.values
    lowers = lower.values

    for i in range(len(df)):
        if vals[i] > uppers[i]:
            curr = 1
        elif vals[i] < lowers[i]:
            curr = -1
        states[i] = curr

    last_idx = -1
    state = states[last_idx]
    price = vals[last_idx]
    curr_ma = ma.iloc[last_idx]
    curr_upper = upper.iloc[last_idx]
    curr_lower = lower.iloc[last_idx]

    dist_to_upper = (curr_upper - price) / price
    dist_to_lower = (price - curr_lower) / price

    return {
        "ticker": ticker,
        "price": price,
        "state": state,  # 1: Bull, -1: Bear
        "ma": curr_ma,
        "upper": curr_upper,
        "lower": curr_lower,
        "dist_up": dist_to_upper,
        "dist_down": dist_to_lower,
    }


def run_daily_check():
    logger.info("Starting Daily Strategy Check (Multi-Asset)...")

    # 1. Config & Webhook
    config = load_config()
    webhook_url = config.get("discord", {}).get("webhook_url")

    if not webhook_url:
        logger.warning("No Webhook URL found.")

    # 2. Define Tickers
    # Main Portfolio
    main_ticker = "VTI"
    macro_tickers = ["^TNX", "^IRX"]

    # Watchlist (User Request)
    # KOSPI(^KS11), Bitcoin(BTC-USD), Google, Chevron, NVR
    watchlist = {
        "^KS11": "🇰🇷 KOSPI",
        "BTC-USD": "🪙 Bitcoin",
        "GOOGL": "🔎 Google",
        "CVX": "🛢️ Chevron",
        "NVR": "🏠 NVR (Cons)",
    }

    all_tickers = [main_ticker] + macro_tickers + list(watchlist.keys())

    # 3. Download Data
    start_date = (datetime.datetime.now() - datetime.timedelta(days=400)).strftime(
        "%Y-%m-%d"
    )
    end_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )  # +1 for safely getting today/yesterday

    try:
        data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return

    # Flatten Data
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            df = data["Close"].copy()
        else:
            df = data.copy()  # Might differ by version
            # If columns are (Ticker, PriceType), we might need to adjust.
            # Usually yfinance with group_by='ticker' is safer, but default is by column.
            # If MultiIndex with 'Close' level missing, we assume simple columns or handle it.
            pass
    else:
        df = data.copy()

    df = df.ffill()  # Fill missing first

    # 4. Analyze Main Strategy (VTI)
    if main_ticker not in df.columns or "^TNX" not in df.columns:
        logger.error("Critical Data Missing (VTI or Yields)")
        return

    df = (
        df.dropna()
    )  # Drop rows where any data missing (might truncate history for BTC, be careful)
    # Actually, BTC trades weekends, stocks don't. Aligning indexes might drop weekends or leave NaNs.
    # It's better to analyze each series independently or ffill thoroughly.

    # Analyze VTI
    vti_res = get_trend_state(df.dropna(), main_ticker)

    # Analyze Macros
    # Handle Yield Spread
    last_tnx = df["^TNX"].iloc[-1]
    last_irx = df["^IRX"].iloc[-1]
    yield_spread = last_tnx - last_irx

    # Build Main Report Logic
    allocation_text = ""
    color = 0x000000
    regime_name = ""

    if vti_res["state"] == 1:  # Bull
        regime_name = "🚀 BULL MARKET (상승장)"
        color = 0x00FF00  # Green
        allocation_text = "✅ **주식 (Stock): 100%**"
        buffer_msg = f"📉 매도 전환가: ${vti_res['lower']:.2f} ({vti_res['dist_down'] * 100:.2f}%)"
    else:  # Bear
        if yield_spread < 0:  # Crisis
            regime_name = "💀 BEAR + CRISIS (위기)"
            color = 0xFF0000  # Red
            allocation_text = "⛔ 주식 0% / ✅ **현금 100%**"
            buffer_msg = f"📈 매수 전환가: ${vti_res['upper']:.2f} ({vti_res['dist_up'] * 100:.2f}%)"
        else:  # Correction
            regime_name = "🐻 BEAR + NORMAL (조정)"
            color = 0xFFA500  # Orange
            allocation_text = "⚠️ **주식 30%** / ✅ 현금 70%"
            buffer_msg = f"📈 매수 전환가: ${vti_res['upper']:.2f} ({vti_res['dist_up'] * 100:.2f}%)"

    # Fields Construction
    fields = [
        {
            "name": "🚦 메인 포트폴리오 (VTI)",
            "value": f"{regime_name}\n{allocation_text}",
            "inline": False,
        },
        {
            "name": "📉 VTI 가격 / 버퍼",
            "value": f"${vti_res['price']:.2f} / {buffer_msg}",
            "inline": False,
        },
        {
            "name": "🏦 금리차 (10Y-3M)",
            "value": f"{yield_spread:.2f}bp ({'Inverted' if yield_spread < 0 else 'Normal'})",
            "inline": True,
        },
    ]

    # 5. Analyze Watchlist
    watch_text_lines = []

    for ticker, name in watchlist.items():
        if ticker not in df.columns:
            continue

        # Separate DropNA for each asset to handle different trading calendars (Crypto vs Stock)
        asset_series = df[[ticker]].dropna()
        if len(asset_series) < 200:
            continue

        res = get_trend_state(asset_series, ticker)

        # Icon
        icon = "📈" if res["state"] == 1 else "📉"
        # Action
        action = "**BUY**" if res["state"] == 1 else "SELL"

        # Format: 📈 KOSPI: $2500 (BUY)
        line = f"{icon} **{name}**: ${res['price']:,.2f} ({action})"
        watch_text_lines.append(line)

    if watch_text_lines:
        fields.append(
            {
                "name": "🔭 주요 자산 신호 (Watchlist)",
                "value": "\n".join(watch_text_lines),
                "inline": False,
            }
        )

    # Send
    last_date = df.index[-1].strftime("%Y-%m-%d")

    print("-" * 50)
    print(f"Date: {last_date}")
    print(f"Regime: {regime_name}")
    print("Watchlist:")
    for l in watch_text_lines:
        print(l)
    print("-" * 50)

    if webhook_url:
        send_discord_message(
            webhook_url,
            "🔮 Antigravity V4 Daily Signal",
            color,
            fields,
            f"기준일: {last_date} | Powered by Gemini",
        )


if __name__ == "__main__":
    run_daily_check()
