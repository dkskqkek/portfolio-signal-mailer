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


def run_daily_check():
    logger.info("Starting Daily Strategy Check...")

    # 1. Config
    config = load_config()
    webhook_url = config.get("discord", {}).get("webhook_url")

    if not webhook_url:
        print("⚠️ Warning: No Discord Webhook URL found in config.yaml")
        # For testing, we proceed but don't fail hard if user just wants dry run output

    # 2. Data
    # Need enough history for MA185 + Lag
    start_date = (datetime.datetime.now() - datetime.timedelta(days=400)).strftime(
        "%Y-%m-%d"
    )
    end_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime(
        "%Y-%m-%d"
    )

    tickers = ["VTI", "^TNX", "^IRX"]

    logger.info(f"Downloading data from {start_date}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return

    # Flatten
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            df = data["Close"].copy()
        else:
            df = data.copy()
    else:
        df = data.copy()

    df = df.ffill().dropna()

    if df.empty:
        logger.error("Dataframe is empty after download.")
        return

    # 3. Indicators
    # MA185
    df["MA185"] = df["VTI"].rolling(window=185).mean()

    # Buffer Band
    buffer = 0.03
    df["Upper"] = df["MA185"] * (1 + buffer)
    df["Lower"] = df["MA185"] * (1 - buffer)

    # Macro Spread
    # Handle older data where TNX/IRX might be missing slightly, assume ffill did job
    df["Spread"] = df["^TNX"] - df["^IRX"]

    df.dropna(inplace=True)

    # 4. Determine State (Hysteresis)
    # We need to iterate to find current state
    # 1 = Bull, -1 = Bear
    states = np.zeros(len(df))
    # Init state
    current_state = 1 if df["VTI"].iloc[0] > df["MA185"].iloc[0] else -1

    prices = df["VTI"].values
    uppers = df["Upper"].values
    lowers = df["Lower"].values

    for i in range(len(df)):
        p = prices[i]
        if p > uppers[i]:
            current_state = 1
        elif p < lowers[i]:
            current_state = -1
        # Else hold previous state
        states[i] = current_state

    df["State"] = states

    # 5. Latest Status
    last_row = df.iloc[-1]
    last_date = df.index[-1].strftime("%Y-%m-%d")

    curr_state = last_row["State"]
    curr_spread = last_row["Spread"]
    curr_price = last_row["VTI"]
    curr_ma = last_row["MA185"]
    curr_upper = last_row["Upper"]
    curr_lower = last_row["Lower"]

    dist_to_upper = (curr_upper - curr_price) / curr_price
    dist_to_lower = (curr_price - curr_lower) / curr_price

    # Allocation Logic (Antigravity V4 Sortino Optimized)
    allocation_text = ""
    color = 0x000000
    regime_name = ""

    if curr_state == 1:
        # BULL
        regime_name = "🚀 BULL MARKET (상승장)"
        color = 0x00FF00  # Green
        allocation_text = "✅ **주식 (Stock): 100%**\n🛡️ 현금 (Defensive): 0%"

        # Buffer Info
        buffer_msg = f"📉 매도 전환가: ${curr_lower:.2f} (동공지진까지 {dist_to_lower * 100:.2f}% 남음)"

    else:
        # BEAR
        if curr_spread < 0:
            # INVERTED (CRISIS)
            regime_name = "💀 BEAR + INVERTED (금융 위기)"
            color = 0xFF0000  # Red
            allocation_text = "⛔ 주식 (Stock): 0%\n✅ **현금/달러 (Defensive): 100%**"
            buffer_msg = f"📈 매수 전환가: ${curr_upper:.2f} (회복까지 {dist_to_upper * 100:.2f}% 남음)"

        else:
            # NORMAL BEAR (CORRECTION)
            regime_name = "🐻 BEAR + NORMAL (단순 하락장)"
            color = 0xFFA500  # Orange
            allocation_text = (
                "⚠️ **주식 (Stock): 30%**\n✅ **현금/달러 (Defensive): 70%**"
            )
            buffer_msg = f"📈 매수 전환가: ${curr_upper:.2f} (회복까지 {dist_to_upper * 100:.2f}% 남음)"

    # message fields
    fields = [
        {"name": "📅 기준일 (Data Date)", "value": last_date, "inline": True},
        {"name": "📊 현재 주가 (VTI)", "value": f"${curr_price:.2f}", "inline": True},
        {"name": "📉 MA 185", "value": f"${curr_ma:.2f}", "inline": True},
        {
            "name": "🚥 현재 상태 (Regime)",
            "value": f"**{regime_name}**",
            "inline": False,
        },
        {
            "name": "💼 추천 비중 (Allocation)",
            "value": allocation_text,
            "inline": False,
        },
        {"name": "📏 버퍼 현황 (Buffer Status)", "value": buffer_msg, "inline": False},
        {
            "name": "🏦 수익률 곡선 (Yield Spread)",
            "value": f"{curr_spread:.2f}bp ({'Inverted!' if curr_spread < 0 else 'Normal'})",
            "inline": True,
        },
    ]

    print("-" * 50)
    print(f"Date: {last_date}")
    print(f"Regime: {regime_name}")
    print(f"Alloc: {allocation_text.replace('**', '').replace(chr(10), ', ')}")
    print("-" * 50)

    if webhook_url:
        send_discord_message(
            webhook_url,
            "🔮 Antigravity V4 Daily Signal",
            color,
            fields,
            "Powered by Gemini & Antigravity Engine",
        )
    else:
        print("Skipping Discord output (No URL).")


if __name__ == "__main__":
    run_daily_check()
