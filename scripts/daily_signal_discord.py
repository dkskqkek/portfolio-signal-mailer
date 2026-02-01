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
import sys
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

    # ... (existing VTI macro logic) ...

    # 4. Analyze Main Strategy (VTI)
    if main_ticker not in df.columns or "^TNX" not in df.columns:
        logger.error("Critical Data Missing (VTI or Yields)")
        return

    # Ensure SPY is present for MAMA Opt
    if "SPY" not in df.columns:
        logger.warning("SPY missing for MAMA Opt, skipping MAMA section.")
        spy_available = False
    else:
        spy_available = True

    df = df.dropna()

    # Analyze VTI (V4)
    vti_res = get_trend_state(df, main_ticker)

    # Analyze Macros
    last_tnx = df["^TNX"].iloc[-1]
    last_irx = df["^IRX"].iloc[-1]
    yield_spread = last_tnx - last_irx

    # V4 Report Logic
    allocation_text = ""
    color = 0x000000
    regime_name = ""

    if vti_res["state"] == 1:  # Bull
        regime_name = "🚀 V4: BULL (상승장)"
        color = 0x00FF00  # Green
        allocation_text = "✅ **주식 100% (VTI)**"
        buffer_msg = (
            f"📉 매도 전환: ${vti_res['lower']:.2f} ({vti_res['dist_down'] * 100:.2f}%)"
        )
    else:  # Bear
        if yield_spread < 0:  # Crisis
            regime_name = "💀 V4: CRISIS (위기)"
            color = 0xFF0000  # Red
            allocation_text = "⛔ 주식 0% / ✅ **현금 100%**"
            buffer_msg = f"📈 매수 전환: ${vti_res['upper']:.2f} ({vti_res['dist_up'] * 100:.2f}%)"
        else:  # Correction
            regime_name = "🐻 V4: CORRECTION (조정)"
            color = 0xFFA500  # Orange
            allocation_text = "⚠️ **주식 30%** / ✅ 현금 70%"
            buffer_msg = f"📈 매수 전환: ${vti_res['upper']:.2f} ({vti_res['dist_up'] * 100:.2f}%)"

    # MAMA Opt Logic
    mama_field = {}
    if spy_available:
        try:
            # Need to append path for signal_mailer imports
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            if os.path.join(parent_dir, "signal_mailer") not in sys.path:
                sys.path.append(os.path.join(parent_dir, "signal_mailer"))

            from signal_mailer.mama_lite_predictor import MAMAPredictor

            # Predictor needs config path
            predictor = MAMAPredictor(
                config_path=os.path.join(parent_dir, "signal_mailer", "config.yaml")
            )

            # 1. AI Prop Check
            # Need to feed data to predictor or use its fetch mechanism?
            # Predictor has fetch_data. Let's use it or calculate features manually.
            # Using predictor.predict_portfolio() is easiest but it fetches data again.
            # Let's calculate manually to be fast and use same DF.

            # But predictor needs specific features.
            # Let's rely on predict_portfolio() but suppress its logging or just use helper?
            # Actually, calculate_gnn_features needs close prices.

            # Simplest: Just re-calculate features here.
            # Features: VIX_Z, TNX_MOM, SPY_MOM

            # Ensure enough history
            if len(df) > 260:
                vix_z = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df[
                    "^VIX"
                ].rolling(252).std()
                tnx_mom = df["^TNX"].pct_change(20)
                spy_mom = df["SPY"].pct_change(60)

                feat = pd.DataFrame(
                    {"vix_z": vix_z, "tnx_mom": tnx_mom, "spy_mom": spy_mom}
                ).dropna()

                if not feat.empty:
                    X_srl = predictor.scaler.transform(feat)
                    regime_labels = predictor.kmeans.predict(X_srl)

                    bull_id = predictor.bull_regime_id
                    is_bull = (regime_labels == bull_id).astype(int)

                    # Rolling 5 mean
                    bull_prob = pd.Series(is_bull).rolling(5).mean().iloc[-1]

                    # 2. Trend Check (SPY MA120)
                    spy_ma120 = df["SPY"].rolling(120).mean().iloc[-1]
                    spy_price = df["SPY"].iloc[-1]
                    trend_bull = spy_price > spy_ma120

                    # 3. Decision
                    ai_bull = bull_prob >= 0.5

                    if ai_bull:
                        mama_status = "🟢 적극 매수 (Buy)"
                        mama_desc = "AI가 상승장을 확신합니다."
                    elif trend_bull:  # AI Bear but Trend Bull
                        mama_status = "🟡 버티기 (Hold)"
                        mama_desc = (
                            "AI는 불안해하지만, 추세(MA120)가 살아있습니다. 매도 금지."
                        )
                    else:  # Both Bear
                        mama_status = "🔴 전량 매도 (Sell)"
                        mama_desc = "AI와 추세 모두 하락을 가리킵니다. 도망치세요."

                    mama_val = f"**{mama_status}**\n• AI 확률: {bull_prob:.0%} (Bull)\n• 추세 확인: {'✅ 살아있음' if trend_bull else '❌ 꺾임'} (Price > MA120)\n• {mama_desc}"

                    mama_field = {
                        "name": "⚡ MAMA Opt (AI + Trend)",
                        "value": mama_val,
                        "inline": False,
                    }
        except Exception as e:
            logger.error(f"MAMA Opt calculation failed: {e}")

    # Fields Construction
    fields = [
        {
            "name": "🛡️ Antigravity V4 (Main)",
            "value": f"{regime_name}\n{allocation_text}\n{buffer_msg}",
            "inline": False,
        }
    ]

    if mama_field:
        fields.append(mama_field)

    fields.append(
        {
            "name": "🏦 금리차 (10Y-3M)",
            "value": f"{yield_spread:.2f}bp ({'Inverted' if yield_spread < 0 else 'Normal'})",
            "inline": True,
        }
    )

    # 5. Analyze Watchlist (Keep as is)
    watch_text_lines = []

    for ticker, name in watchlist.items():
        if ticker not in df.columns:
            continue

        # Separate DropNA for each asset to handle different trading calendars (Crypto vs Stock)
        asset_series = df[[ticker]].dropna()
        if len(asset_series) < 200:
            continue

        res = get_trend_state(asset_series, ticker)

        if res["state"] == 1:
            # Bull -> Show Sell Trigger
            icon = "📈"
            action = "**BUY**"
            trigger_msg = (
                f"📉Switch: ${res['lower']:,.2f} ({res['dist_down'] * 100:.1f}%)"
            )
        else:
            # Bear -> Show Buy Trigger
            icon = "📉"
            action = "SELL"
            trigger_msg = (
                f"📈Switch: ${res['upper']:,.2f} (+{res['dist_up'] * 100:.1f}%)"
            )

        # Format: 📈 KOSPI: $2500 (BUY) | 📉Switch: $2400 (-4.0%)
        line = f"{icon} **{name}**: ${res['price']:,.2f} ({action}) | {trigger_msg}"
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
