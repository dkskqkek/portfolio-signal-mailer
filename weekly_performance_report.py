# -*- coding: utf-8 -*-
"""
Weekly Performance Report Generator
Analyzes equity_log.csv and generates performance metrics.
Automatically syncs with live account before analysis.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

# Add current dir to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor

logger = logging.getLogger("WeeklyReport")
logging.basicConfig(level=logging.INFO)


def send_discord_msg(config, title, message, color=0x00FF00):
    """Send Discord webhook message."""
    webhook_url = config.get("discord", {}).get("webhook_url")
    if not webhook_url:
        logger.warning("No Discord webhook configured")
        return

    embed = {
        "title": title,
        "description": message,
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
    }
    payload = {"embeds": [embed]}

    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Discord notice failed: {e}")


def calculate_performance_metrics(equity_series: pd.Series):
    """
    Calculate CAGR, MDD, Sharpe from equity series.

    Args:
        equity_series: Daily equity values (Series with DatetimeIndex)

    Returns:
        dict with metrics
    """
    if len(equity_series) < 2:
        return {"CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0, "Total_Return": 0.0, "Days": 0}

    # Sort by date
    equity_series = equity_series.sort_index()

    # Days
    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = days / 365.25 if days > 0 else 1

    # Total Return
    start_val = equity_series.iloc[0]
    end_val = equity_series.iloc[-1]
    total_return = (end_val / start_val - 1) if start_val > 0 else 0.0

    # CAGR
    cagr = (
        (end_val / start_val) ** (1 / years) - 1 if years > 0 and start_val > 0 else 0.0
    )

    # MDD (Maximum Drawdown)
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    mdd = drawdown.min()

    # Sharpe Ratio (Daily returns annualized)
    daily_returns = equity_series.pct_change().dropna()
    if len(daily_returns) > 1:
        sharpe = (
            (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            if daily_returns.std() > 0
            else 0.0
        )
    else:
        sharpe = 0.0

    return {
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Total_Return": total_return,
        "Days": days,
    }


def sync_latest_equity(config):
    """
    Sync latest equity from live account to equity_log.csv.
    Prevents duplicate entries for the same day.
    """
    log_file = os.path.join(current_dir, "data", "equity_log.csv")

    try:
        # Initialize KIS API
        kis = KISAPIWrapper(config["kis"])
        executor = OrderExecutor(kis)

        # Get detailed breakdown (same as log_daily_equity.py)
        kr_holdings = executor.get_balance()
        kr_stocks = sum(float(h.get("evlu_amt", 0)) for h in kr_holdings)
        kr_cash = executor.get_cash()
        kr_total = kr_stocks + kr_cash

        us_holdings = executor.get_us_balance()
        us_stocks_usd = sum(float(h.get("frcr_evlu_amt2", 0)) for h in us_holdings)
        us_cash_usd = executor.get_us_cash()
        us_total_usd = us_cash_usd + us_stocks_usd

        # Approximate USD to KRW
        exch_rate = 1474.0
        us_total_krw = us_total_usd * exch_rate

        total_krw = kr_total + us_total_krw

        # Current timestamp
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M")

        # Load existing log
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)

            # Check if today already logged
            if today_str in df["Date"].values:
                # Update today's entry
                df.loc[
                    df["Date"] == today_str,
                    [
                        "Time",
                        "Total_KRW",
                        "KR_Stocks_KRW",
                        "KR_Cash_KRW",
                        "US_Stocks_USD",
                        "US_Cash_USD",
                        "US_Total_KRW",
                    ],
                ] = [
                    time_str,
                    total_krw,
                    kr_stocks,
                    kr_cash,
                    us_stocks_usd,
                    us_cash_usd,
                    us_total_krw,
                ]
            else:
                # Append new entry
                new_row = pd.DataFrame(
                    {
                        "Date": [today_str],
                        "Time": [time_str],
                        "Total_KRW": [total_krw],
                        "KR_Stocks_KRW": [kr_stocks],
                        "KR_Cash_KRW": [kr_cash],
                        "US_Stocks_USD": [us_stocks_usd],
                        "US_Cash_USD": [us_cash_usd],
                        "US_Total_KRW": [us_total_krw],
                    }
                )
                df = pd.concat([df, new_row], ignore_index=True)
        else:
            # Create new log file
            df = pd.DataFrame(
                {
                    "Date": [today_str],
                    "Time": [time_str],
                    "Total_KRW": [total_krw],
                    "KR_Stocks_KRW": [kr_stocks],
                    "KR_Cash_KRW": [kr_cash],
                    "US_Stocks_USD": [us_stocks_usd],
                    "US_Cash_USD": [us_cash_usd],
                    "US_Total_KRW": [us_total_krw],
                }
            )

        # Save
        df.to_csv(log_file, index=False)
        logger.info(f"✅ Synced latest equity: {total_krw:,.0f}원")

    except Exception as e:
        logger.error(f"Failed to sync latest equity: {e}")
        # Continue anyway with existing data


def generate_weekly_report():
    """Generate weekly performance report from equity_log.csv."""
    # Priority: 1. Env, 2. Config File
    config = {}
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Env Var Override
    env_webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if env_webhook:
        if "discord" not in config:
            config["discord"] = {}
        config["discord"]["webhook_url"] = env_webhook

    # Sync latest data from live account
    # Sync latest data from live account (Only if KIS config exists)
    if "kis" in config:
        logger.info("🔄 Syncing latest account balance...")
        try:
            sync_latest_equity(config)
        except Exception as e:
            logger.warning(f"Sync failed (Non-fatal): {e}")
    else:
        logger.info("No KIS config found. Skipping live sync (using cached logs).")

    log_file = os.path.join(current_dir, "data", "equity_log.csv")

    if not os.path.exists(log_file):
        logger.error(f"Equity log not found: {log_file}")
        return

    # Load data
    df = pd.read_csv(log_file)

    # Combine Date and Time into Timestamp
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.set_index("Timestamp").sort_index()

    # Rename columns for consistency
    if "KR_Stocks_KRW" in df.columns:
        df = df.rename(columns={"KR_Stocks_KRW": "KR_Stocks", "KR_Cash_KRW": "KR_Cash"})

    # Calculate KR_Total if not present
    if (
        "KR_Total" not in df.columns
        and "KR_Stocks" in df.columns
        and "KR_Cash" in df.columns
    ):
        df["KR_Total"] = df["KR_Stocks"] + df["KR_Cash"]

    if len(df) < 2:
        logger.warning("Not enough data for weekly report (need at least 2 days)")
        return

    # Calculate metrics for different periods
    now = datetime.now()

    # This week (last 7 days)
    week_ago = now - timedelta(days=7)
    week_data = df[df.index >= week_ago]

    # All time
    all_time_data = df

    # Calculate metrics
    week_metrics = calculate_performance_metrics(week_data["Total_KRW"])
    all_time_metrics = calculate_performance_metrics(all_time_data["Total_KRW"])

    # Latest equity
    latest = df.iloc[-1]

    # Build report message
    msg = f"""
📊 **주간 성과 리포트** (최근 7일)

**현재 자산**
━━━━━━━━━━━━━━━━━━━━
💰 총 자산: {latest["Total_KRW"]:,.0f}원
🇰🇷 한국: {latest["KR_Total"]:,.0f}원
🇺🇸 미국: {latest["US_Total_KRW"]:,.0f}원

**이번 주 성과**
━━━━━━━━━━━━━━━━━━━━
📈 수익률: {week_metrics["Total_Return"]:.2%}
📉 MDD: {week_metrics["MDD"]:.2%}
📊 Sharpe: {week_metrics["Sharpe"]:.2f}

**누적 성과** ({all_time_metrics["Days"]}일)
━━━━━━━━━━━━━━━━━━━━
📈 CAGR: {all_time_metrics["CAGR"]:.2%}
📉 MDD: {all_time_metrics["MDD"]:.2%}
📊 Sharpe: {all_time_metrics["Sharpe"]:.2f}
💵 누적 수익률: {all_time_metrics["Total_Return"]:.2%}
"""

    # Determine color based on weekly performance
    color = 0x00FF00 if week_metrics["Total_Return"] >= 0 else 0xFF6B6B

    send_discord_msg(config, "📊 [Antigravity] 주간 성과 리포트", msg, color=color)
    logger.info("Weekly report sent successfully")

    print(msg)


if __name__ == "__main__":
    generate_weekly_report()
