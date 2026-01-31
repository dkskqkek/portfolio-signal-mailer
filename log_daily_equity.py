# -*- coding: utf-8 -*-
"""
Daily Equity Logger
Logs total portfolio equity (KR + US) to CSV for performance tracking.
Run this script twice daily: 09:00 (after KR market open) and 23:30 (after US market close).
"""

import logging
import yaml
import os
import sys
from datetime import datetime
import csv

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EquityLogger")


def log_daily_equity():
    """Log current total equity to CSV file."""
    # Load config
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Initialize
    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)

    # Get current equity
    total_equity_krw = executor.get_total_equity()

    # Get detailed breakdown
    kr_holdings = executor.get_balance()
    kr_stocks_value = sum(float(h.get("evlu_amt", 0)) for h in kr_holdings)
    kr_cash = executor.get_cash()

    us_holdings = executor.get_us_balance()
    us_stocks_value_usd = sum(float(h.get("frcr_evlu_amt2", 0)) for h in us_holdings)
    us_cash_usd = executor.get_us_cash()

    # Approximate USD to KRW
    exch_rate = 1400.0
    us_total_usd = us_cash_usd + us_stocks_value_usd
    us_total_krw = us_total_usd * exch_rate

    # Get timestamp
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    # CSV file path
    log_file = os.path.join(current_dir, "data", "equity_log.csv")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Check if file exists to write header
    file_exists = os.path.exists(log_file)

    # Write to CSV
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "Date",
                    "Time",
                    "Total_KRW",
                    "KR_Stocks_KRW",
                    "KR_Cash_KRW",
                    "US_Stocks_USD",
                    "US_Cash_USD",
                    "US_Total_KRW",
                ]
            )

        writer.writerow(
            [
                date_str,
                time_str,
                f"{total_equity_krw:.0f}",
                f"{kr_stocks_value:.0f}",
                f"{kr_cash:.0f}",
                f"{us_stocks_value_usd:.2f}",
                f"{us_cash_usd:.2f}",
                f"{us_total_krw:.0f}",
            ]
        )

    logger.info(
        f"📊 Equity logged: Total {total_equity_krw:,.0f} KRW (KR: {kr_stocks_value + kr_cash:,.0f}, US: {us_total_krw:,.0f})"
    )
    print(f"✅ Equity log saved to {log_file}")


if __name__ == "__main__":
    log_daily_equity()
