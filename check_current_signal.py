# -*- coding: utf-8 -*-
import sys
import os
import logging
import traceback
from datetime import datetime, time
import requests

# [Infrastructure Hardening] Resolve Path Dependencies
# Add project root to sys.path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = (
    current_dir
    if os.path.exists(os.path.join(current_dir, "signal_mailer"))
    else os.path.dirname(current_dir)
)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from signal_mailer.signal_detector import SignalDetector
except ImportError:
    # Fallback for manual executions in specific subfolders
    sys.path.append("d:/gg")
    from signal_mailer.signal_detector import SignalDetector

# [Infrastructure Hardening] Persistent Logging Setup
log_dir = os.path.join(project_root, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(
    log_dir, f"signal_audit_{datetime.now().strftime('%Y%m')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# [Infrastructure Hardening] Telegram Notification Config
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE"  # FIXME: User to provide
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"  # FIXME: User to provide
USE_TELEGRAM = False  # Set to True after configuration


def send_telegram(message):
    if not USE_TELEGRAM:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
    except Exception as e:
        logging.warning(f"Telegram failed: {e}")


def check_time_safety():
    """Safety Guard: Warn if run during market noise or stale periods."""
    now_kst = datetime.now()
    now_time = now_kst.time()

    # 09:00 ~ 15:30 KST (KOSPI Market Hours) - Potential Noise
    if time(9, 0) <= now_time <= time(15, 30):
        logging.warning(
            "⚠️ MARKET OPEN: Running during KOSPI market hours. Signals may reflect intraday noise."
        )

    # Early Morning before yfinance data stabilizes (e.g., 04:00~05:30)
    if time(4, 0) <= now_time <= time(5, 45):
        logging.warning(
            "⚠️ DATA STALE: US market just closed. yfinance data may be inconsistent. Recommend waiting until 06:00 KST."
        )


def run_analysis():
    logging.info("--- [START] Antigravity Signal Analysis (v3.1 Hardened) ---")
    check_time_safety()

    try:
        detector = SignalDetector()
        logging.info("Analyzing market regimes and calculating MDD...")

        signal_info = detector.detect()

        status = signal_info["status_label"]
        emoji = "🟢" if status == "NORMAL" else "🔴" if status == "DANGER" else "🟡"
        if status == "EMERGENCY (STOP)":
            emoji = "🛑"

        s1, s2 = signal_info["s_params"]

        # Formatting Output
        report = []
        report.append(f"Result: {emoji} {status}")
        report.append(f"Condition: {signal_info['regime']}")
        report.append(
            f"Emergency Mode: {'🚨 ACTIVE (CASH-OUT)' if signal_info['is_emergency'] else '🟢 STANDBY'}"
        )
        report.append(f"Internal MDD: {signal_info['calculated_mdd'] * 100:.2f}%")

        report.append(f"\n[Dynamic Allocation]")
        report.append(f"KST Date: {signal_info['date'].strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(
            f"K-Premium Drivers: DXY {signal_info['dxy_90d'] * 100:+.1f}%, KOSPI {signal_info['kospi_126d'] * 100:+.1f}%"
        )
        report.append(
            f"Target: KRW {signal_info['krw_ratio'] * 100:.0f}% / USD {(1 - signal_info['krw_ratio']) * 100:.0f}%"
        )

        report.append(f"\n[Technical Context]")
        report.append(f"QQQ Price: ${signal_info['qqq_price']:.2f}")
        report.append(
            f"SMA {s1}/{s2}: ${signal_info['ma_fast']:.0f} / ${signal_info['ma_slow']:.0f}"
        )

        report.append(f"\n[Recommendations]")
        if status == "NORMAL":
            report.append(
                f"- USD Leg ({(1 - signal_info['krw_ratio']) * 100:.0f}%): QLD / QQQ Split"
            )
            report.append(
                f"- KRW Leg ({signal_info['krw_ratio'] * 100:.0f}%): KOSPI / KRX Gold-Spot"
            )
        elif status == "EMERGENCY (STOP)":
            report.append("- 🚨 ACTIONS REQUIRED: SELL ALL RISK ASSETS -> MOVE TO CASH")
        else:
            report.append(f"- DEFENSIVE: {', '.join(signal_info['defensive_assets'])}")

        final_report = "\n".join(report)
        logging.info("\n" + final_report)

        # Notify
        if USE_TELEGRAM:
            send_telegram(f"🔔 *Antigravity Signal Report*\n{final_report}")

    except Exception as e:
        err_detail = traceback.format_exc()
        logging.error(f"System Crash: {e}\n{err_detail}")
        if USE_TELEGRAM:
            send_telegram(f"❌ *System Error Alert*\n{e}")

    logging.info("--- [END] Analysis Complete ---\n")


if __name__ == "__main__":
    run_analysis()
