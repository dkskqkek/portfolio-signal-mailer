# -*- coding: utf-8 -*-
import sys
import os
import logging
import traceback
from datetime import datetime, time
import yaml
from typing import Any, Dict

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

from signal_mailer.signal_detector import SignalDetector
from signal_mailer.notification.discord_webhook import DiscordWebhook

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
# [Infrastructure Hardening] Telegram Notification Config (Deprecated in favor of Discord)
# TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE"
# TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
# USE_TELEGRAM = False


def load_config() -> Dict[str, Any]:
    config_path = os.path.join(project_root, "signal_mailer", "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logging.warning(f"Config load failed: {e}")
        return {}


def check_time_safety() -> None:
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


def run_analysis() -> None:
    logging.info("--- [START] Antigravity Signal Analysis (v4.1 Live) ---")
    check_time_safety()

    try:
        detector = SignalDetector()
        logging.info("Analyzing market regimes and calculating MDD...")

        signal_info = detector.detect()

        # [v4.1 Adaptation] Mapping New Schema
        status = signal_info.get("signal", "UNKNOWN")

        emoji = "🟢" if status == "NORMAL" else "🔴" if status == "DANGER" else "🟡"

        # Formatting Output
        report = []
        report.append(f"Result: {emoji} {status}")
        report.append(f"Date: {signal_info.get('date', 'N/A')}")

        # Market Context
        qqq_price = signal_info.get("qqq_price", 0.0)
        report.append("\n[Market Context]")
        report.append(f"QQQ Price: ${qqq_price:.2f}")

        if "dist_sma110" in signal_info:
            report.append(f"Dist SMA110: {signal_info['dist_sma110'] * 100:+.2f}%")
        if "dist_sma250" in signal_info:
            report.append(f"Dist SMA250: {signal_info['dist_sma250'] * 100:+.2f}%")

        # VIX
        report.append(f"VIX: {signal_info.get('vix', 'N/A')}")

        # Index Sniper
        sniper_state = signal_info.get("sniper_signal")
        sniper_str = sniper_state.current_state if sniper_state else "N/A"
        report.append(f"Index Sniper: {sniper_str}")

        # Council
        report.append("\n[Dynamic Allocation]")
        report.append(f"Verdict: {signal_info.get('council_verdict', 'N/A')}")
        report.append(f"Discount: {signal_info.get('council_discount', 1.0)}")

        # Action Plan
        report.append("\n[Recommendations]")
        report.append(signal_info.get("action_plan", "No plan generated."))

        # [NEW] Phase 6: KR Stock Scanner
        config = load_config()
        kr_candidates = []
        if "kis" in config:
            try:
                from signal_mailer.kis_api_wrapper import KISAPIWrapper
                from signal_mailer.kr_stock_scanner import KRStockScanner

                logging.info("Initializing KR Stock Scanner (Hybrid Alpha)...")
                kis_wrapper = KISAPIWrapper(config["kis"])
                scanner = KRStockScanner(kis_wrapper)
                kr_candidates = scanner.scan_full_market(
                    limit=200
                )  # Scan top 200 active stocks

                signal_info["kr_candidates"] = kr_candidates
                logging.info(f"Scan complete: Found {len(kr_candidates)} candidates.")
            except Exception as scan_err:
                logging.error(f"Stock Scanner failed: {scan_err}")

        final_report = "\n".join(report)
        logging.info("\n" + final_report)

        # Notify
        discord_cfg = config.get("discord", {})

        if discord_cfg.get("enabled", False):
            webhook_url = discord_cfg.get("webhook_url")
            if webhook_url and "YOUR_DISCORD" not in webhook_url:
                notifier = DiscordWebhook(webhook_url)
                notifier.send_signal_report(signal_info)
                logging.info("Discord notification sent.")
            else:
                logging.warning("Discord enabled but URL not configured in config.yaml")

    except Exception as e:
        err_detail = traceback.format_exc()
        logging.error(f"System Crash: {e}\n{err_detail}")

        # Try to notify via Discord if possible
        try:
            config = load_config()
            discord_cfg = config.get("discord", {})
            if discord_cfg.get("enabled", False):
                webhook = discord_cfg.get("webhook_url")
                if webhook and "YOUR_DISCORD" not in webhook:
                    # Simple error notification via class
                    notifier = DiscordWebhook(webhook)
                    notifier.send_signal_report(
                        {
                            "signal": "ERROR",
                            "action_plan": f"🚨 **SYSTEM CRASH**\n```{e}```",
                            "qqq_price": 0.0,
                        }
                    )
        except Exception:
            pass

    logging.info("--- [END] Analysis Complete ---\n")


if __name__ == "__main__":
    run_analysis()
