# -*- coding: utf-8 -*-
import sys
import os
import traceback
import requests
import yfinance as yf
from datetime import datetime

# [Automated Pipeline] Project Root & Module Loading
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from signal_mailer.signal_detector import SignalDetector
    from signal_mailer.mailer_service import MailerService
except ImportError as e:
    print(f"❌ Module Loading Failed: {e}")
    sys.exit(1)

# [Automated Pipeline] Configuration (Production Ready)
CONFIG = {
    "email": {
        "sender_email": "gamjatangjo@gmail.com",
        "sender_password": "skekcgozubcmjjqq",
        "recipient_email": "gamjatangjo@gmail.com",
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
    },
    "telegram": {
        "use": False,  # DISABLED BY DEFAULT
        "bot_token": "YOUR_BOT_TOKEN",
        "chat_id": "YOUR_CHAT_ID",
    },
    "log_file": os.path.join(current_dir, "logs", "system.log"),
    "history_file": os.path.join(current_dir, "data", "signal_history.json"),
    "debug_mode": False,  # Set to True for testing locally
}


def calculate_market_mdd():
    """Fail-Safe: Calculate QQQ MDD if account info is missing."""
    try:
        data = yf.download("QQQ", period="2y", progress=False)
        if data.empty:
            return 0.0
        # MultiIndex fix if necessary
        close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
        peak = close.cummax()
        drawdown = (close - peak) / peak
        return float(drawdown.iloc[-1])
    except Exception:
        return 0.0


def send_telegram(message):
    """Messenger: Telegram Instant Alert"""
    if not CONFIG["telegram"]["use"]:
        return
    try:
        url = (
            f"https://api.telegram.org/bot{CONFIG['telegram']['bot_token']}/sendMessage"
        )
        requests.post(
            url,
            json={
                "chat_id": CONFIG["telegram"]["chat_id"],
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=5,
        )
    except Exception:
        pass


def run_integrated_system():
    print(
        f"--- 🚀 Antigravity v3.1 Final Operations [{datetime.now().strftime('%Y-%m-%d')}] ---"
    )

    mailer = MailerService(CONFIG)
    detector = SignalDetector()

    try:
        # 1. Load Continuity Context
        prev_status = mailer.get_previous_status()
        market_mdd = calculate_market_mdd()
        mailer.logger.info(
            f"Starting analysis. Prev Status: {prev_status} | Market MDD: {market_mdd * 100:.2f}%"
        )

        # 2. Strategy Execution (The Brain)
        signal_info = detector.detect(
            previous_status=prev_status, current_mdd=market_mdd
        )

        if signal_info.get("error"):
            raise Exception(f"SignalDetector Error: {signal_info.get('reason')}")

        # 3. Persistence (The Archivist)
        # Save before sending to ensure state consistency even if mailing fails.
        mailer.save_history(signal_info)

        # 4. Reporting (The Messenger)
        report = detector.format_signal_report(signal_info, previous_status=prev_status)

        # 4-1. Email (Full Report)
        email_result = mailer.send_email(
            subject=f"[Antigravity] {report['title']}", body_text=report["body"]
        )

        # 4-2. Telegram (Flash Alert)
        tg_emoji = (
            "🚨"
            if signal_info["is_emergency"]
            else "✅"
            if signal_info["status_label"] == "NORMAL"
            else "🔴"
        )
        tg_msg = f"{tg_emoji} *Daily Signal: {signal_info['status_label']}*\nWeight: KRW {signal_info['krw_ratio'] * 100:.0f}% / USD {(1 - signal_info['krw_ratio']) * 100:.0f}%\nMDD: {signal_info['calculated_mdd'] * 100:.2f}%"
        send_telegram(tg_msg)

        print(f"\n✅ Pipeline Execution Complete.")
        print(f"   - Status: {signal_info['status_label']}")
        print(
            f"   - Email: {'Sent' if email_result['success'] else 'Failed (' + email_result['message'] + ')'}"
        )

    except Exception as e:
        error_msg = f"❌ CRITICAL SYSTEM FAILURE:\n{traceback.format_exc()}"
        print(error_msg)
        mailer.logger.critical(error_msg)
        send_telegram(f"🔥 *SYSTEM CRASH ALERT*\n{str(e)}")


if __name__ == "__main__":
    run_integrated_system()
