# -*- coding: utf-8 -*-
import sys
import os
from datetime import datetime
import json
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from signal_detector import SignalDetector
from mailer_service import MailerService
from html_generator import generate_html_report


def load_config():
    """Load config from YAML or Env Vars (GitHub Actions friendly)"""
    base_dir = Path(__file__).parent
    config_path = base_dir / "config.yaml"
    config = {"email": {}}

    # 1. Load YAML if exists
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            if "email" not in config:
                config["email"] = {}

    # 2. Override with Env Vars (Priority)
    if os.getenv("SENDER_EMAIL"):
        config["email"]["sender_email"] = os.getenv("SENDER_EMAIL")
    if os.getenv("SENDER_PASSWORD"):
        config["email"]["sender_password"] = os.getenv("SENDER_PASSWORD")
    if os.getenv("RECIPIENT_EMAIL"):
        config["email"]["recipient_email"] = os.getenv("RECIPIENT_EMAIL")

    return config


def main():
    print(f"[{datetime.now()}] Starting Antigravity v4.1 Engine...")

    # 1. Initialize Components
    try:
        config = load_config()
        # api_key for Gemini
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyB37foZBuGH17Vrgv6IXF9_-eeCimZ7HFA")

        detector = SignalDetector(api_key=api_key)
        mailer = MailerService(config)  # Correctly pass config
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # 2. Run Analysis
    try:
        report_data = detector.detect(verbose=True)
        if "error" in report_data:
            print(f"Signal Detection Error: {report_data['error']}")
            return

        # 3. Generate Content
        html_body = generate_html_report(report_data)

        # 4. Dispatch
        subject = f"🌌 [Antigravity] {report_data['signal']} | {report_data['date']}"
        if report_data["signal"] == "DANGER":
            subject = f"🛑 [URGENT] DEFENSIVE MODE ACTIVATED | {report_data['date']}"

        # Send
        # MailerService.send_email returns dict {'success': bool, 'message': str}
        result = mailer.send_email(
            subject, "Please enable HTML to view report.", html_body=html_body
        )

        if result.get("success"):
            print("Email dispatched successfully.")
        else:
            print(f"Email Failed: {result.get('message')}")

    except Exception as e:
        print(f"Runtime Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
