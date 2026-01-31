# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.header import Header

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.kr_stock_scanner import KRStockScanner

logging.basicConfig(level=logging.INFO)


def send_scan_email():
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 1. Run Scan
    kis_wrapper = KISAPIWrapper(config["kis"])
    scanner = KRStockScanner(kis_wrapper)
    candidates = scanner.scan_full_market(limit=200)

    if not candidates:
        print("No candidates found.")
        return

    # 2. Format Email
    body = "💎 [Antigravity] KR Hybrid Alpha 스캔 결과\n"
    body += "------------------------------------------\n"
    body += "Logic: (Close > SMA_5) AND (ROC_1 > 0)\n"
    body += f"대상: 거래대금 상위 200 종목\n\n"

    for i, c in enumerate(candidates[:20], 1):
        body += f"{i:2d}. {c['name']} ({c['ticker']})\n"
        body += f"    └ 현재가: {c['price']:,}원 | 1일 수익률: {c['roc_1'] * 100:+.1f}% | 5일선 이격: {c['dist_sma'] * 100:+.1f}%\n"

    body += "\n------------------------------------------\n"
    body += "본 메일은 디스코드 봇 오류로 인해 수동 발송되었습니다."

    # 3. Send Email
    email_cfg = config["email"]
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = Header("[Antigravity] KR Market Scan Candidates", "utf-8")
    msg["From"] = email_cfg["sender_email"]
    msg["To"] = email_cfg["recipient_email"]

    try:
        with smtplib.SMTP(email_cfg["smtp_server"], email_cfg["smtp_port"]) as server:
            server.starttls()
            server.login(email_cfg["sender_email"], email_cfg["sender_password"])
            server.send_message(msg)
        print("✅ Email sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


if __name__ == "__main__":
    send_scan_email()
