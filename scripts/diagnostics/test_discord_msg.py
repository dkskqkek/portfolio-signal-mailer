# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys
import requests
from datetime import datetime

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiscordTest")


def test_discord_alert():
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    webhook_url = config.get("discord", {}).get("webhook_url")
    if not webhook_url:
        print("❌ Webhook URL이 설정되지 않았습니다.")
        return

    print(f"🔗 테스팅 웹훅 URL: {webhook_url[:30]}...")

    payload = {
        "embeds": [
            {
                "title": "🔔 Antigravity 디스코드 알림 테스트",
                "description": "현재 마켓 스캔 및 주문 알림 기능이 정상적으로 연결되었습니다.\n\n**테스트 시각**: "
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "color": 0x3498DB,
                "footer": {"text": "Hybrid Alpha Execution System"},
            }
        ]
    }

    try:
        r = requests.post(webhook_url, json=payload, timeout=5)
        if r.status_code in [200, 204]:
            print("✅ 디스코드 알림 전송 성공!")
        else:
            print(f"❌ 전송 실패 (Status: {r.status_code}): {r.text}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    test_discord_alert()
