import yaml
import os
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiscordTest")


def send_discord_test():
    current_dir = os.getcwd()
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    webhook_url = config.get("discord", {}).get("webhook_url")
    if not webhook_url:
        logger.error("No webhook found!")
        return

    payload = {
        "content": "🔔 **Antigravity System Check**\nHybrid Alpha bot connectivity test. If you see this, the notification system is working.",
        "username": "Antigravity Bot",
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        if 200 <= response.status_code < 300:
            logger.info(f"✅ Discord Test Sent! Status: {response.status_code}")
        else:
            logger.error(f"❌ Failed: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"❌ Exception: {e}")


if __name__ == "__main__":
    send_discord_test()
