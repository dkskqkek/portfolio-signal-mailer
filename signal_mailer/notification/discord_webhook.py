# -*- coding: utf-8 -*-
import requests
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime


class DiscordWebhook:
    """
    Stateless Discord Webhook notifier.
    Sends rich embeds for Antigravity signal reports.
    """

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.headers = {"Content-Type": "application/json"}

    def send_signal_report(self, report_data: Dict[str, Any]) -> bool:
        """
        Constructs and sends a structured embed based on the signal report.
        """
        if not self.webhook_url or "YOUR_DISCORD" in self.webhook_url:
            logging.warning("Discord Webhook URL not configured.")
            return False

        # Extract Data
        signal = report_data.get("signal", "UNKNOWN")
        date_str = report_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        qqq_price = report_data.get("qqq_price", 0.0)
        action_plan = report_data.get("action_plan", "No plan.")

        # Color coding
        color = 0x00FF00  # Green (NORMAL)
        if signal == "DANGER":
            color = 0xFF0000  # Red
        elif signal == "WARNING":
            color = 0xFFFF00  # Yellow

        # Fields
        fields = [
            {
                "name": "Market Context",
                "value": f"QQQ: ${qqq_price:.2f}\nVIX: {report_data.get('vix', 'N/A')}",
                "inline": True,
            },
            {
                "name": "Moving Averages",
                "value": f"SMA110 Diff: {report_data.get('dist_sma110', 0) * 100:+.2f}%\nSMA250 Diff: {report_data.get('dist_sma250', 0) * 100:+.2f}%",
                "inline": True,
            },
            {
                "name": "Index Sniper",
                "value": f"{self._format_sniper(report_data.get('sniper_signal'))}",
                "inline": False,
            },
        ]

        # [NEW] Add Hybrid Alpha Candidates (KR Market)
        kr_candidates = report_data.get("kr_candidates", [])
        if kr_candidates:
            cand_str = ""
            for i, c in enumerate(kr_candidates[:10], 1):
                cand_str += f"{i}. **{c['name']}** ({c['ticker']}) | {c['price']:,}원 (+{c['roc_1'] * 100:.1f}%)\n"

            fields.append(
                {
                    "name": "💎 오늘의 한국 급등주 (Hybrid Alpha)",
                    "value": cand_str if cand_str else "조건 충족 종목 없음",
                    "inline": False,
                }
            )

        # Build Embed
        embed = {
            "title": f"🚀 Antigravity Signal: {signal}",
            "description": f"**Date**: {date_str}\n\n**Action Plan**:\n{action_plan}",
            "color": color,
            "fields": fields,
            "footer": {"text": "Antigravity Quant System v4.1"},
            "timestamp": datetime.now().isoformat(),
        }

        payload = {
            "username": "Antigravity Monitor",
            "avatar_url": "https://i.imgur.com/4M34hi2.png",  # Optional: Robot icon
            "embeds": [embed],
        }

        try:
            response = requests.post(
                self.webhook_url, json=payload, headers=self.headers, timeout=5
            )
            response.raise_for_status()
            logging.info("Discord notification sent successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to send Discord webhook: {e}")
            return False

    def _format_sniper(self, sniper_obj: Any) -> str:
        if not sniper_obj:
            return "N/A"
        # Assuming sniper_obj has .current_state or similar, logic handled here
        # If it's a dict or object, handle gracefully
        if hasattr(sniper_obj, "current_state"):
            return sniper_obj.current_state
        return str(sniper_obj)
