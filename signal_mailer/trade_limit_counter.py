# -*- coding: utf-8 -*-
"""
Trade Limit Counter
Tracks daily trade count to prevent runaway trading.
Resets automatically at midnight.
"""

import json
import os
from datetime import date
from typing import Dict


class TradeLimitCounter:
    """Track and enforce daily trade limits per strategy."""

    def __init__(
        self, limits_file: str = "data/trade_limits.json", max_daily_trades: int = 10
    ):
        """
        Initialize trade limit counter.

        Args:
            limits_file: Path to JSON file storing trade counts
            max_daily_trades: Maximum trades allowed per strategy per day
        """
        self.limits_file = limits_file
        self.max_daily_trades = max_daily_trades
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create limits file if it doesn't exist."""
        os.makedirs(os.path.dirname(self.limits_file), exist_ok=True)
        if not os.path.exists(self.limits_file):
            self._save_data({})

    def _load_data(self) -> Dict:
        """Load trade count data from file."""
        try:
            with open(self.limits_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_data(self, data: Dict):
        """Save trade count data to file."""
        with open(self.limits_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _get_today_key(self) -> str:
        """Get today's date key (YYYY-MM-DD)."""
        return date.today().isoformat()

    def check_and_increment(self, strategy: str) -> bool:
        """
        Check if strategy can trade today, and increment counter if allowed.

        Args:
            strategy: Strategy name ("hybrid_alpha" or "mama_lite")

        Returns:
            True if trade is allowed, False if limit reached
        """
        data = self._load_data()
        today = self._get_today_key()

        # Initialize today's data if needed
        if today not in data:
            data[today] = {}

        # Get current count for this strategy
        current_count = data[today].get(strategy, 0)

        # Check limit
        if current_count >= self.max_daily_trades:
            return False

        # Increment and save
        data[today][strategy] = current_count + 1
        self._save_data(data)

        return True

    def get_today_count(self, strategy: str) -> int:
        """Get today's trade count for strategy."""
        data = self._load_data()
        today = self._get_today_key()
        return data.get(today, {}).get(strategy, 0)

    def get_remaining(self, strategy: str) -> int:
        """Get remaining trades allowed for strategy today."""
        return max(0, self.max_daily_trades - self.get_today_count(strategy))

    def reset_today(self, strategy: str):
        """Reset today's counter for strategy (for testing/emergency)."""
        data = self._load_data()
        today = self._get_today_key()
        if today in data and strategy in data[today]:
            data[today][strategy] = 0
            self._save_data(data)
