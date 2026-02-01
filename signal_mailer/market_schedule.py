# -*- coding: utf-8 -*-
"""
Market Scheduler Module
Handles market hours, weekends, and holidays for KR (KOSPI/KOSDAQ) and US (NYSE/NASDAQ).
optimized for 2026.
"""

import datetime
from datetime import time
import logging

logger = logging.getLogger("MarketSchedule")

# 2026 Holidays (YYYY-MM-DD)
HOLIDAYS_KR = {
    "2026-01-01",  # New Year
    "2026-02-16",
    "2026-02-17",
    "2026-02-18",  # Seolnal
    "2026-03-02",  # Samiljeol (Mon for 3/1)
    "2026-05-01",  # Labor Day
    "2026-05-05",  # Children's Day
    "2026-05-25",  # Buddha's Birthday
    "2026-06-06",  # Memorial Day (Sat, check sub?) -> Sub not confirmed, assume safe check
    "2026-08-15",  # Liberation Day (Sat) -> 8/17 Sub?
    "2026-08-17",  # Liberation Day Sub
    "2026-09-24",
    "2026-09-25",  # Chuseok
    "2026-10-03",  # Foundation Day (Sat) -> 10/5 Sub?
    "2026-10-05",  # Foundation Day Sub
    "2026-10-09",  # Hangul Day
    "2026-12-25",  # Christmas
    "2026-12-31",  # Year End Close
}

HOLIDAYS_US = {
    "2026-01-01",  # New Year
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Washington's Birthday
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (Observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
}


def is_holiday(date_obj, market="KR"):
    date_str = date_obj.strftime("%Y-%m-%d")
    if market == "KR":
        return date_str in HOLIDAYS_KR
    elif market == "US":
        return date_str in HOLIDAYS_US
    return False


def is_market_open(market="KR", now=None):
    """
    Check if the specific market is currently open.
    """
    if now is None:
        now = datetime.datetime.now()

    # 1. Weekend Check
    weekday = now.weekday()  # 0=Mon, 6=Sun
    if weekday >= 5:
        return False, "Weekend"

    # 2. Holiday Check
    if is_holiday(now, market):
        return False, "Holiday"

    # 3. Time Check (KST Base)
    curr_time = now.time()

    if market == "KR":
        # KR: 09:00 ~ 15:30
        start = time(9, 0)
        end = time(15, 30)
        if start <= curr_time <= end:
            return True, "Open"
        return False, "Closed (Hours)"

    elif market == "US":
        # US: 23:30 ~ 06:00 (Standard) or 22:30 ~ 05:00 (DST)
        # Simplified Logic for KST:
        # Open if time is > 22:30 OR < 06:00
        # CAUTION: 'now' is KST.
        # If it's 04:00 AM KST (Tue), it's Monday afternoon in NY.
        # We need to check if the NY time is a holiday.

        # Convert KST to EST/EDT (Approx -14 or -13 hours)
        # Simple fix: If current KST time is 00:00~06:00, use yesterday for holiday check.

        check_date = now
        if curr_time < time(9, 0):  # Early morning KST = NY Afternoon (Previous Day)
            check_date = now - datetime.timedelta(days=1)

        # Re-check holiday for US with adjusted date
        if check_date.weekday() >= 5:
            return False, "Weekend (US Time)"
        if is_holiday(check_date, "US"):
            return False, "Holiday (US Time)"

        # Time Check:
        # Standard: 23:30 ~ 06:00
        # DST: 22:30 ~ 05:00
        # Let's be permissive: 22:30 ~ 06:00

        is_night_session = curr_time >= time(22, 30)
        is_morning_session = curr_time <= time(6, 0)

        if is_night_session or is_morning_session:
            return True, "Open"

        return False, "Closed (Hours)"

    return False, "Unknown Market"
