"""
Regime History 누적 테스트 - 5일간 연속 호출
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mama_lite_predictor import MAMAPredictor
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("REGIME HISTORY ACCUMULATION TEST")
    print("=" * 80)

    predictor = MAMAPredictor()

    # 최근 30일 데이터 로드
    end_date = datetime.now()
    start_date = end_date - timedelta(days=50)

    print(f"\nDownloading data from {start_date.date()} to {end_date.date()}...")
    data = yf.download(
        ["^VIX", "^TNX", "SPY"], start=start_date, end=end_date, progress=False
    )["Close"]

    # 최근 5일 시뮬레이션
    dates = data.index[-5:]

    print(f"\nSimulating 5 consecutive days:\n")

    for i, date in enumerate(dates, 1):
        print(f"--- Day {i}: {date.date()} ---")

        # 해당 날짜까지의 데이터만 전달
        data_until_today = data.loc[:date]

        # regime history 업데이트
        predictor.update_regime_history(df=data_until_today)

        print(f"  Regime History: {predictor.regime_history}")
        print(f"  History Length: {len(predictor.regime_history)}")

        if len(predictor.regime_history) > 0:
            bull_count = sum(
                1 for r in predictor.regime_history if r == predictor.bull_regime_id
            )
            bull_prob = bull_count / len(predictor.regime_history)
            print(f"  Bull Probability: {bull_prob:.2%}")
        print()

    print("=" * 80)
    print(f"Final Regime History: {predictor.regime_history}")
    print(f"Expected: List of length 5 (smoothing_window)")
    print(f"Actual Length: {len(predictor.regime_history)}")
    print("=" * 80)
