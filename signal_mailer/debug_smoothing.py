"""
Regime Smoothing 디버깅 스크립트
실제 동작 확인을 위한 단순 테스트
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mama_lite_predictor import MAMAPredictor
import logging

# 로깅 레벨 설정
logging.basicConfig(level=logging.INFO, format="%(message)s")

if __name__ == "__main__":
    print("=" * 80)
    print("REGIME SMOOTHING DEBUG TEST")
    print("=" * 80)

    predictor = MAMAPredictor()

    # 5번 연속 예측 (regime_history가 쌓이는지 확인)
    for i in range(5):
        print(f"\n--- Iteration {i + 1} ---")
        weights = predictor.predict_portfolio()

        print(f"\nPortfolio Weights:")
        for ticker, weight in weights.items():
            print(f"  {ticker}: {weight:.4f} ({weight * 100:.2f}%)")

        total_weight = sum(weights.values())
        print(f"\nTotal Weight: {total_weight:.4f}")
        print("-" * 80)
