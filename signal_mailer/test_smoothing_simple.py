"""
단순 Regime Smoothing 테스트 - 한 번만 예측
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mama_lite_predictor import MAMAPredictor
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("d:/gg/smoothing_test.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REGIME SMOOTHING - SINGLE PREDICTION TEST")
    print("=" * 60 + "\n")

    predictor = MAMAPredictor()

    print("\n>>> Calling predict_portfolio()...\n")
    weights = predictor.predict_portfolio()

    print("\n" + "-" * 60)
    print("PORTFOLIO ALLOCATION:")
    print("-" * 60)

    stock_total = 0
    bond_total = 0

    for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {ticker:6s}: {weight * 100:5.2f}%")
        if ticker in ["BIL", "TLT"]:
            bond_total += weight
        else:
            stock_total += weight

    print("-" * 60)
    print(f"  TOTAL STOCK: {stock_total * 100:.2f}%")
    print(f"  TOTAL BOND:  {bond_total * 100:.2f}%")
    print(f"  GRAND TOTAL: {sum(weights.values()) * 100:.2f}%")
    print("=" * 60)

    print(f"\nLog saved to: d:/gg/smoothing_test.log")
