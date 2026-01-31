"""
KMeans 모델의 안정성 검증 스크립트
v3.1 Critical Test: 같은 데이터로 여러 번 예측해도 결과가 동일한지 확인
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mama_lite_predictor import MAMAPredictor


def test_kmeans_stability():
    """같은 데이터로 10번 예측하여 레이블 일관성 확인"""
    print("🧪 KMeans Stability Test")
    print("=" * 60)

    predictor = MAMAPredictor()

    # 10번 예측
    results = []
    for i in range(10):
        regime = predictor.get_current_regime()
        results.append(regime)
        print(f"   Iteration {i + 1}: {regime}")

    # 모두 동일해야 함
    unique_results = set(results)

    if len(unique_results) == 1:
        print(f"\n✅ PASS: KMeans Stability Test")
        print(f"   All predictions: {results[0]}")
        return True
    else:
        print(f"\n❌ FAIL: Inconsistent predictions")
        print(f"   Unique results: {unique_results}")
        return False


def test_portfolio_weights_sum():
    """포트폴리오 가중치 합 = 1 테스트"""
    print("\n🧪 Portfolio Weights Sum Test")
    print("=" * 60)

    predictor = MAMAPredictor()
    weights = predictor.predict_portfolio()

    total_weight = sum(weights.values())
    print(f"   Portfolio: {weights}")
    print(f"   Total Weight: {total_weight:.4f}")

    if abs(total_weight - 1.0) < 0.01:
        print(f"✅ PASS: Weights sum to 1.0")
        return True
    else:
        print(f"❌ FAIL: Weights sum to {total_weight:.4f}, expected 1.0")
        return False


if __name__ == "__main__":
    test1 = test_kmeans_stability()
    test2 = test_portfolio_weights_sum()

    print("\n" + "=" * 60)
    if test1 and test2:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
