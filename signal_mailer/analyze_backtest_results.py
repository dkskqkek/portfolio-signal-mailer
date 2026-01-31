"""
백테스트 결과 상세 분석 리포트 생성
"""

import json
import pandas as pd

# 결과 로드
with open("d:/gg/backtest_results.json", "r") as f:
    results = json.load(f)

print("\n" + "=" * 80)
print("ANTIGRAVITY v3.1 백테스트 실제 성과 분석")
print("=" * 80)

print(f"\n기간: {results['period']}")
print(f"초기 자본: ${results['initial_capital']:,.0f}")
print(f"최종 가치: ${results['final_value']:,.2f}")

print("\n" + "-" * 80)
print("핵심 성과 지표")
print("-" * 80)

print(f"\n📈 수익률")
print(f"  • CAGR (연평균):        {results['cagr'] * 100:7.2f}%")
print(f"  • Total Return (누적):  {results['total_return'] * 100:7.2f}%")
print(
    f"  • 투자 기간 수익:       ${results['final_value'] - results['initial_capital']:,.2f}"
)

print(f"\n⚖️  위험 조정 수익")
print(f"  • Sharpe Ratio:         {results['sharpe_ratio']:7.2f}")
print(f"  • Calmar Ratio:         {results['calmar_ratio']:7.2f}")
print(f"  • Volatility (연):      {results['volatility'] * 100:7.2f}%")

print(f"\n📉 리스크 지표")
print(f"  • Max Drawdown:         {results['max_drawdown'] * 100:7.2f}%")
print(f"  • Win Rate:             {results['win_rate'] * 100:7.2f}%")

print(f"\n💰 거래 비용")
print(f"  • 총 거래 횟수:         {results['total_trades']:7.0f}회")
print(f"  • 평균 거래 비용:       ${results['avg_trade_cost']:7.2f}")
print(
    f"  • 총 거래 비용 (추정):  ${results['avg_trade_cost'] * results['total_trades']:,.2f}"
)

print("\n" + "-" * 80)
print("목표 대비 평가")
print("-" * 80)

# 목표 기준
targets = {
    "minimum": {"cagr": 0.10, "sharpe": 0.7, "mdd": -0.30},
    "target": {"cagr": 0.15, "sharpe": 1.0, "mdd": -0.20},
    "excellent": {"cagr": 0.20, "sharpe": 1.3, "mdd": -0.15},
}


def evaluate_metric(value, target_low, excellent_low, ascending=True):
    if ascending:
        if value >= excellent_low:
            return "✅ EXCELLENT"
        elif value >= target_low:
            return "✅ TARGET"
        else:
            return "⚠️  MINIMUM" if value >= 0 else "❌ BELOW MIN"
    else:  # For MDD (lower is better)
        if abs(value) <= abs(excellent_low):
            return "✅ EXCELLENT"
        elif abs(value) <= abs(target_low):
            return "✅ TARGET"
        else:
            return "⚠️  MINIMUM" if abs(value) <= 0.30 else "❌ BELOW MIN"


cagr_eval = evaluate_metric(results["cagr"], 0.15, 0.20)
sharpe_eval = evaluate_metric(results["sharpe_ratio"], 1.0, 1.3)
mdd_eval = evaluate_metric(results["max_drawdown"], -0.20, -0.15, ascending=False)

print(f"\n  CAGR:        {cagr_eval:20s}  (목표: >15%, Excellent: >20%)")
print(f"  Sharpe:      {sharpe_eval:20s}  (목표: >1.0, Excellent: >1.3)")
print(f"  MDD:         {mdd_eval:20s}  (목표: <-20%, Excellent: <-15%)")

# 종합 평가
if "✅ EXCELLENT" in cagr_eval and "✅" in sharpe_eval and "✅" in mdd_eval:
    overall = "🌟 OUTSTANDING - 모든 지표 우수"
elif "❌" in [cagr_eval, sharpe_eval, mdd_eval]:
    overall = "⚠️  NEEDS IMPROVEMENT - 일부 지표 미달"
else:
    overall = "✅ GOOD - 목표 달성"

print(f"\n종합 평가: {overall}")

print("\n" + "=" * 80)

# SPY 비교 (참고)
print("\n※ 참고: SPY (S&P 500) 동일 기간 CAGR 약 15% (레버리지 없음)")
print(
    f"   Antigravity는 SPY 대비 약 {results['cagr'] / 0.15:.1f}배의 연평균 수익률 달성"
)
print("=" * 80)
