"""
간소화된 백테스트 실행 스크립트 (인코딩 이슈 해결)
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest_v3_engine import BacktestEngine

if __name__ == "__main__":
    # 5년 백테스트
    engine = BacktestEngine(
        start_date="2019-01-01", end_date="2024-12-31", initial_capital=100000
    )

    metrics, df = engine.run(rebalance_freq="monthly")

    # 간소화된 출력 (이모지 제거)
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (2019-2024)")
    print("=" * 60)
    print(f"Initial Capital: ${engine.initial_capital:,.2f}")
    print(f"Final Value: ${metrics['Final Value']:,.2f}")
    print("-" * 60)
    print(f"CAGR: {metrics['CAGR']:.2%}")
    print(f"Total Return: {metrics['Total Return']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {metrics['Max Drawdown']:.2%}")
    print(f"Calmar Ratio: {metrics['Calmar Ratio']:.2f}")
    print(f"Win Rate: {metrics['Win Rate']:.2%}")
    print(f"Volatility (Ann.): {metrics['Volatility']:.2%}")
    print("-" * 60)
    print(f"Total Trades: {metrics['Total Trades']}")
    print(f"Avg Transaction Cost: ${metrics['Avg Trade Cost']:.2f}")
    print("=" * 60)

    # 목표 대비 평가
    print("\nPerformance vs Targets:")

    cagr_status = (
        "EXCELLENT"
        if metrics["CAGR"] > 0.20
        else "TARGET"
        if metrics["CAGR"] > 0.15
        else "MINIMUM"
        if metrics["CAGR"] > 0.10
        else "BELOW MIN"
    )
    sharpe_status = (
        "EXCELLENT"
        if metrics["Sharpe Ratio"] > 1.3
        else "TARGET"
        if metrics["Sharpe Ratio"] > 1.0
        else "MINIMUM"
        if metrics["Sharpe Ratio"] > 0.7
        else "BELOW MIN"
    )
    mdd_status = (
        "EXCELLENT"
        if abs(metrics["Max Drawdown"]) < 0.15
        else "TARGET"
        if abs(metrics["Max Drawdown"]) < 0.20
        else "MINIMUM"
        if abs(metrics["Max Drawdown"]) < 0.30
        else "BELOW MIN"
    )

    print(f"   CAGR: {cagr_status} (Target: >15%, Excellent: >20%)")
    print(f"   Sharpe: {sharpe_status} (Target: >1.0, Excellent: >1.3)")
    print(f"   MDD: {mdd_status} (Target: <-20%, Excellent: <-15%)")
    print("=" * 60)
