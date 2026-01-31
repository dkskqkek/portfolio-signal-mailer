"""
백테스트 실행 및 결과를 JSON 파일로 저장
"""

import sys
import os
import json
import logging
import numpy as np

# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from backtest_v3_engine import BacktestEngine


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    print("Running backtest...")

    # 5년 백테스트 (2021-2025)
    engine = BacktestEngine(
        start_date="2021-01-01", end_date="2025-12-31", initial_capital=100000
    )

    metrics, df = engine.run(rebalance_freq="monthly")

    # 결과를 dict로 변환 (JSON serializable)
    results = {
        "period": "2019-2024",
        "initial_capital": engine.initial_capital,
        "final_value": metrics["Final Value"],
        "cagr": metrics["CAGR"],
        "total_return": metrics["Total Return"],
        "sharpe_ratio": metrics["Sharpe Ratio"],
        "max_drawdown": metrics["Max Drawdown"],
        "calmar_ratio": metrics["Calmar Ratio"],
        "win_rate": metrics["Win Rate"],
        "volatility": metrics["Volatility"],
        "total_trades": metrics["Total Trades"],
        "avg_trade_cost": metrics["Avg Trade Cost"],
    }

    # JSON 파일로 저장
    output_path = r"d:\gg\backtest_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to: {output_path}")
    print("\nQuick Summary:")
    print(f"  CAGR: {metrics['CAGR']:.2%}")
    print(f"  Sharpe: {metrics['Sharpe Ratio']:.2f}")
    print(f"  MDD: {metrics['Max Drawdown']:.2%}")
