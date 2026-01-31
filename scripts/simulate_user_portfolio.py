"""
User Portfolio Monte Carlo Simulation Script
- Loads user's US portfolio tickers (Toss Account)
- Runs Backtest (2021-2025)
- Runs Monte Carlo Simulation
- Saves results to JSON
"""

import sys
import os
import json
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_mailer.backtest_v3_engine import BacktestEngine
from signal_mailer.monte_carlo_simulation import MonteCarloSimulator


def run_simulation():
    # 1. Define User Portfolio (Toss Account)
    user_tickers = ["GOOGL", "VGT", "COWZ", "XLV", "CVX", "VXUS", "GLDM"]

    print("=" * 60)
    print(f"👤 User Portfolio Simulation: {len(user_tickers)} Assets")
    print(f"   Tickers: {user_tickers}")
    print("=" * 60)

    # 2. Run Backtest
    engine = BacktestEngine(
        start_date="2021-01-01", end_date="2025-12-31", initial_capital=150000
    )

    metrics, df = engine.run(rebalance_freq="monthly", custom_tickers=user_tickers)

    # Save Backtest Results for Monte Carlo
    results = {
        "cagr": metrics["CAGR"],
        "max_drawdown": metrics["Max Drawdown"],
        "sharpe_ratio": metrics["Sharpe Ratio"],
        "volatility": metrics["Volatility"],
        "initial_capital": engine.initial_capital,
    }

    temp_json_path = r"d:\gg\data\user_portfolio_backtest.json"
    os.makedirs(os.path.dirname(temp_json_path), exist_ok=True)
    with open(temp_json_path, "w") as f:
        json.dump(results, f)

    print(f"\n✅ Backtest Metrics for User Portfolio:")
    print(f"   CAGR: {metrics['CAGR']:.2%}")
    print(f"   MDD: {metrics['Max Drawdown']:.2%}")
    print(f"   Sharpe: {metrics['Sharpe Ratio']:.2f}")

    # 3. Run Monte Carlo
    print("\n🎲 Running Monte Carlo Simulation (Next 5 Years)...")
    mc = MonteCarloSimulator(backtest_results_path=temp_json_path)
    sim_df = mc.simulate(n_simulations=1000)

    # Analyze
    analysis = mc.analyze_results()

    # Save MC Results
    mc_json_path = r"d:\gg\data\user_portfolio_mc_results.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(mc_json_path, "w") as f:
        json.dump(analysis, f, cls=NumpyEncoder)

    return analysis


if __name__ == "__main__":
    run_simulation()
