"""
Script: Monte Carlo Simulation (V4 Final Strategy - Hybrid)
Method: Block Bootstrap (to preserve some serial correlation) or Simple Bootstrap.
        Given the trend-following nature, Simple Bootstrap of strategy returns is decent,
        but let's stick to Simple Bootstrap of Daily Returns for simplicity and robustness.

Asset: SCHG
Signal: VTI (MA185 + 3% Buffer)
Crisis: Yield Inversion (Immediate 0% Cash)

Author: Antigravity
Date: 2026-02-01
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os


def run_monte_carlo():
    print("🚀 Starting Monte Carlo Simulation (V4 Final: Hybrid SCHG)...")

    # 1. Hybrid Strategy Backtest Logic (Same as test_hybrid_signal.py)
    tickers = ["VTI", "SCHG", "^TNX", "^IRX"]
    # SCHG start date is around 2009. We simulate from 2010.
    start_date = "2010-01-01"
    end_date = "2025-12-31"

    print(f"Downloading data for {tickers}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"Download error: {e}")
        return

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            df = data["Close"].copy()
        else:
            df = data.copy()
    else:
        df = data.copy()

    df = df.ffill().dropna()

    # Logic: VTI Signal
    vti_prices = df["VTI"]
    ma_vti = vti_prices.rolling(185).mean()
    upper = ma_vti * 1.03
    lower = ma_vti * 0.97

    # Trend State (1=Bull, -1=Bear)
    trend = np.zeros(len(df))
    curr = 1
    p_arr = vti_prices.values
    u_arr = upper.values
    l_arr = lower.values

    for i in range(len(df)):
        if p_arr[i] > u_arr[i]:
            curr = 1
        elif p_arr[i] < l_arr[i]:
            curr = -1
        trend[i] = curr

    # Crisis Logic (Yield Inversion)
    spread = df["^TNX"] - df["^IRX"]
    is_crisis = spread < 0

    # Allocations
    # Bull: 100% SCHG
    # Bear: 30% SCHG
    # Crisis: 0% SCHG (Cash)
    weights = np.zeros(len(df))
    for i in range(len(df)):
        if trend[i] == 1:
            weights[i] = 1.0
        else:
            if is_crisis.iloc[i]:
                weights[i] = 0.0
            else:
                weights[i] = 0.3

    # Strategy Daily Returns
    schg_rets = df["SCHG"].pct_change().fillna(0)
    # Shift weights by 1 day to simulation execution
    strat_rets = schg_rets * pd.Series(weights, index=df.index).shift(1).fillna(0)

    # 2. Monte Carlo (Bootstrapping)
    # Reuse the historical daily returns of the strategy to simulate future paths.

    n_simulations = 10000
    sim_years = 10
    n_days = 252 * sim_years
    initial_capital = 10000.0

    print(f"Simulating {n_simulations} scenarios for {sim_years} years (Bootstrap)...")

    # Convert series to numpy array for speed
    daily_returns_pool = strat_rets.values

    # Result arrays
    final_cagrs = []
    final_mdds = []

    for i in range(n_simulations):
        # Randomly sample n_days from the pool (with replacement)
        random_returns = np.random.choice(daily_returns_pool, size=n_days, replace=True)

        # Calculate Equity Curve
        cum_returns = np.cumprod(1 + random_returns)
        equity_curve = initial_capital * cum_returns

        # Final CAGR
        total_ret = cum_returns[-1]
        cagr = (total_ret) ** (1 / sim_years) - 1
        final_cagrs.append(cagr)

        # MDD
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        mdd = drawdown.min()
        final_mdds.append(mdd)

        if (i + 1) % 2000 == 0:
            print(f"Progress: {i + 1}/{n_simulations}")

    # 3. Analysis
    final_cagrs = np.array(final_cagrs)
    final_mdds = np.array(final_mdds)

    # Save RAW CSV (User Request)
    raw_df = pd.DataFrame(
        {"Run_ID": range(1, n_simulations + 1), "CAGR": final_cagrs, "MDD": final_mdds}
    )
    csv_path = "d:/gg/data/backtest_results/monte_carlo_v4_raw.csv"
    raw_df.to_csv(csv_path, index=False)
    print(f"Saved RAW CSV to {csv_path}")

    results = {
        "runs": n_simulations,
        "years": sim_years,
        "cagr": {
            "mean": float(np.mean(final_cagrs)),
            "median": float(np.median(final_cagrs)),
            "worst_5_percent": float(np.percentile(final_cagrs, 5)),
            "best_5_percent": float(np.percentile(final_cagrs, 95)),
        },
        "mdd": {
            "mean": float(np.mean(final_mdds)),
            "worst_5_percent": float(
                np.percentile(final_mdds, 5)
            ),  # MDD is negative, so 5th percentile is the deep tail (e.g. -50%)
            "safe_95_percent": float(np.percentile(final_mdds, 95)),
        },
    }

    # Print Console Report
    print("\n" + "=" * 50)
    print("🎲 Monte Carlo Simulation Results (V4 Final)")
    print(f"   (Bootstrap 10,000 runs, {sim_years}y Horizon)")
    print("=" * 50)
    print(f"Expectancy (Mean):")
    print(f"  CAGR: {results['cagr']['mean']:.2%}")
    print(f"  MDD:  {results['mdd']['mean']:.2%}")
    print("-" * 50)
    print(f"Risk Scenario (Worst 5%):")
    print(f"  CAGR: {results['cagr']['worst_5_percent']:.2%} (Bad Luck)")
    print(f"  MDD:  {results['mdd']['worst_5_percent']:.2%} (Crash)")
    print("=" * 50)

    # Save JSON
    output_path = "d:/gg/data/backtest_results/monte_carlo_v4_result.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    run_monte_carlo()
