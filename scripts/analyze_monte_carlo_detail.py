"""
Script: Analyze Monte Carlo Raw Data (Detail)
Author: Antigravity
Date: 2026-02-01

Goal: Extract deeper statistical insights from the raw simulation data.
      - Win Rate (vs SPY 10%)
      - Ruin Probability (MDD < -50%)
      - Skewness (Upside Potential vs Downside Risk)
      - Consistency (Std Dev of CAGR)
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


def analyze_raw_data():
    csv_path = "d:/gg/data/backtest_results/monte_carlo_v4_raw.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    cagrs = df["CAGR"]
    mdds = df["MDD"]

    # 1. Benchmark Comparison (Assuming SPY Historic CAGR ~10%)
    benchmark_cagr = 0.10
    win_rate = (cagrs > benchmark_cagr).mean()

    # 2. Safety Metrics
    ruin_prob_50 = (mdds < -0.50).mean()  # Probability of -50% loss
    ruin_prob_40 = (mdds < -0.40).mean()  # Probability of -40% loss

    # 3. Distribution Shape
    # Skewness: > 0 means "Frequent small losses, rare huge gains" (or tail on the right).
    #           For investment, Positive Skew is usually preferred (limited downside, unlimited upside).
    #           However, Trend Following often keeps right tail open.
    dist_skew = skew(cagrs)
    dist_kurt = kurtosis(cagrs)

    # 4. Consistency
    # Coefficient of Variation (CV) = StdDev / Mean
    # Lower is more consistent.
    cagr_mean = cagrs.mean()
    cagr_std = cagrs.std()
    cv = cagr_std / cagr_mean if cagr_mean != 0 else 0

    lines = []
    lines.append("=" * 60)
    lines.append("Deep Dive: Monte Carlo Raw Data Analysis")
    lines.append("=" * 60)

    lines.append(f"1. Alpha Probability (vs SPY 10%):")
    lines.append(f"   - Win Rate: {win_rate:.1%} (Chance to beat market)")

    lines.append(f"\n2. Survival Probability:")
    lines.append(
        f"   - Risk of Halving (-50%): {ruin_prob_50:.2%} ({'Safe' if ruin_prob_50 < 0.01 else 'Risk'})"
    )
    lines.append(f"   - Risk of Deep Pain (-40%): {ruin_prob_40:.2%}")

    lines.append(f"\n3. Distribution Characteristics:")
    lines.append(
        f"   - Skewness: {dist_skew:.2f} ({'Positive (Good Tail)' if dist_skew > 0 else 'Negative (Crash Risk)'})"
    )
    lines.append(f"     > Positive: Gains > Losses probability.")
    lines.append(f"     > Negative: Crash risk exist.")
    lines.append(f"   - Consistency (CV): {cv:.2f} (Low is stable)")

    lines.append(f"\n4. Risk-Adjusted Quality:")
    # Calmar Ratio approximation using means
    calmar = cagr_mean / abs(mdds.mean()) if mdds.mean() != 0 else 0
    lines.append(f"   - Avg Calmar Ratio: {calmar:.2f} (Return per Unit of Drawdown)")

    lines.append("=" * 60)

    output_txt = "\n".join(lines)
    print(output_txt)

    with open("d:/gg/mc_analysis_log.txt", "w", encoding="utf-8") as f:
        f.write(output_txt)


if __name__ == "__main__":
    analyze_raw_data()
