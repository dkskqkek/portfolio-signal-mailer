"""
Analysis: Yield Curve Impact Verification
Author: Antigravity
Date: 2026-01-31

Goal: Compare two Bear Market strategies.
1. With Yield Logic (Antigravity V4):
   - Bear + Inverted Yield -> 0% Stock (Crisis Mode)
   - Bear + Normal Yield -> 30% Stock (Correction Mode)
2. Without Yield Logic (Simple):
   - Bear -> Always 30% Stock (Regardless of Yield Curve)

Hypothesis: Yield Logic should save us from 'Deep' crashes (2000, 2008) but might add noise (false alarms).
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_yield_verification():
    print("🚀 Verifying Yield Curve Impact (1993-2025)...")

    tickers = ["SPY", "^TNX", "^IRX"]
    start_date = "1993-01-01"
    end_date = "2025-12-31"

    print(f"📥 Downloading data...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Flatten
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            df = data["Close"].copy()
        else:
            df = data.copy()
    else:
        df = data.copy()

    df = df.ffill().dropna()

    # Indicators
    df["MA185"] = df["SPY"].rolling(window=185).mean()
    df["Spread"] = df["^TNX"] - df["^IRX"]

    # Buffer Logic
    buffer = 0.03
    upper = df["MA185"] * (1 + buffer)
    lower = df["MA185"] * (1 - buffer)

    prices = df["SPY"].values
    u_vals = upper.values
    l_vals = lower.values

    # Trend Signal
    trend = np.zeros(len(df))
    current_trend = 1 if prices[0] > u_vals[0] else 0
    for i in range(len(df)):
        p = prices[i]
        if p > u_vals[i]:
            current_trend = 1
        elif p < l_vals[i]:
            current_trend = 0
        trend[i] = current_trend
    df["Trend"] = trend

    spreads = df["Spread"].values
    trends = df["Trend"].values
    rets = df["SPY"].pct_change().fillna(0)

    # Simulation
    def run_sim(use_yield_logic):
        weights = np.zeros(len(df))

        for i in range(len(df)):
            if trends[i] == 1:  # Bull
                weights[i] = 1.0
            else:  # Bear
                if use_yield_logic:
                    if spreads[i] < 0:  # Inverted
                        weights[i] = 0.0  # Crisis Mode
                    else:  # Normal
                        weights[i] = 0.3  # Correction Mode (as optimized)
                else:
                    # Simple Mode: Always 30% in Bear
                    weights[i] = 0.3

        # Shift
        pos = pd.Series(weights, index=df.index).shift(1).fillna(0)
        strat_ret = rets * pos

        cum = (1 + strat_ret).cumprod()
        final_val = cum.iloc[-1]
        years = len(df) / 252
        cagr = final_val ** (1 / years) - 1

        roll_max = cum.cummax()
        mdd = ((cum - roll_max) / roll_max).min()
        sharpe = (
            (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
            if strat_ret.std() != 0
            else 0
        )

        # Count Trades (Weight change > 0.1)
        trades = pd.Series(weights).diff().abs().gt(0.1).sum()

        return cagr, mdd, sharpe, final_val

    cagr_y, mdd_y, sharpe_y, final_y = run_sim(use_yield_logic=True)
    cagr_n, mdd_n, sharpe_n, final_n = run_sim(use_yield_logic=False)

    print("\n" + "=" * 60)
    print("⚖️ Yield Curve Logic Impact Analysis")
    print("=" * 60)

    print(f"1. WITH Yield Logic (V4 Standard)")
    print(
        f"   CAGR: {cagr_y * 100:.2f}% | MDD: {mdd_y * 100:.2f}% | Sharpe: {sharpe_y:.2f}"
    )
    print(f"   Final Value: ${final_y:,.2f}")

    print("-" * 60)

    print(f"2. WITHOUT Yield Logic (Always 30% in Bear)")
    print(
        f"   CAGR: {cagr_n * 100:.2f}% | MDD: {mdd_n * 100:.2f}% | Sharpe: {sharpe_n:.2f}"
    )
    print(f"   Final Value: ${final_n:,.2f}")

    print("-" * 60)

    # Conclusion
    diff_cagr = cagr_y - cagr_n
    diff_mdd = (
        mdd_y - mdd_n
    )  # negative is better? No, mdd is negative. higher is better (closer to 0)

    # Save Results
    results = [
        {
            "Scenario": "With Yield Logic",
            "CAGR": cagr_y,
            "MDD": mdd_y,
            "Sharpe": sharpe_y,
            "Final": final_y,
        },
        {
            "Scenario": "Without Yield Logic",
            "CAGR": cagr_n,
            "MDD": mdd_n,
            "Sharpe": sharpe_n,
            "Final": final_n,
        },
    ]
    pd.DataFrame(results).to_csv("d:/gg/yield_impact_results.csv", index=False)
    print("Saved to d:/gg/yield_impact_results.csv")

    print("🔍 Conclusion:")
    if diff_cagr > 0 and mdd_y > mdd_n:
        print("✅ Yield Logic IMPROVES both Return and Safety! (Must Have)")
    elif diff_cagr < 0 and mdd_y > mdd_n:
        print("🛡️ Yield Logic sacrifices Return for Safety.")
    elif diff_cagr > 0 and mdd_y < mdd_n:
        print("⚠️ Yield Logic improves Return but increases Risk.")
    else:
        print("❌ Yield Logic makes it WORSE. (Remove it)")


if __name__ == "__main__":
    run_yield_verification()
