"""
Strategy: VTI MA185 + Yield Curve Inversion + QQQ Inverse (PSQ)
Author: Antigravity
Date: 2026-01-31

Logic:
1. VTI > MA185: 100% VTI (Bull Market)
2. VTI < MA185:
   A. Yield Curve Inverted (Recession Warning): 100% PSQ (Aggressive Short)
   B. Yield Curve Normal: 50% VTI + 50% PSQ (Hedge / Market Neutral)

Data:
- VTI: Market
- PSQ: ProShares Short QQQ (1x Inverse)
- ^TNX, ^IRX: Yields
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_strategy():
    print("🚀 Running MA185 + Yield Curve + Inverse (PSQ) Strategy...")

    # 1. Fetch Data
    tickers = ["VTI", "^TNX", "^IRX", "PSQ"]
    start_date = "2006-06-20"  # PSQ Inception around June 2006
    end_date = "2025-12-31"

    print(f"📥 Downloading data ({start_date} ~ {end_date})...")
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    data = data.ffill()

    # 2. Indicators
    data["MA185"] = data["VTI"].rolling(window=185).mean()
    data["Spread"] = data["^TNX"] - data["^IRX"]

    # Returns
    data["VTI_Pct"] = data["VTI"].pct_change()
    data["PSQ_Pct"] = data["PSQ"].pct_change()

    # 3. Logic Signal
    # Signal: 1.0 = 100% VTI, 0.5 = 50% VTI + 50% PSQ, 0.0 = 100% PSQ
    data["Signal"] = 0.5  # Default Bear/Normal

    cond_bull = data["VTI"] > data["MA185"]
    cond_inverted = data["Spread"] < 0

    # Bear & Inverted -> 100% PSQ (Signal 0.0 captures this logic below)
    # Let's redefine weights explicitly to avoid confusion

    # Weights columns
    data["W_VTI"] = 0.0
    data["W_PSQ"] = 0.0

    # Vectorized conditions
    # 1. Bull: VTI > MA185
    data.loc[cond_bull, "W_VTI"] = 1.0
    data.loc[cond_bull, "W_PSQ"] = 0.0

    # 2. Bear (Implicit)
    mask_bear = ~cond_bull

    # 2-A. Bear + Inverted (Crash Imminent) -> 100% PSQ
    data.loc[mask_bear & cond_inverted, "W_VTI"] = 0.0
    data.loc[mask_bear & cond_inverted, "W_PSQ"] = 1.0

    # 2-B. Bear + Normal (Correction) -> 50% VTI + 50% PSQ
    data.loc[mask_bear & ~cond_inverted, "W_VTI"] = 0.5
    data.loc[mask_bear & ~cond_inverted, "W_PSQ"] = 0.5

    # Shift weights (Today's signal acts on Tomorrow's return)
    data["W_VTI"] = data["W_VTI"].shift(1)
    data["W_PSQ"] = data["W_PSQ"].shift(1)

    data.dropna(inplace=True)

    # 4. Calculate Strategy Return
    data["Strategy_Ret"] = (data["VTI_Pct"] * data["W_VTI"]) + (
        data["PSQ_Pct"] * data["W_PSQ"]
    )

    # Cumulative
    initial_capital = 100000
    data["Strategy_Cum"] = (1 + data["Strategy_Ret"]).cumprod() * initial_capital
    data["VTI_Cum"] = (1 + data["VTI_Pct"]).cumprod() * initial_capital

    # Metrics
    final_val = data["Strategy_Cum"].iloc[-1]
    years = len(data) / 252
    cagr = (final_val / initial_capital) ** (1 / years) - 1

    cummax = data["Strategy_Cum"].cummax()
    mdd = ((data["Strategy_Cum"] - cummax) / cummax).min()

    sharpe = (data["Strategy_Ret"].mean() / data["Strategy_Ret"].std()) * np.sqrt(252)

    print("-" * 50)
    print(
        f"Inverse Strategy Results ({data.index[0].date()} ~ {data.index[-1].date()})"
    )
    print("-" * 50)
    print(f"Final Value: ${final_val:,.2f}")
    print(f"CAGR: {cagr * 100:.2f}%")
    print(f"MDD: {mdd * 100:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")

    # VTI Benchmark
    vti_val = data["VTI_Cum"].iloc[-1]
    vti_cagr = (vti_val / initial_capital) ** (1 / years) - 1
    vti_mdd = (
        (data["VTI_Cum"] - data["VTI_Cum"].cummax()) / data["VTI_Cum"].cummax()
    ).min()

    print("-" * 50)
    print(f"Benchmark (VTI Only)")
    print(f"Final Value: ${vti_val:,.2f}")
    print(f"CAGR: {vti_cagr * 100:.2f}%")
    print(f"MDD: {vti_mdd * 100:.2f}%")

    # Save
    data.to_csv("d:/gg/strategy_inverse_results.csv")
    print("Saved to d:/gg/strategy_inverse_results.csv")


if __name__ == "__main__":
    run_strategy()
