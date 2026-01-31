"""
Strategy: VTI MA185 + Yield Curve + SWAN ETF
Author: Antigravity
Date: 2026-01-31

Logic:
1. VTI > MA185: 100% VTI (Bull Market)
2. VTI < MA185:
   A. Yield Curve Inverted: 100% SWAN (Defensive w/ Upside)
   B. Yield Curve Normal: 50% VTI + 50% SWAN

Note: SWAN Inception is Dec 2018. Backtest will start from 2019.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_strategy():
    print("🚀 Running MA185 + Yield Curve + SWAN Strategy...")

    # 1. Fetch Data
    tickers = ["VTI", "^TNX", "^IRX", "SWAN"]
    start_date = "2018-12-01"  # SWAN Inception
    end_date = "2025-12-31"

    print(f"📥 Downloading data ({start_date} ~ {end_date})...")
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    data = data.ffill()
    data.dropna(inplace=True)

    # 2. Indicators
    data["MA185"] = data["VTI"].rolling(window=185).mean()
    data["Spread"] = data["^TNX"] - data["^IRX"]

    # Returns
    data["VTI_Pct"] = data["VTI"].pct_change()
    data["SWAN_Pct"] = data["SWAN"].pct_change()

    # 3. Logic: Define Weights
    data["W_VTI"] = 0.0
    data["W_SWAN"] = 0.0

    cond_bull = data["VTI"] > data["MA185"]
    cond_inverted = data["Spread"] < 0
    mask_bear = ~cond_bull

    # A. Bull -> 100% VTI
    data.loc[cond_bull, "W_VTI"] = 1.0

    # B. Bear + Inverted -> 100% SWAN (Replace Cash)
    data.loc[mask_bear & cond_inverted, "W_SWAN"] = 1.0

    # C. Bear + Normal -> 50% VTI + 50% SWAN
    data.loc[mask_bear & ~cond_inverted, "W_VTI"] = 0.5
    data.loc[mask_bear & ~cond_inverted, "W_SWAN"] = 0.5

    # Shift weights
    data["W_VTI"] = data["W_VTI"].shift(1)
    data["W_SWAN"] = data["W_SWAN"].shift(1)

    data.dropna(inplace=True)

    # 4. Returns
    data["Strategy_Ret"] = (data["VTI_Pct"] * data["W_VTI"]) + (
        data["SWAN_Pct"] * data["W_SWAN"]
    )

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

    # Benchmark
    vti_val = data["VTI_Cum"].iloc[-1]
    vti_cagr = (vti_val / initial_capital) ** (1 / years) - 1
    vti_mdd = (
        (data["VTI_Cum"] - data["VTI_Cum"].cummax()) / data["VTI_Cum"].cummax()
    ).min()

    print("-" * 50)
    print(f"Results with SWAN ({data.index[0].date()} ~ {data.index[-1].date()})")
    print("-" * 50)
    print(f"Final Value: ${final_val:,.2f}")
    print(f"CAGR: {cagr * 100:.2f}%")
    print(f"MDD: {mdd * 100:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")

    print("-" * 50)
    print(f"Benchmark (VTI Only)")
    print(f"Final Value: ${vti_val:,.2f}")
    print(f"CAGR: {vti_cagr * 100:.2f}%")
    print(f"MDD: {vti_mdd * 100:.2f}%")

    # Save
    data.to_csv("d:/gg/strategy_swan_results.csv")
    print("Saved to d:/gg/strategy_swan_results.csv")


if __name__ == "__main__":
    run_strategy()
