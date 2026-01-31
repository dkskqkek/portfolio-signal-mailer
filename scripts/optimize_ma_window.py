"""
Analysis: Optimize Moving Average Window
Author: Antigravity
Date: 2026-01-31

Goal: Determine if MA185 is truly the optimal window or if another length performs better.

Range: 50 to 300 days (Step 10 or selected key MAs)
Metric: Sharpe Ratio, CAGR, MDD
Strategy: VTI > MA -> Invest, VTI < MA -> Cash (0% return assumption for clean comparison)
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_ma_optimization():
    print("🚀 Optimizing Moving Average Window...")

    # Data
    ticker = "VTI"
    start_date = "2000-01-01"
    end_date = "2025-12-31"

    print(f"📥 Downloading data for {ticker} ({start_date} ~ {end_date})...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Flatten
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            df = data["Close"].copy()
        else:
            df = data.copy()
    else:
        df = data.copy()

    df = df.ffill().dropna()
    print(f"Data Shape: {df.shape}")

    # Windows to test
    # Common MAs + 185
    windows = [50, 60, 90, 100, 120, 150, 180, 185, 200, 220, 250, 300]

    results = []

    # Pre-calculate returns
    df["Rets"] = df["VTI"].pct_change().fillna(0)

    for w in windows:
        col_name = f"MA{w}"
        ma_series = df["VTI"].rolling(window=w).mean()

        # Valid data start
        valid_idx = ma_series.first_valid_index()
        if valid_idx is None:
            continue

        sim_df = df.loc[valid_idx:].copy()
        ma_vals = ma_series.loc[valid_idx:]

        # Signal: Price > MA = 1, else 0
        # Shift 1 day
        signal = (sim_df["VTI"] > ma_vals).astype(int).shift(1).fillna(0)

        # Strategy Return
        # (Assuming Cash return = 0 for pure curve comparison)
        strat_ret = sim_df["Rets"] * signal

        # Metrics
        cum = (1 + strat_ret).cumprod()
        final_val = cum.iloc[-1]

        years = len(sim_df) / 252
        cagr = final_val ** (1 / years) - 1

        mdd = ((cum - cum.cummax()) / cum.cummax()).min()

        vol = strat_ret.std() * np.sqrt(252)
        sharpe = (cagr / vol) if vol != 0 else 0

        # Trades
        trades = signal.diff().abs().sum() / 2

        results.append(
            {
                "Window": w,
                "CAGR": cagr,
                "MDD": mdd,
                "Sharpe": sharpe,
                "Trades": int(trades),
            }
        )

    res_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)

    print("\n" + "=" * 60)
    print("📏 Moving Average Optimization Results (Sorted by Sharpe)")
    print("=" * 60)
    print(
        res_df.to_string(
            index=False,
            formatters={
                "CAGR": "{:.2%}".format,
                "MDD": "{:.2%}".format,
                "Sharpe": "{:.2f}".format,
            },
        )
    )

    # Save
    res_df.to_csv("d:/gg/ma_optimization_results.csv", index=False)


if __name__ == "__main__":
    run_ma_optimization()
