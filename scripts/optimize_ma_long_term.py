"""
Analysis: Long-term Moving Average Optimization (1993-2025)
Author: Antigravity
Date: 2026-01-31

Goal:
1. Test MA Windows with MAX duration (Using SPY as VTI proxy).
2. Report CAGR, MDD, and 'Max Recovery Period' (Underwater Days).
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_long_term_opt():
    print("🚀 Running Long-term MA Optimization (SPY 1993~)...")

    ticker = "SPY"
    start_date = "1993-01-01"
    end_date = "2025-12-31"

    print(f"📥 Downloading data for {ticker}...")
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
    print(f"Data Range: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} days)")

    # Windows
    windows = [150, 185, 200, 250, 300, 365, 400]

    # Pre-calc returns
    df["Rets"] = df["SPY"].pct_change().fillna(0)

    results = []

    for w in windows:
        col = f"MA{w}"
        ma = df["SPY"].rolling(window=w).mean()

        valid_idx = ma.first_valid_index()
        if valid_idx is None:
            continue

        sim_df = df.loc[valid_idx:].copy()
        ma_vals = ma.loc[valid_idx:]

        # Signal: Price > MA
        # Shift 1 day
        signal = (sim_df["SPY"] > ma_vals).astype(int).shift(1).fillna(0)

        # Strategy Ret
        strat_ret = sim_df["Rets"] * signal

        # Cumulative
        cum = (1 + strat_ret).cumprod()
        final_val = cum.iloc[-1]

        # CAGR
        years = len(sim_df) / 252
        cagr = final_val ** (1 / years) - 1

        # MDD
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max
        mdd = drawdown.min()

        # Recovery Period Calculation
        # Identify underwater periods
        is_underwater = drawdown < 0

        max_recovery_days = 0
        current_streak = 0

        # Vectorized streak calc is hard, using loop for safety/clarity on mixed freq
        # Actually, let's group by consecutive True
        # Group ID for underwater
        groups = (is_underwater != is_underwater.shift()).cumsum()
        underwater_groups = groups[is_underwater]

        if not underwater_groups.empty:
            # Count days per group
            # Since index is Datetime, we should take (End - Start)
            durations = []
            for _, g in sim_df.groupby(groups):
                if drawdown.loc[g.index[0]] < 0:  # Check if it's underwater group
                    duration = (g.index[-1] - g.index[0]).days
                    durations.append(duration)
            if durations:
                max_recovery_days = max(durations)

        sharpe = (
            (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
            if strat_ret.std() != 0
            else 0
        )
        trades = signal.diff().abs().sum() / 2

        results.append(
            {
                "Window": w,
                "CAGR": cagr,
                "MDD": mdd,
                "Recovery (Days)": max_recovery_days,
                "Sharpe": sharpe,
                "Trades": int(trades),
            }
        )

    res_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)

    print("\n" + "=" * 80)
    print("⏳ Long-term MA Optimization (1994-2025 SPY)")
    print("=" * 80)
    print(
        res_df.to_string(
            index=False,
            formatters={
                "CAGR": "{:.2%}".format,
                "MDD": "{:.2%}".format,
                "Sharpe": "{:.2f}".format,
                "Recovery (Days)": "{:,} days".format,
            },
        )
    )

    res_df.to_csv("d:/gg/ma_long_term_results.csv", index=False)


if __name__ == "__main__":
    run_long_term_opt()
