"""
Analysis: Long-term MA Optimization WITH 3% Buffer (1993-2025)
Author: Antigravity
Date: 2026-01-31

Goal:
1. Test MA Windows (185, 200, 300) combined with 3% Buffer.
2. Verify if "MA300 + 3% Buffer" is truly the ultimate strategy.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_ma_buffer_long():
    print("🚀 Running Long-term MA + 3% Buffer Optimization (SPY 1993~)...")

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

    # Pre-calc returns
    df["Rets"] = df["SPY"].pct_change().fillna(0)

    # Windows to test
    windows = [185, 200, 300]
    buffer = 0.03  # 3%

    results = []

    for w in windows:
        # Calculate MA
        ma = df["SPY"].rolling(window=w).mean()

        valid_idx = ma.first_valid_index()
        if valid_idx is None:
            continue

        sim_df = df.loc[valid_idx:].copy()
        ma_vals = ma.loc[valid_idx:]
        prices = sim_df["SPY"].values
        m_vals = ma_vals.values

        # Apply Buffer Logic
        # Buy if Price > MA * 1.03
        # Sell if Price < MA * 0.97
        upper = m_vals * (1 + buffer)
        lower = m_vals * (1 - buffer)

        sigs = np.zeros(len(sim_df))
        current = (
            1.0  # Start invested? Or 0? Let's assume invested if Price > MA initially
        )
        if prices[0] > m_vals[0]:
            current = 1.0
        else:
            current = 0.0

        for i in range(len(sim_df)):
            p = prices[i]
            if p > upper[i]:
                current = 1.0
            elif p < lower[i]:
                current = 0.0
            sigs[i] = current

        # Shift signal (Today's signal -> Tomorrow's return)
        pos = pd.Series(sigs, index=sim_df.index).shift(1).fillna(0)

        # Strategy Return
        strat_ret = sim_df["Rets"] * pos

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

        # Recovery
        is_underwater = drawdown < 0
        groups = (is_underwater != is_underwater.shift()).cumsum()
        underwater_groups = groups[is_underwater]

        max_recovery_days = 0
        if not underwater_groups.empty:
            durations = []
            for _, g in sim_df.groupby(groups):
                if drawdown.loc[g.index[0]] < 0:
                    duration = (g.index[-1] - g.index[0]).days
                    durations.append(duration)
            if durations:
                max_recovery_days = max(durations)

        # Trades
        trades = pos.diff().abs().sum() / 2

        sharpe = (
            (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
            if strat_ret.std() != 0
            else 0
        )

        results.append(
            {
                "Window": f"MA{w} + 3% Buffer",
                "CAGR": cagr,
                "MDD": mdd,
                "Recovery (Days)": max_recovery_days,
                "Sharpe": sharpe,
                "Trades": int(trades),
            }
        )

    res_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)

    print("\n" + "=" * 80)
    print("💎 Ultimate Optimization: MA + 3% Buffer (1994-2025 SPY)")
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

    res_df.to_csv("d:/gg/ma_buffer_long_results.csv", index=False)


if __name__ == "__main__":
    run_ma_buffer_long()
