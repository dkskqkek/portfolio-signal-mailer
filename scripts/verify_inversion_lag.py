"""
Analysis: Yield Inversion 3-Month Confirmation Test
Author: Antigravity
Date: 2026-02-01

Goal: Verify if waiting for 3 months of continuous inversion improves performance.
Hypothesis:
- Stage 2 (Inversion < 3 months): Ignore (Normal Bear, 30% Stock)
- Stage 3 (Inversion >= 3 months): Crisis (0% Stock)

Universe: SCHG (since it was the winner), fallbacks to VTI if SCHG not avail.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_lag_test():
    print("Starting Yield Inversion Lag Test (Immediate vs 3-Month)...")

    # We use SCHG data if possible, else VTI. Ideally VTI for longer history (1993~).
    # But user wants to trade SCHG.
    # Let's verify Logic on VTI (Long History) to catch 2000/2008 properly,
    # but verify Returns on SCHG (if data avail) or mapped VTI->SCHG proxy.
    # To be safe and utilize long history for "Yield Logic", we must use VTI (or SPY data proxy).

    tickers = ["SPY", "^TNX", "^IRX"]  # SPY covers 2000 dotcom bubble
    start_date = "1993-01-01"
    end_date = "2025-12-31"

    print(f"Downloading data for {tickers}...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    except Exception as e:
        print(f"Download error: {e}")
        return

    # Flatten
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            df = data["Close"].copy()
        else:
            df = data.copy()
    else:
        df = data.copy()

    df = df.ffill().dropna()

    # Spread
    spread = df["^TNX"] - df["^IRX"]

    # Logic: MA185 + Buffer
    ma = df["SPY"].rolling(185).mean()
    upper = ma * 1.03
    lower = ma * 0.97

    # Trend State
    trend_states = np.zeros(len(df))
    prices = df["SPY"].values
    uppers = upper.values
    lowers = lower.values

    curr = 1
    for i in range(len(df)):
        if prices[i] > uppers[i]:
            curr = 1
        elif prices[i] < lowers[i]:
            curr = -1
        trend_states[i] = curr

    # --- Inversion Logic ---

    # 1. Immediate Inversion
    # If Spread < 0 -> Crisis
    is_inverted_immediate = spread < 0

    # 2. 3-Month Lag Inversion
    # Condition: Spread MUST be negative for the last 63 days continuously.
    # rolling(63).max() < 0 means all elements were negative.
    is_inverted_3m = spread.rolling(window=63).max() < 0

    # Weights Calc
    # Bear = 0.3, Crisis = 0.0

    def calc_weights(trend, inversion_mask):
        w = np.zeros(len(trend))
        for i in range(len(trend)):
            if trend[i] == 1:  # Bull
                w[i] = 1.0
            else:  # Bear
                if inversion_mask[i]:  # Crisis
                    w[i] = 0.0
                else:  # Normal Bear
                    w[i] = 0.3
        return w

    w_immediate = calc_weights(trend_states, is_inverted_immediate)
    w_lagged = calc_weights(trend_states, is_inverted_3m)

    # Returns
    rets = df["SPY"].pct_change().fillna(0)

    # Shift weights
    strat_immediate = rets * pd.Series(w_immediate, index=df.index).shift(1).fillna(0)
    strat_lagged = rets * pd.Series(w_lagged, index=df.index).shift(1).fillna(0)

    # Metrics
    def get_metrics(r):
        cum = (1 + r).cumprod()
        final = cum.iloc[-1]
        years = len(r) / 252
        cagr = (final) ** (1 / years) - 1
        dd = (cum - cum.cummax()) / cum.cummax()
        mdd = dd.min()
        sharpe = (r.mean() / r.std()) * np.sqrt(252)
        return cagr, mdd, sharpe

    m_imm = get_metrics(strat_immediate)
    m_lag = get_metrics(strat_lagged)

    print("\n" + "=" * 60)
    print(f"Yield Inversion Lag Test (SPY 1993-2025)")
    print("=" * 60)
    print(f"{'Strategy':<20} | {'CAGR':<10} | {'MDD':<10} | {'Sharpe':<10}")
    print("-" * 60)
    print(
        f"{'1. Immediate':<20} | {m_imm[0]:.2%}   | {m_imm[1]:.2%}   | {m_imm[2]:.2f}"
    )
    print(
        f"{'2. 3-Month Lag':<20} | {m_lag[0]:.2%}   | {m_lag[1]:.2%}   | {m_lag[2]:.2f}"
    )
    print("=" * 60)

    # Analysis
    if (
        m_lag[1] > m_imm[1] * 1.1
    ):  # If Lag makes MDD significantly worse (e.g. -25% -> -30%)
        print("\nCaution: Waiting 3 months exposed us to more downside.")
    elif m_lag[0] > m_imm[0]:
        print("\nSuccess: Waiting improved returns without ruining safety.")
    else:
        print("\nNeutral/Mixed Result.")

    # Save to CSV
    results = [
        {
            "Strategy": "1. Immediate",
            "CAGR": m_imm[0],
            "MDD": m_imm[1],
            "Sharpe": m_imm[2],
        },
        {
            "Strategy": "2. 3-Month Lag",
            "CAGR": m_lag[0],
            "MDD": m_lag[1],
            "Sharpe": m_lag[2],
        },
    ]
    pd.DataFrame(results).to_csv(
        "d:/gg/data/backtest_results/inversion_lag_test.csv", index=False
    )
    print(f"\nSaved results to d:/gg/data/backtest_results/inversion_lag_test.csv")


if __name__ == "__main__":
    run_lag_test()
