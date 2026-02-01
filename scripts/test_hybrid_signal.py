"""
Analysis: Hybrid Signal Strategy Test
Author: Antigravity
Date: 2026-02-01

Goal: Compare "Self-Driving" vs "Market-Signal" strategies.
Hypothesis: VTI (Broad Market) signals might filter out noise better than SCHG's own volatile price action.

Strategies:
1. VTI Self: VTI Signal -> VTI Trade
2. SCHG Self: SCHG Signal -> SCHG Trade
3. Hybrid: VTI Signal -> SCHG Trade
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_hybrid_test():
    print("🚀 Starting Hybrid Signal Test (VTI vs SCHG)...")

    tickers = ["VTI", "SCHG", "^TNX", "^IRX"]
    start_date = "2010-01-01"  # SCHG inception ~2009. Safe start 2010.
    end_date = "2025-12-31"

    print(f"📥 Downloading data for {tickers}...")
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

    # Macro Spread
    spread = df["^TNX"] - df["^IRX"]

    # --- Indicator Calc Function ---
    def calc_signal(price_series):
        ma = price_series.rolling(window=185).mean()
        buffer = 0.03
        upper = ma * (1 + buffer)
        lower = ma * (1 - buffer)

        states = np.zeros(len(price_series))
        vals = price_series.values
        ups = upper.values
        lows = lower.values

        curr = 1  # Init Bull
        for i in range(len(vals)):
            if vals[i] > ups[i]:
                curr = 1
            elif vals[i] < lows[i]:
                curr = -1
            states[i] = curr

        return states

    # 1. Calc Signals
    vti_signal = calc_signal(df["VTI"])
    schg_signal = calc_signal(df["SCHG"])

    # 2. Logic to Weights (Bear=30%, Crisis=0%)
    def signal_to_weight(signal_arr, spread_arr):
        w = np.zeros(len(signal_arr))
        for i in range(len(signal_arr)):
            if signal_arr[i] == 1:  # Bull
                w[i] = 1.0
            else:  # Bear
                if spread_arr[i] < 0:  # Crisis
                    w[i] = 0.0
                else:  # Normal Bear
                    w[i] = 0.3  # 30% Stock
        return w

    w_vti_self = signal_to_weight(vti_signal, spread.values)
    w_schg_self = signal_to_weight(schg_signal, spread.values)

    # Hybrid: VTI Signal -> SCHG Trade
    w_hybrid = w_vti_self  # Use VTI's allocation decision

    # 3. Calculate Returns
    # Need to match lengths due to rolling window drop in calc_signal (handled by same Index)
    # Actually calc_signal returns array same length as input (with NaNs at start implicitly if rolling used but here we handled loop manually).
    # Wait, rolling mean will have NaNs for first 184 days.
    # We should trim everything to valid range.

    valid_mask = ~np.isnan(df["VTI"].rolling(window=185).mean())
    df_valid = df[valid_mask].copy()

    # Re-slice signals and weights to match valid df
    # Indices must match. Let's do it safer.

    # Re-run calc properly on aligned DF? No, rolling needs history.
    # Simple way: Calculate on full DF, then slice results.

    # Slice weights
    start_idx = 185
    weights_vti_self = w_vti_self[start_idx:]
    weights_schg_self = w_schg_self[start_idx:]
    weights_hybrid = w_hybrid[start_idx:]

    # Slice Prices for Returns
    clean_df = df.iloc[start_idx:].copy()

    schg_rets = clean_df["SCHG"].pct_change().fillna(0)
    vti_rets = clean_df["VTI"].pct_change().fillna(0)

    # Strategy Returns
    # Shift weights by 1 day to simulate "Trade at Close using Yesterday's Signal" or "Trade Today using Today's Open"?
    # Standard Backtest: Signal calculated at Close(t), Trade executed at Close(t) or Open(t+1).
    # Here we multiply ret(t) * weight(t-1).

    strat_vti_self = vti_rets * pd.Series(weights_vti_self, index=clean_df.index).shift(
        1
    ).fillna(0)
    strat_schg_self = schg_rets * pd.Series(
        weights_schg_self, index=clean_df.index
    ).shift(1).fillna(0)
    strat_hybrid = schg_rets * pd.Series(weights_hybrid, index=clean_df.index).shift(
        1
    ).fillna(0)  # Logic VTI, Asset SCHG

    # Metrics
    def get_metrics(r):
        cum = (1 + r).cumprod()
        final = cum.iloc[-1]
        years = len(r) / 252
        cagr = (final) ** (1 / years) - 1

        dd = (cum - cum.cummax()) / cum.cummax()
        mdd = dd.min()

        sharpe = (r.mean() / r.std()) * np.sqrt(252)
        sortino = (r.mean() / r[r < 0].std()) * np.sqrt(252)

        return cagr, mdd, sharpe, sortino

    m_vti = get_metrics(strat_vti_self)
    m_schg = get_metrics(strat_schg_self)
    m_hybrid = get_metrics(strat_hybrid)

    print("\n" + "=" * 80)
    print(
        f"📊 Hybrid Strategy Test Results ({clean_df.index[0].date()} ~ {clean_df.index[-1].date()})"
    )
    print("=" * 80)
    print(
        f"{'Strategy':<25} | {'CAGR':<10} | {'MDD':<10} | {'Sortino':<10} | {'Sharpe':<10}"
    )
    print("-" * 80)
    print(
        f"{'1. VTI Self (Base)':<25} | {m_vti[0]:.2%}   | {m_vti[1]:.2%}   | {m_vti[3]:.2f}       | {m_vti[2]:.2f}"
    )
    print(
        f"{'2. SCHG Self (Winner)':<25} | {m_schg[0]:.2%}   | {m_schg[1]:.2%}   | {m_schg[3]:.2f}       | {m_schg[2]:.2f}"
    )
    print(
        f"{'3. Hybrid (VTI->SCHG)':<25} | {m_hybrid[0]:.2%}   | {m_hybrid[1]:.2%}   | {m_hybrid[3]:.2f}       | {m_hybrid[2]:.2f}"
    )
    print("=" * 80)

    # Interpretation
    if m_schg[3] > m_hybrid[3]:
        print("\n✅ Conclusion: SCHG Self-Driving is BETTER.")
        print(
            "   Reason: SCHG often moves differently from VTI. Waiting for VTI signals causes lag."
        )
    else:
        print("\n✅ Conclusion: Hybrid (VTI Signal) is BETTER.")
        print("   Reason: VTI filters out SCHG's noise effectively.")
    # Save to CSV
    results = [
        {
            "Strategy": "1. VTI Self (Base)",
            "CAGR": m_vti[0],
            "MDD": m_vti[1],
            "Sortino": m_vti[3],
            "Sharpe": m_vti[2],
        },
        {
            "Strategy": "2. SCHG Self (Winner)",
            "CAGR": m_schg[0],
            "MDD": m_schg[1],
            "Sortino": m_schg[3],
            "Sharpe": m_schg[2],
        },
        {
            "Strategy": "3. Hybrid (VTI->SCHG)",
            "CAGR": m_hybrid[0],
            "MDD": m_hybrid[1],
            "Sortino": m_hybrid[3],
            "Sharpe": m_hybrid[2],
        },
    ]
    pd.DataFrame(results).to_csv(
        "d:/gg/data/backtest_results/hybrid_signal_test.csv", index=False
    )
    print(f"\nSaved results to d:/gg/data/backtest_results/hybrid_signal_test.csv")


if __name__ == "__main__":
    run_hybrid_test()
