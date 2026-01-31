"""
Strategy Comparison: Replacing 'Cash' in MA185 Strategy
Author: Antigravity
Date: 2026-01-31

Logic:
1. VTI > MA185: 100% VTI
2. VTI < MA185:
   A. Spread < 0 (Recession): 100% [DEFENSE_ASSET]
   B. Spread >= 0 (Correction): 50% VTI + 50% [DEFENSE_ASSET]

Candidates:
- BIL (Cash Proxy)
- IEF (7-10y Treasury)
- TLT (20y+ Treasury)
- GLD (Gold)
- XLP (Consumer Staples)
- SCHD (Dividend - shorter history, will limit start date if used. Let's skip or handle carefully. Start 2011)
- LQD (Corporate Bond)
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_comparison():
    print("🚀 Comparing Defensive Assets for MA185 Strategy...")

    # Candidates
    defense_tickers = ["BIL", "IEF", "TLT", "GLD", "XLP", "SCHD", "LQD"]
    # VTI + Macro
    base_tickers = ["VTI", "^TNX", "^IRX"]

    start_date = "2005-01-01"  # Most ETFs available (except SCHD)
    end_date = "2025-12-31"

    all_tickers = base_tickers + defense_tickers
    print(f"📥 Downloading data for {all_tickers}...")
    data = yf.download(all_tickers, start=start_date, end=end_date)["Close"]
    data = data.ffill()

    # Indicators
    data["MA185"] = data["VTI"].rolling(window=185).mean()
    data["Spread"] = data["^TNX"] - data["^IRX"]

    # Base Signals
    cond_bull = data["VTI"] > data["MA185"]
    cond_inverted = data["Spread"] < 0
    mask_bear = ~cond_bull

    # Returns
    returns_df = data.pct_change()

    results = []

    for asset in defense_tickers:
        # Check data availability
        first_valid = data[asset].first_valid_index()
        if first_valid is None:
            print(f"Skipping {asset} (No Data)")
            continue

        # Cut dataframe to start when asset is available
        sim_data = data.loc[first_valid:].copy()
        sim_rets = returns_df.loc[first_valid:].copy()

        # Re-calc signals for aligned index
        # (Signal calculation depends on VTI history which is long, so we can use original data and align)
        # But let's keep it simple

        # Weights
        w_vti = pd.Series(0.0, index=sim_data.index)
        w_def = pd.Series(0.0, index=sim_data.index)

        # Logic
        # Bull -> 100% VTI
        bull_idx = cond_bull.loc[sim_data.index]
        w_vti[bull_idx] = 1.0

        # Bear + Inverted -> 100% Defense
        bear_inv_idx = (mask_bear & cond_inverted).loc[sim_data.index]
        w_def[bear_inv_idx] = 1.0

        # Bear + Normal -> 50% VTI + 50% Defense
        bear_norm_idx = (mask_bear & ~cond_inverted).loc[sim_data.index]
        w_vti[bear_norm_idx] = 0.5
        w_def[bear_norm_idx] = 0.5

        # Shift
        w_vti = w_vti.shift(1).fillna(0)
        w_def = w_def.shift(1).fillna(0)

        # Strategy Return
        strat_ret = (sim_rets["VTI"] * w_vti) + (sim_rets[asset] * w_def)

        # Metrics
        initial = 10000
        cum_ret = (1 + strat_ret).cumprod()
        final_val = cum_ret.iloc[-1] * initial

        years = len(sim_data) / 252
        cagr = (final_val / initial) ** (1 / years) - 1

        cummax = cum_ret.cummax()
        mdd = ((cum_ret - cummax) / cummax).min()

        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)

        results.append(
            {
                "Asset": asset,
                "Values": f"${final_val:,.0f}",
                "CAGR": cagr,
                "MDD": mdd,
                "Sharpe": sharpe,
                "History": f"{sim_data.index[0].date()}~",
            }
        )

    # Convert to DF
    res_df = pd.DataFrame(results).sort_values("CAGR", ascending=False)

    print("\n" + "=" * 60)
    print("🏆 Best Defense Asset Comparison")
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
    res_df.to_csv("d:/gg/defense_asset_comparison.csv", index=False)


if __name__ == "__main__":
    run_comparison()
