"""
Analysis: Finding the 'MDD King' Defensive Asset
Author: Antigravity
Date: 2026-01-31

Goal: Find an ETF that, when used instead of Cash in the MA185 Strategy, results in the LOWEST Portfolio MDD.

Strategy Logic:
1. VTI > MA185: 100% VTI
2. VTI < MA185:
   A. Inverted: 100% [Candidate] (Aggressive Defense)
   B. Normal: 50% VTI + 50% [Candidate] (Hedge)

Candidates:
- Treasuries: SHY (1-3y), IEF (7-10y), TLT (20y+), EDV (Extended), ZROZ (Strips)
- Commodities: GLD (Gold), DBC (Commodity Index)
- Currencies: UUP (USD Bull - Cash Proxy but anti-market)
- Defensive Equity: XLP (Staples), XLU (Utilities)
- Cash: BIL (Baseline)

Period: 2008-01-01 ~ 2025-12-31 (To cover 2008 & 2022)
"""

import yfinance as yf
import pandas as pd
import numpy as np


def find_mdd_king():
    print("🚀 searching for the MDD King (Lowest Drawdown)...")

    # Candidates
    candidates = [
        "BIL",
        "SHY",
        "IEF",
        "TLT",
        "EDV",
        "ZROZ",
        "GLD",
        "DBC",
        "UUP",
        "XLP",
        "XLU",
    ]
    base_tickers = ["VTI", "^TNX", "^IRX"]

    all_tickers = base_tickers + candidates
    start_date = "2007-06-01"  # Limited by ZROZ? ZROZ is 2009. EDV 2007.
    # To include 2008, we might need to skip ZROZ/EDV if they start later.
    # EDV inception Dec 2007. ZROZ Oct 2009.
    # Trying to cover 2008 is crucial. Let's exclude ZROZ for 2008 test.
    # Keep EDV, but check if data exists.

    print(f"📥 Downloading data...")
    data = yf.download(
        all_tickers, start="2007-01-01", end="2025-12-31", progress=False
    )

    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            df_all = data["Close"].copy()
        else:
            df_all = data.copy()
    else:
        df_all = data.copy()

    df_all = df_all.ffill()

    results = []

    for asset in candidates:
        if asset not in df_all.columns:
            print(f"Skipping {asset} (Missing)")
            continue

        # Slice for asset history
        # We want common period if possible, but 2008 is key.
        # If asset starts after 2008, we can't compare MDD fairness for 2008 crisis.
        # So we stick to assets available since Jan 2008.
        valid_idx = df_all[asset].first_valid_index()
        if valid_idx > pd.Timestamp("2008-02-01"):
            print(f"Skipping {asset} (Starts too late: {valid_idx.date()})")
            continue

        df = df_all.loc["2007-06-01":].copy()

        # Indicators
        df["MA185"] = df["VTI"].rolling(window=185).mean()
        df["Spread"] = df["^TNX"] - df["^IRX"]
        df.dropna(inplace=True)

        # Logic
        df["W_VTI"] = 0.0
        df["W_DEF"] = 0.0

        cond_bull = df["VTI"] > df["MA185"]
        cond_inverted = df["Spread"] < 0
        mask_bear = ~cond_bull

        # 1. Bull
        df.loc[cond_bull, "W_VTI"] = 1.0

        # 2. Bear + Inverted -> 100% Defense
        df.loc[mask_bear & cond_inverted, "W_DEF"] = 1.0

        # 3. Bear + Normal -> 50% VTI + 50% Defense
        df.loc[mask_bear & ~cond_inverted, "W_VTI"] = 0.5
        df.loc[mask_bear & ~cond_inverted, "W_DEF"] = 0.5

        # Shift
        df["W_VTI"] = df["W_VTI"].shift(1)
        df["W_DEF"] = df["W_DEF"].shift(1)
        df.dropna(inplace=True)

        # Returns
        rets = df.pct_change()
        strat_ret = (rets["VTI"] * df["W_VTI"]) + (rets[asset] * df["W_DEF"])

        # Metrics
        cum = (1 + strat_ret).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        cagr = (cum.iloc[-1]) ** (1 / (len(df) / 252)) - 1

        results.append(
            {
                "Asset": asset,
                "Strategy_MDD": mdd,
                "Strategy_CAGR": cagr,
                "Final_Value": cum.iloc[-1],
            }
        )

    res_df = pd.DataFrame(results).sort_values(
        "Strategy_MDD", ascending=False
    )  # Closer to 0 is better

    print("\n" + "=" * 60)
    print("🛡️ MDD King Ranking (2008 & 2022 Included)")
    print("=" * 60)
    print(
        res_df.to_string(
            index=False,
            formatters={
                "Strategy_MDD": "{:.2%}".format,
                "Strategy_CAGR": "{:.2%}".format,
                "Final_Value": "{:.2f}".format,
            },
        )
    )

    # Save
    res_df.to_csv("d:/gg/mdd_king_results.csv", index=False)


if __name__ == "__main__":
    find_mdd_king()
