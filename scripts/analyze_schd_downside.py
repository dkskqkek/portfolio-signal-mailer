import yfinance as yf
import pandas as pd
import numpy as np
import json


def analyze_downside():
    print("Analyze: SCHD vs Cash (Flattened)")

    tickers = ["VTI", "SCHD", "BIL"]
    # 2011-10-20 start
    data = yf.download(tickers, start="2011-10-20", end="2025-12-31", progress=False)

    # Flatten columns if MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        # Prefer 'Close'
        if "Close" in data.columns.levels[0]:
            df = data["Close"].copy()
        else:
            # Maybe it's just tickers if auto_adjust=True
            df = data.copy()
    else:
        df = data.copy()

    df = df.ffill().dropna()
    print(f"Data Shape: {df.shape}")

    # Check tickers
    for t in tickers:
        if t not in df.columns:
            print(f"Missing {t}")
            return

    # MA185
    df["MA185"] = df["VTI"].rolling(window=185).mean()
    df.dropna(inplace=True)

    # Bear Signal
    df["Is_Bear"] = df["VTI"] < df["MA185"]

    # Returns
    rets = df[tickers].pct_change().dropna()
    # Align mask
    bear_mask = df.loc[rets.index, "Is_Bear"]

    # Correlation
    bear_rets = rets[bear_mask]
    corr = bear_rets.corr().loc["VTI"]

    print("-" * 50)
    print("Bear Market Correlation (VTI vs X):")
    print(f"  SCHD: {corr['SCHD']:.2f}")
    print(f"  BIL:  {corr['BIL']:.2f}")

    # COVID
    cov_rets = rets.loc["2020-02-19":"2020-03-23"]
    if not cov_rets.empty:
        vti_c = (1 + cov_rets["VTI"]).cumprod().iloc[-1] - 1
        schd_c = (1 + (0.5 * cov_rets["VTI"] + 0.5 * cov_rets["SCHD"])).cumprod().iloc[
            -1
        ] - 1
        cash_c = (1 + (0.5 * cov_rets["VTI"] + 0.5 * cov_rets["BIL"])).cumprod().iloc[
            -1
        ] - 1

        print("\nCOVID (Feb-Mar 2020):")
        print(f"  VTI: {vti_c * 100:.2f}%")
        print(f"  50/50 SCHD: {schd_c * 100:.2f}%")
        print(f"  50/50 Cash: {cash_c * 100:.2f}%")

    # 2022
    bear_22 = rets.loc["2022-01-03":"2022-10-12"]
    if not bear_22.empty:
        vti_b = (1 + bear_22["VTI"]).cumprod().iloc[-1] - 1
        schd_b = (1 + (0.5 * bear_22["VTI"] + 0.5 * bear_22["SCHD"])).cumprod().iloc[
            -1
        ] - 1
        cash_b = (1 + (0.5 * bear_22["VTI"] + 0.5 * bear_22["BIL"])).cumprod().iloc[
            -1
        ] - 1

        print("\n2022 Bear (Jan-Oct):")
        print(f"  VTI: {vti_b * 100:.2f}%")
        print(f"  50/50 SCHD: {schd_b * 100:.2f}%")
        print(f"  50/50 Cash: {cash_b * 100:.2f}%")


if __name__ == "__main__":
    analyze_downside()
