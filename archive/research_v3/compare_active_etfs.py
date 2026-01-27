# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np


def calc_stats(series):
    if series.empty:
        return {"CAGR": 0, "MDD": 0}

    # CAGR
    days = (series.index[-1] - series.index[0]).days
    if days < 365:
        # Less than 1 year, just return total return
        ret = (series.iloc[-1] / series.iloc[0]) - 1
        return {"CAGR": ret * 100, "MDD": 0, "Total": ret * 100}

    tr = series.iloc[-1] / series.iloc[0]
    cagr = (tr ** (365 / days)) - 1

    # MDD
    peak = series.cummax()
    dd = (series - peak) / peak
    mdd = dd.min()

    return {"CAGR": cagr * 100, "MDD": mdd * 100, "Total": (tr - 1) * 100}


def compare_performance():
    tickers = [
        "QQQ",
        "FBCG",
        "TCHP",
        "BTEK",
        "QLD",
    ]  # ARKK excluded to focus on user request
    print(f"Fetching data for: {tickers}...")

    # Recent 3 years (approx 2023-2026)
    data = yf.download(
        tickers, start="2023-01-01", progress=False, group_by="ticker", auto_adjust=True
    )

    # Extract Close
    df = pd.DataFrame()
    for t in tickers:
        try:
            if t in data.columns.levels[0]:
                series = data[t]["Close"]
            else:
                series = data["Close"][t]

            # Check availability
            first = series.first_valid_index()
            last = series.last_valid_index()
            if first is not None:
                print(f"[{t}] Data: {first.date()} ~ {last.date()}")
                df[t] = series
        except:
            pass

    # Handle flat structure fallback
    if df.empty and "Close" in data.columns:
        df = data["Close"]

    df = df.dropna()
    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date = df.index[-1].strftime("%Y-%m-%d")

    print(f"\nTime Period: {start_date} ~ {end_date}")
    print("-" * 65)
    print(f"{'ETF':<6} | {'Name':<15} | {'CAGR':<8} | {'MDD':<8} | {'Total Ret':<8}")
    print("-" * 65)

    # Normalize to 100 for graph-like comparison in head
    norm = df / df.iloc[0] * 100

    for t in tickers:
        if t not in df.columns:
            continue
        stats = calc_stats(df[t])

        name_map = {
            "QQQ": "Nasdaq 100",
            "FBCG": "Fidelity Active",
            "TCHP": "T.Rowe Active",
            "BTEK": "BlackRock Tech",
            "QLD": "ProShares 2x",
        }

        print(
            f"{t:<6} | {name_map.get(t, t):<15} | {stats['CAGR']:<7.2f}% | {stats['MDD']:<7.2f}% | {stats['Total']:<7.1f}%"
        )

    print("-" * 65)
    print("\n[Analysis]")

    # Quick diff
    qqq_cagr = calc_stats(df["QQQ"])["CAGR"]
    fbcg_cagr = calc_stats(df["FBCG"])["CAGR"]

    if fbcg_cagr > qqq_cagr:
        print(f"✅ FBCG beat QQQ by {fbcg_cagr - qqq_cagr:.2f}%p annually.")
    else:
        print(f"❌ FBCG trailed QQQ by {qqq_cagr - fbcg_cagr:.2f}%p annually.")


if __name__ == "__main__":
    compare_performance()
