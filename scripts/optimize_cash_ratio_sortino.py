"""
Analysis: Optimize Cash Ratio in Bear Markets (Sortino Ratio)
Author: Antigravity
Date: 2026-01-31

Goal: Determine the optimal Cash % during "Normal Bear" markets to maximize Sortino Ratio.
Logic:
1. Bull (SPY > MA185 + Buffer): 100% Stock
2. Bear (SPY < MA185 - Buffer):
   A. Inverted Yield: 100% Cash (Fixed Crisis Mode)
   B. Normal Yield:   (1 - X)% Stock + X% Cash
   --> We rely on Yield Curve to distinguish 'Crisis' vs 'Correction'.
   --> We optimize X (Cash Ratio) from 0% to 100%.

Metric: Sortino Ratio = (Annualized Return) / (Downside Deviation)
"""

import yfinance as yf
import pandas as pd
import numpy as np


def calculate_sortino(strategy_rets, risk_free_rate=0.0):
    # Annualized Mean Return
    mean_ret = strategy_rets.mean() * 252

    # Downside Deviation (Annualized)
    # Filter only negative returns
    neg_rets = strategy_rets[strategy_rets < 0]
    downside_std = neg_rets.std() * np.sqrt(252)

    if downside_std == 0:
        return 0

    sortino = (mean_ret - risk_free_rate) / downside_std
    return sortino


def run_cash_opt_sortino():
    print("🚀 Optimizing Cash Ratio (Sortino) (SPY 1993~)...")

    tickers = ["SPY", "^TNX", "^IRX"]
    start_date = "1993-01-01"
    end_date = "2025-12-31"

    print(f"📥 Downloading data...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Flatten
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            df = data["Close"].copy()
        else:
            df = data.copy()
    else:
        df = data.copy()

    df = df.ffill().dropna()
    print(f"Period: {df.index[0].date()} ~ {df.index[-1].date()}")

    # Indicators
    df["MA185"] = df["SPY"].rolling(window=185).mean()
    df["Spread"] = df["^TNX"] - df["^IRX"]

    # Buffer Logic (Fixed)
    buffer = 0.03
    upper = df["MA185"] * (1 + buffer)
    lower = df["MA185"] * (1 - buffer)

    prices = df["SPY"].values
    u_vals = upper.values
    l_vals = lower.values

    # Trend Signal (1=Bull, 0=Bear)
    trend = np.zeros(len(df))
    current_trend = 1 if prices[0] > u_vals[0] else 0

    for i in range(len(df)):
        p = prices[i]
        if p > u_vals[i]:
            current_trend = 1
        elif p < l_vals[i]:
            current_trend = 0
        trend[i] = current_trend

    df["Trend"] = trend
    df["Rets"] = df["SPY"].pct_change().fillna(0)

    results = []

    # Sweep Cash Ratio for Normal Bear (0% to 100% step 10%)
    cash_ratios = np.arange(0.0, 1.1, 0.1)  # 0.0, 0.1 ... 1.0

    for cr in cash_ratios:
        # Calculate Weight of SPY
        # Bull: 1.0
        # Bear (Inverted): 0.0 (100% Cash)
        # Bear (Normal): 1.0 - cr

        test_w_spy = np.zeros(len(df))
        spreads = df["Spread"].values
        trends = df["Trend"].values

        for i in range(len(df)):
            if trends[i] == 1:
                test_w_spy[i] = 1.0
            else:
                if spreads[i] < 0:  # Inverted -> Crisis
                    test_w_spy[i] = 0.0
                else:  # Normal Bear -> Correction
                    test_w_spy[i] = 1.0 - cr

        # Shift
        pos = pd.Series(test_w_spy, index=df.index).shift(1).fillna(0)

        # Returns
        strat_ret = df["Rets"] * pos

        # Metrics
        sortino = calculate_sortino(strat_ret)

        cum = (1 + strat_ret).cumprod()
        final_val = cum.iloc[-1]
        years = len(df) / 252
        cagr = final_val ** (1 / years) - 1

        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        sharpe = (
            (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
            if strat_ret.std() != 0
            else 0
        )

        results.append(
            {
                "Cash_Ratio": cr,
                "Sortino": sortino,
                "CAGR": cagr,
                "MDD": mdd,
                "Sharpe": sharpe,
            }
        )

    res_df = pd.DataFrame(results).sort_values("Sortino", ascending=False)

    print("\n" + "=" * 80)
    print("📊 Bear Market Cash Ratio Optimization (Sorted by Sortino)")
    print("=" * 80)
    print(
        res_df.to_string(
            index=False,
            formatters={
                "Cash_Ratio": "{:.0%}".format,
                "Sortino": "{:.3f}".format,
                "CAGR": "{:.2%}".format,
                "MDD": "{:.2%}".format,
                "Sharpe": "{:.2f}".format,
            },
        )
    )

    res_df.to_csv("d:/gg/cash_ratio_sortino_results.csv", index=False)


if __name__ == "__main__":
    run_cash_opt_sortino()
