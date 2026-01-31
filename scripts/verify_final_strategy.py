"""
Analysis: Final Strategy Verification (The "Real" Version)
Author: Antigravity
Date: 2026-01-31

Components:
1. Signal: MA185 + 3% Buffer (Hysteresis)
2. Logic (Bear Mode):
   - Yield Curve Inverted (10Y-3M < 0): 100% Cash (Defensive)
   - Yield Curve Normal: 50% SPY + 50% Cash (Hedge)
3. Asset: SPY (1993-2025) for Long-term test.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_final_verification():
    print("🚀 Running Final Strategy Verification (MA185 + 3% Buffer + 50% Hedge)...")

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

    # Indicators
    df["MA185"] = df["SPY"].rolling(window=185).mean()
    df["Spread"] = df["^TNX"] - df["^IRX"]

    # Buffer Logic
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

    # Strategy Weights
    # Bull (Trend=1): 100% SPY
    # Bear (Trend=0):
    #   if Spread < 0: 100% Cash (W_SPY=0)
    #   else: 50% SPY + 50% Cash (W_SPY=0.5)

    w_spy = np.zeros(len(df))

    spreads = df["Spread"].values
    trends = df["Trend"].values

    for i in range(len(df)):
        if trends[i] == 1:
            w_spy[i] = 1.0
        else:
            if spreads[i] < 0:  # Inverted
                w_spy[i] = 0.0  # 100% Cash
            else:
                w_spy[i] = 0.5  # 50% Hedge

    # Shift weights (Today's decision -> Tomorrow's return)
    # Assuming Cash Return = 0 for simplicity (or Risk Free Proxy)
    # Let's use 0 to be conservative/clean.

    df["Pos"] = pd.Series(w_spy, index=df.index).shift(1).fillna(0)
    df["Rets"] = df["SPY"].pct_change().fillna(0)

    df["Strat_Ret"] = df["Rets"] * df["Pos"]

    # Metrics
    cum = (1 + df["Strat_Ret"]).cumprod()
    final_val = cum.iloc[-1]
    years = len(df) / 252
    cagr = final_val ** (1 / years) - 1

    roll_max = cum.cummax()
    mdd = ((cum - roll_max) / roll_max).min()

    # Recovery
    is_underwater = (cum - roll_max) / roll_max < 0
    groups = (is_underwater != is_underwater.shift()).cumsum()
    underwater_groups = groups[is_underwater]
    max_rec = 0
    if not underwater_groups.empty:
        durations = []
        for _, g in df.groupby(groups):
            # Ensure it's underwater
            if (cum.loc[g.index[0]] - roll_max.loc[g.index[0]]) < 0:
                durations.append((g.index[-1] - g.index[0]).days)
        if durations:
            max_rec = max(durations)

    sharpe = (df["Strat_Ret"].mean() / df["Strat_Ret"].std()) * np.sqrt(252)

    # Trades (Trend changes only? Or Weight changes?)
    # Weight changes are more accurate for cost.
    trades = df["Pos"].diff().abs().sum() / 2

    print("\n" + "=" * 60)
    print("🏁 FINAL STRATEGY: MA185 + 3% Buffer + Yield Logic")
    print("=" * 60)
    print(f"Period: {df.index[0].date()} ~ {df.index[-1].date()}")
    print("-" * 60)
    print(f"Final Value: {final_val:.2f}x (Initial 1.0)")
    print(f"CAGR:       {cagr * 100:.2f}%")
    print(f"MDD:        {mdd * 100:.2f}%")
    print(f"Recovery:   {max_rec:,} days")
    print(f"Sharpe:     {sharpe:.2f}")
    print(f"Trades:     {int(trades)}")
    print("-" * 60)

    # Compare to 100% Switch Version (from previous step)
    # Re-calc simply
    # 100% Switch: if Trend=0 -> Pos=0. No 50% logic.
    pos_switch = pd.Series(trend, index=df.index).shift(1).fillna(0)
    ret_switch = df["Rets"] * pos_switch
    cum_switch = (1 + ret_switch).cumprod()
    cagr_switch = cum_switch.iloc[-1] ** (1 / years) - 1
    mdd_switch = ((cum_switch - cum_switch.cummax()) / cum_switch.cummax()).min()

    print("vs. 100% Switch Version (No 50% Hedge):")
    print(f"CAGR: {cagr_switch * 100:.2f}%")
    print(f"MDD:  {mdd_switch * 100:.2f}%")


if __name__ == "__main__":
    run_final_verification()
