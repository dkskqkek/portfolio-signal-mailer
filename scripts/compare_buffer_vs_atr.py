"""
Analysis: Fixed Buffer vs ATR Filter + 2-Day Confirmation
Author: Antigravity
Date: 2026-01-31

Goal: Compare two trend filter logics under V4 allocation rules (Bear=30% Stock).

Strategy A (Current V4):
- Signal: Close > MA185 * 1.03 (Buy), Close < MA185 * 0.97 (Sell)
- Confirmation: Immediate (1-day)

Strategy B (Challenger):
- Signal: Close > MA185 + 3*ATR (Buy), Close < MA185 - 3*ATR (Sell)
- Confirmation: 2-Day (Signal must hold for 2 consecutive days)
- Why? Dynamic buffer adapts to volatility. 2-Day reduces 'fake' breakouts.

Common Logic (Bear Allocation):
- Inverted Yield: 0% Stock (Crisis)
- Normal Yield: 30% Stock (Correction - 70% Cash)
- Bull: 100% Stock
"""

import yfinance as yf
import pandas as pd
import numpy as np


def calculate_atr(df, window=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr


def run_comparison():
    print("🚀 Comparing V4 (Fixed 3%) vs Challenger (ATR + 2-Day)...")

    tickers = ["SPY", "^TNX", "^IRX"]
    start_date = "1993-01-01"
    end_date = "2025-12-31"

    print(f"📥 Downloading data...")
    # Need High/Low for ATR
    data = yf.download(
        tickers, start=start_date, end=end_date, progress=False, group_by="ticker"
    )

    # Process Data
    # structure: data['SPY']['Close'], data['^TNX']['Close']...
    # yfinance multi-level columns can be tricky.

    df = pd.DataFrame()
    df["SPY_Close"] = data["SPY"]["Close"]
    df["SPY_High"] = data["SPY"]["High"]
    df["SPY_Low"] = data["SPY"]["Low"]
    df["TNX"] = data["^TNX"]["Close"]
    df["IRX"] = data["^IRX"]["Close"]

    df = df.ffill().dropna()

    # Common Indicators
    df["MA185"] = df["SPY_Close"].rolling(window=185).mean()
    df["Spread"] = df["TNX"] - df["IRX"]

    # ATR Calculation
    # Need helper DF for High/Low/Close
    atr_df = pd.DataFrame(
        {"High": df["SPY_High"], "Low": df["SPY_Low"], "Close": df["SPY_Close"]}
    )
    df["ATR"] = calculate_atr(atr_df, window=14)  # Standard 14

    # Prep Logic
    prices = df["SPY_Close"].values
    ma = df["MA185"].values
    atr = df["ATR"].values
    spreads = df["Spread"].values

    # Need to start after valid indices
    valid_idx = max(185, 14)
    df = df.iloc[valid_idx:].copy()
    prices = df["SPY_Close"].values
    ma = df["MA185"].values
    atr = df["ATR"].values
    spreads = df["Spread"].values

    # --- Strategy A: Fixed 3% Buffer (Immediate) ---
    buffer_fixed = 0.03
    upper_fixed = ma * (1 + buffer_fixed)
    lower_fixed = ma * (1 - buffer_fixed)

    state_a = np.zeros(len(df))
    curr_a = 1  # Start Bull

    for i in range(len(df)):
        if prices[i] > upper_fixed[i]:
            curr_a = 1
        elif prices[i] < lower_fixed[i]:
            curr_a = 0
        state_a[i] = curr_a

    # --- Strategy B: 3*ATR + 2-Day Confirm ---
    k_atr = 3.0
    upper_atr = ma + (k_atr * atr)
    lower_atr = ma - (k_atr * atr)

    state_b = np.zeros(len(df))
    curr_b = 1

    # For confirmation, we need to track tentative signals
    # Logic:
    # If Bull -> Limit is Lower Band. If P < Lower for 2 days -> Bear.
    # If Bear -> Limit is Upper Band. If P > Upper for 2 days -> Bull.

    # We can count consecutive breach days
    consecutive_breach = 0

    for i in range(len(df)):
        if curr_b == 1:  # Bull Mode
            if prices[i] < lower_atr[i]:
                consecutive_breach += 1
            else:
                consecutive_breach = 0  # Reset if price bounces back

            if consecutive_breach >= 2:  # Confirmed Sell
                curr_b = 0
                consecutive_breach = 0

        else:  # Bear Mode
            if prices[i] > upper_atr[i]:
                consecutive_breach += 1
            else:
                consecutive_breach = 0

            if consecutive_breach >= 2:  # Confirmed Buy
                curr_b = 1
                consecutive_breach = 0

        state_b[i] = curr_b

    # --- Metrics Logic ---
    rets = df["SPY_Close"].pct_change().fillna(0)

    def calc_performance(states, label):
        weights = np.zeros(len(df))
        trend_changes = 0
        prev_trend = states[0]

        for i in range(len(df)):
            trend = states[i]

            if trend != prev_trend:
                trend_changes += 1
                prev_trend = trend

            if trend == 1:  # Bull
                weights[i] = 1.0
            else:  # Bear
                if spreads[i] < 0:  # Crisis
                    weights[i] = 0.0
                else:  # Correction
                    weights[i] = 0.3  # 30% Stock

        pos = pd.Series(weights, index=df.index).shift(1).fillna(0)
        strat = rets * pos

        cum = (1 + strat).cumprod()
        final_val = cum.iloc[-1]
        years = len(df) / 252
        cagr = final_val ** (1 / years) - 1

        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        sharpe = (strat.mean() / strat.std()) * np.sqrt(252) if strat.std() != 0 else 0

        return {
            "Label": label,
            "CAGR": cagr,
            "MDD": mdd,
            "Sharpe": sharpe,
            "Trades": trend_changes,  # Rough count of major trend shifts
            "Final": final_val,
        }

    res_a = calc_performance(state_a, "V4: Fixed 3% (Current)")
    res_b = calc_performance(state_b, "Challenger: 3*ATR + 2-Day")

    results = pd.DataFrame([res_a, res_b])

    print("\n" + "=" * 80)
    print("⚔️ Strategy Duel: Fixed Buffer vs Dynamic ATR")
    print("=" * 80)
    print(
        results.to_string(
            index=False,
            formatters={
                "CAGR": "{:.2%}".format,
                "MDD": "{:.2%}".format,
                "Sharpe": "{:.2f}".format,
                "Final": "${:,.2f}".format,
            },
        )
    )

    results.to_csv("d:/gg/buffer_vs_atr_results.csv", index=False)

    # Deep Dive Explanation
    print("-" * 80)
    if res_b["CAGR"] > res_a["CAGR"]:
        print("💡 ATR Wins on Return. Adaptive volatility captures more upside.")
    else:
        print("💡 Fixed 3% Wins on Return. Simplicity beats complexity here.")

    if res_b["Trades"] < res_a["Trades"]:
        print("💡 ATR + 2-Day reduces trades (Whipsaw reduction successful).")
    else:
        print("💡 Fixed 3% is surprisingly stable (ATR might be too noisy signals).")


if __name__ == "__main__":
    run_comparison()
