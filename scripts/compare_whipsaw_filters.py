"""
Analysis: Whipsaw Protection for MA185 Strategy
Author: Antigravity
Date: 2026-01-31

Goal: Compare auxiliary strategies to reduce 'Whipsaw' (false signals).

Strategies to Test:
1. Base: VTI < MA185 (Daily close)
2. Time Delay (Wait): Signal must persist for N=3 days to switch.
3. Price Threshold (Buffer): Price must be X% (e.g., 1%, 3%) below MA to sell.
4. Double MA (Golden Cross): MA50 < MA200 (Slower but smoother).
5. RSI Filter: Only Sell if RSI < 50 (Momentum confirmation).
6. VIX Filter: Only Sell if VIX > 20 (High volatility confirmation).

Metrics:
- CAGR, MDD
- Number of Trades (Crucial for Whipsaw analysis)
- Win Rate
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_whipsaw_analysis():
    print("🚀 Comparing Whipsaw Protection Strategies...")

    tickers = ["VTI", "^VIX"]
    start_date = "2000-01-01"
    end_date = "2025-12-31"

    print(f"📥 Downloading data...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Flatten Helper
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            df = data["Close"].copy()
        else:
            df = data.copy()
    else:
        df = data.copy()

    df = df.ffill().dropna()

    # Basic Indicators
    df["MA185"] = df["VTI"].rolling(window=185).mean()
    df["MA50"] = df["VTI"].rolling(window=50).mean()
    df["MA200"] = df["VTI"].rolling(window=200).mean()

    # RSI
    delta = df["VTI"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)

    # Returns
    df["VTI_Pct"] = df["VTI"].pct_change()

    # Define Logic Check Function
    def backtest_logic(signal_series, name):
        # signal: 1 = Invested, 0 = Cash
        # Shift signal
        pos = signal_series.shift(1).fillna(1)

        # Returns (Assume 0% return for Cash for simplicity to isolate alpha)
        # Or should we use BIL? For comparing "Whipsaw itself", 0% cash is cleaner.
        # But we used BIL before. Let's stick to 0% to focus on trade mechanics.
        strat_ret = df["VTI_Pct"] * pos

        cum = (1 + strat_ret).cumprod()
        final = cum.iloc[-1]
        years = len(df) / 252
        cagr = final ** (1 / years) - 1
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()

        # Count Trades (Change in position)
        trades = pos.diff().abs().sum() / 2  # Buy+Sell = 1 trade cycle approx

        return {
            "Strategy": name,
            "CAGR": cagr,
            "MDD": mdd,
            "Trades": int(trades),
            "Final_Val": final,
        }

    results = []

    # 1. Base Strategy (Daily Cross)
    # 1 = Bull, 0 = Bear
    s_base = (df["VTI"] > df["MA185"]).astype(int)
    results.append(backtest_logic(s_base, "Base (MA185 Daily)"))

    # 2. Time Delay (3 Days)
    # Signal must form for 3 consecutive days to flip
    # We use rolling min/max logic or looping
    # Rolling min of boolean (1=True) over 3 days == 1 => Bull confirmed
    # Rolling max of boolean (0=False) over 3 days == 0 => Bear confirmed
    # This is tricky vectorized. Let's use simple logic:
    # If 3 days > MA -> Buy. If 3 days < MA -> Sell. Else Hold previous.

    def apply_delay(delay=3):
        raw_bull = df["VTI"] > df["MA185"]
        # Rolling sum of True. If 3, then Bull. If 0, then Bear.
        roll = raw_bull.rolling(window=delay).sum()

        sigs = np.zeros(len(df))
        current = 1  # Start invested

        for i in range(len(df)):
            r = roll.iloc[i]
            if r == delay:  # All True
                current = 1
            elif r == 0:  # All False
                current = 0
            # Else keep current
            sigs[i] = current
        return pd.Series(sigs, index=df.index)

    s_delay3 = apply_delay(3)
    results.append(backtest_logic(s_delay3, "Delay (3 Days)"))

    s_delay5 = apply_delay(5)
    results.append(backtest_logic(s_delay5, "Delay (5 Days)"))

    # 3. Price Threshold (Buffer 1%, 3%)
    # Buy if VTI > MA * 1.01
    # Sell if VTI < MA * 0.99 (Hysteresis Band)

    def apply_buffer(pct=0.01):
        sigs = np.zeros(len(df))
        current = 1

        upper = df["MA185"] * (1 + pct)
        lower = df["MA185"] * (1 - pct)

        prices = df["VTI"].values
        u_vals = upper.values
        l_vals = lower.values

        for i in range(len(df)):
            p = prices[i]
            if p > u_vals[i]:
                current = 1
            elif p < l_vals[i]:
                current = 0
            sigs[i] = current

        return pd.Series(sigs, index=df.index)

    s_buf1 = apply_buffer(0.01)
    results.append(backtest_logic(s_buf1, "Buffer (+/- 1%)"))

    s_buf3 = apply_buffer(0.03)
    results.append(backtest_logic(s_buf3, "Buffer (+/- 3%)"))

    # 4. Double MA (Golden Cross 50/200)
    s_gc = (df["MA50"] > df["MA200"]).astype(int)
    results.append(backtest_logic(s_gc, "Golden Cross (50/200)"))

    # 5. RSI Filter
    # Base is MA185. But DO NOT SELL if RSI > 40 (Oversold bounce likely? No)
    # Standard logic: Sell only if Momentum confirms (RSI < 50)
    # Buy only if Momentum confirms (RSI > 50)
    # Let's try: Sell condition = Price < MA185 AND RSI < 50
    # Buy condition = Price > MA185 AND RSI > 50

    def apply_rsi_filter(level=50):
        sigs = np.zeros(len(df))
        current = 1

        price_bull = df["VTI"] > df["MA185"]
        rsi_high = df["RSI"] > level

        p = price_bull.values
        r = rsi_high.values

        for i in range(len(df)):
            is_bull = p[i]
            is_rsi_high = r[i]

            # Switch to Bull if Price > MA AND RSI > 50
            if is_bull and is_rsi_high:
                current = 1
            # Switch to Bear if Price < MA AND RSI < 50
            elif (not is_bull) and (not is_rsi_high):
                current = 0
            sigs[i] = current
        return pd.Series(sigs, index=df.index)

    s_rsi = apply_rsi_filter(50)
    results.append(backtest_logic(s_rsi, "RSI Confirm (50)"))

    # 6. VIX Filter
    # Don't sell if VIX is extremely high? (Fear peak usually means bottom)
    # Strategy: Sell only if VIX < 30. If VIX > 30, it's panic, hold for bounce?
    # Or strict: Sell if Price < MA. Exception: If VIX > 35, Hold (Anti-panic).

    def apply_vix_filter(panic_level=35):
        sigs = np.zeros(len(df))
        current = 1

        price_bull = df["VTI"] > df["MA185"]
        panic = df["^VIX"] > panic_level

        p = price_bull.values
        v = panic.values

        for i in range(len(df)):
            if p[i]:  # Bull signal
                current = 1
            else:  # Bear signal
                if v[i]:  # Panic mode!
                    # Don't sell during panic peak??
                    # Actually historically, selling at VIX 40 is usually bottom ticking.
                    # So if Bear signal AND VIX > 35, Ignore Sell (Force Hold / Buy)
                    current = 1  # Forced Bull
                else:
                    current = 0  # Normal Sell
            sigs[i] = current
        return pd.Series(sigs, index=df.index)

    s_vix = apply_vix_filter(33)  # 2008 and 2020 peaks were > 30-40
    results.append(backtest_logic(s_vix, "VIX Filter (Don't Sell > 33)"))

    # Summary
    res_df = pd.DataFrame(results).sort_values("CAGR", ascending=False)

    print("\n" + "=" * 80)
    print("🧹 Whipsaw Protection Strategy Comparison (2000-2025)")
    print("=" * 80)
    print(
        res_df.to_string(
            index=False,
            formatters={
                "CAGR": "{:.2%}".format,
                "MDD": "{:.2%}".format,
                "Final_Val": "{:.2f}".format,
            },
        )
    )

    res_df.to_csv("d:/gg/whipsaw_analysis.csv", index=False)


if __name__ == "__main__":
    run_whipsaw_analysis()
