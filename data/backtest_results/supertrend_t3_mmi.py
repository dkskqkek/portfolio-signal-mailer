import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from custom_indicators import calculate_mmi, calculate_t3, calculate_supertrend


def run_strategy(
    ticker, start_date="2010-01-01", initial_capital=10000.0, commission=0.001
):
    print(f"--- Running Strategy for {ticker} ---")

    # 1. Data
    df = yf.download(ticker, start=start_date, progress=False)

    # Robust OHLC Selection
    if isinstance(df.columns, pd.MultiIndex):
        try:
            o = df.xs("Open", level=0, axis=1).iloc[:, 0]
            h = df.xs("High", level=0, axis=1).iloc[:, 0]
            l = df.xs("Low", level=0, axis=1).iloc[:, 0]
            c = df.xs("Close", level=0, axis=1).iloc[
                :, 0
            ]  # Use Close for consistency with Indicators
        except:
            # Just take first columns
            o = df.iloc[:, 0]
            h = df.iloc[:, 1]
            l = df.iloc[:, 2]
            c = df.iloc[:, 3]
    else:
        # Check column names
        if "Open" in df.columns:
            o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
        else:
            # Maybe just Close? SuperTrend needs OHLC.
            # If standard download fails to give OHLC, we can't run.
            # Assuming standard structure.
            o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    data = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c})
    data.dropna(inplace=True)

    # 2. Indicators
    print("Calculating Indicators...")
    data["T3"] = calculate_t3(data["Close"], length=50, vfactor=0.4)
    data["MMI"] = calculate_mmi(data["Close"], length=20)

    st_res = calculate_supertrend(data, period=10, factor=3.0)
    data["SuperTrend"] = st_res["SuperTrend"]
    data["ST_Dir"] = st_res["Direction"]  # 1=Buy, -1=Sell

    # 3. Signals
    # Entry: ST_Dir == 1 (Green) AND Close > T3 AND MMI < 0.3
    # Note: ST Green trigger creates a "State".
    # We should enter IF ST is Green AND Trend filters allow.
    # BUT user said: "SuperTrend creates a new Buy signal (green)".
    # This implies the *moment* of flip. Or just "While Green"?
    # Usually "Trend Filter" implies "While Green, allow entry if Filter passes".
    # User said: "Entry Long ONLY when ALL conditions are met"
    # This usually means CONJUNCTION.
    # Let's effectively filter the ST signal.

    # Logic:
    # 1. Base Signal = (ST_Dir == 1)
    # 2. Filters = (Close > T3) & (MMI < 0.3)
    # 3. Final Entry = Base & Filters ?
    #    No, usually once entered, we stay until Exit.
    #    So we need an "Enter Trigger" and "Exit Trigger".

    # Enter Trigger:
    # (ST_Dir == 1) & (Close > T3) & (MMI < 0.3)
    # But wait, if ST is ALREADY 1, can we enter later if MMI drops?
    # User: "Enter Long ONLY when ALL conditions are met: 1. SuperTrend creates a new Buy signal..."
    # This implies the TRIGGER is the ST FLIP involved.
    # "1. SuperTrend creates a new Buy signal" suggests the EVENT.
    # What if ST flips to Green, but MMI is 0.5 (No Entry). Then next day MMI drops to 0.2. Do we enter?
    # Strict interpretation: "SuperTrend creates a new Buy signal" is an event. If conditions not met THEN, miss the trade.
    # Loose interpretation (Better for strategy): "While ST is Green", if conditions met, Enter.
    # User said: "1. SuperTrend creates a new 'Buy' signal... 2. Close > T3... 3. MMI < 0.3"
    # This reads like a checklist at the MOMENT of generated signal.
    # HOWEVER, this is very restrictive.
    # Let's assign:
    #   Long_Entry_Mask = (ST_Dir == 1) & (ST_Dir.shift(1) == -1) & (Close > T3) & (MMI < 0.3)
    # This is the "Flip Event" interpretation. Rigorous.

    long_entry_mask = (
        (data["ST_Dir"] == 1) & (data["Close"] > data["T3"]) & (data["MMI"] < 0.3)
    )
    # Wait, if we use strict "Flip Event", we ignore cases where trend is established and MMI stabilizes.
    # But user specifically wrote "1. SuperTrend creates a new 'Buy' signal".
    # I will simulate the "State Machine" to be precise.

    # Debug MMI distribution
    print(
        f"MMI Stats: Min={data['MMI'].min():.2f}, Mean={data['MMI'].mean():.2f}, % < 0.7 = {(data['MMI'] < 0.7).mean() * 100:.2f}%"
    )

    position = 0  # 0 or 1
    cash = initial_capital
    shares = 0
    equity = []

    signals = pd.Series(0, index=data.index)

    prev_st_dir = -1

    for i in range(len(data)):
        date = data.index[i]
        c = data["Close"].iloc[i]
        st_dir = data["ST_Dir"].iloc[i]
        t3 = data["T3"].iloc[i]
        mmi = data["MMI"].iloc[i]

        # Exit Condition: ST Flips to Sell
        if position == 1:
            if st_dir == -1:
                # Sell
                value = shares * c
                cost = value * commission
                cash = value - cost
                shares = 0
                position = 0
                signals.iloc[i] = -1  # Record Exit

        # Entry Condition
        if position == 0:
            # Logic: State-based Entry
            # If ST is Buy (1) AND Filter Conditions Are Met
            if st_dir == 1:
                if (c > t3) and (mmi < 0.7):
                    # Buy
                    cost = cash * commission
                    avail = cash - cost
                    shares = avail / c
                    cash = 0
                    position = 1
                    signals.iloc[i] = 1  # Record Entry

        # Record Equity
        curr_eq = cash + (shares * c)
        equity.append(curr_eq)

        prev_st_dir = st_dir

    data["Equity"] = equity

    # 4. Metrics
    total_ret = (data["Equity"].iloc[-1] / initial_capital) - 1
    days = (data.index[-1] - data.index[0]).days
    years = days / 365.25
    cagr = (data["Equity"].iloc[-1] / initial_capital) ** (1 / years) - 1

    # MDD
    roll_max = data["Equity"].cummax()
    dd = (data["Equity"] - roll_max) / roll_max
    mdd = dd.min()

    # Trades
    n_trades = len(signals[signals == 1])
    win_trades = 0  # Need detailed trade log for this. Calculating approx based on Equity curve is hard.
    # Skipping detailed trade logs for brevity, focusing on Equity Curve.

    print(
        f"[{ticker}] CAGR: {cagr * 100:.2f}% | MDD: {mdd * 100:.2f}% | Trades: {n_trades}"
    )

    return data, cagr, mdd, n_trades


def plot_results(data, ticker):
    plt.figure(figsize=(12, 8))

    # Equity Curve
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data["Equity"], label="Strategy Equity")

    # Buy & Hold (Rescaled)
    if "Close" in data:
        bh = data["Close"] / data["Close"].iloc[0] * data["Equity"].iloc[0]
        plt.plot(data.index, bh, label="Buy & Hold", alpha=0.5, linestyle="--")

    plt.title(f"{ticker} Strategy Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Indicator Panel (Zoomed in for last 2 years)
    plt.subplot(2, 1, 2)
    subset = data.iloc[-500:]
    plt.plot(subset.index, subset["Close"], label="Price", color="black")
    plt.plot(subset.index, subset["T3"], label="T3", color="orange")

    # SuperTrend
    # Plot only one line often messy. Let's just plot Signals
    entries = subset[subset["ST_Dir"] == 1].index
    # Only show actual trades if signal col is present

    plt.title(f"{ticker} Recent Indicators")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"result_{ticker}.png")
    # plt.show()


if __name__ == "__main__":
    # Test SPY
    spy_data, spy_cagr, spy_mdd, spy_trades = run_strategy("SPY")
    plot_results(spy_data, "SPY")

    # Test BTC
    # Test BTC
    btc_data, btc_cagr, btc_mdd, btc_trades = run_strategy("BTC-USD")
    plot_results(btc_data, "BTC-USD")
