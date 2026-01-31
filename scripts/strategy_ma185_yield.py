"""
Strategy: VTI MA185 + Yield Curve Inversion (10Y - 3M)
Author: Antigravity
Date: 2026-01-31

Logic:
1. VTI > MA185: 100% VTI (Bull Market)
2. VTI < MA185:
   A. Yield Curve Inverted (Recession Warning): 100% Cash (Defensive)
   B. Yield Curve Normal: 50% VTI + 50% Cash (Uncertainty)

Data:
- VTI: Market
- ^TNX: 10 Year Treasury Yield
- ^IRX: 13 Week Treasury Yield (Proxy for short term)
- BIL: Cash proxy (Bio-weekly T-Bills) or just 0% return handling?
  -> Let's use BIL for Cash return logic or assumed risk-free rate.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_strategy():
    print("🚀 Running MA185 + Yield Curve Strategy Backtest...")

    # 1. Fetch Data
    tickers = ["VTI", "^TNX", "^IRX", "BIL"]
    start_date = "2000-01-01"
    end_date = "2025-12-31"

    print(f"📥 Downloading data ({start_date} ~ {end_date})...")
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]

    # 2. Calculate Indicators
    # Interpolate missing values (yields sometimes have gaps)
    data = data.ffill()

    # MA185 for VTI
    data["MA185"] = data["VTI"].rolling(window=185).mean()

    # Yield Spread (10Y - 3M)
    # Note: Yahoo tickers are usually index values (e.g. 4.50 for 4.5%)
    # So direct subtraction is fine.
    data["Spread"] = data["^TNX"] - data["^IRX"]

    # 3. Simulate Strategy
    # Initial Capital
    initial_capital = 100000
    cash = initial_capital
    shares = 0
    total_value = []

    # BIL is used for Cash return. Calculate daily return of BIL.
    # If BIL is not available (early 2000s), assume 0% or Use IRX/360.
    # BIL inception is around 2007. Before that, cash is cash (0% or risk free rate).
    # Let's simplify: Cash earns 0% for simplicity, or use risk-free approximation.
    # Better: Use 'Cash' as 0 interest for now to be conservative, or use shift logic.

    # To be precise, let's calculate daily percentage changes.
    data["VTI_Pct"] = data["VTI"].pct_change()
    data["BIL_Pct"] = data["BIL"].pct_change().fillna(0)

    # Portfolio Value Series
    # We will simulate day-by-day to accurately track state

    # Logic Signal (1=100% Stock, 0.5=50% Stock, 0=100% Cash)
    data["Signal"] = 0.0

    # Vectorized Signal Calculation
    # Condition 1: VTI > MA185 -> 1.0
    cond_bull = data["VTI"] > data["MA185"]

    # Condition 2: VTI < MA185
    cond_bear = ~cond_bull

    # Sub-condition: Yield Inverted (Spread < 0)
    cond_inverted = data["Spread"] < 0

    # Apply Logic
    # Default to 0.5 (Bear but Normal Yield)
    data.loc[cond_bear, "Signal"] = 0.5

    # If Bear AND Inverted -> 0.0 (Cash)
    data.loc[cond_bear & cond_inverted, "Signal"] = 0.0

    # If Bull -> 1.0
    data.loc[cond_bull, "Signal"] = 1.0

    # Shift signal by 1 day (today's close determines tomorrow's allocation)
    data["Position_VTI"] = data["Signal"].shift(1)
    data["Position_Cash"] = 1 - data["Position_VTI"]

    # Handle NaNs
    data.dropna(inplace=True)

    # Calculate Strategy Returns
    # Strategy Return = (VTI_Return * VTI_Weight) + (Cash_Return * Cash_Weight)
    # Using BIL for Cash Return when available, else 0

    def get_cash_return(row):
        # If BIL is valid (nonzero price), use BIL return
        # But BIL data might be NaN before 2007.
        # We handled fillna(0) for pct_change but check the index.
        return row["BIL_Pct"] if row.name.year >= 2008 else 0.0

    # Vectorized return calculation
    # Only if BIL existed? Simple approximation: Just use 0 for cash pre-2007
    cash_returns = data[
        "BIL_Pct"
    ]  # Assuming filled with 0 where BIL wasn't around or handled above.
    # Actually BIL IPO was 2007. yfinance download will have NaNs for BIL prices before that.
    # ffill might have propagated nothing.
    # Let's verify BIL column.

    data["Strategy_Ret"] = (data["VTI_Pct"] * data["Position_VTI"]) + (
        cash_returns * data["Position_Cash"]
    )

    # Cumulative Return
    data["Strategy_Cum"] = (1 + data["Strategy_Ret"]).cumprod() * initial_capital
    data["BuyHold_Cum"] = (1 + data["VTI_Pct"]).cumprod() * initial_capital

    # 4. Metrics
    total_days = len(data)
    years = total_days / 252

    final_val = data["Strategy_Cum"].iloc[-1]
    cagr = (final_val / initial_capital) ** (1 / years) - 1

    # MDD
    cummax = data["Strategy_Cum"].cummax()
    drawdown = (data["Strategy_Cum"] - cummax) / cummax
    mdd = drawdown.min()

    # Sharpe
    sharpe = (data["Strategy_Ret"].mean() / data["Strategy_Ret"].std()) * np.sqrt(252)

    print("-" * 50)
    print(f"Strategy Results ({data.index[0].date()} ~ {data.index[-1].date()})")
    print("-" * 50)
    print(f"Final Value: ${final_val:,.2f}")
    print(f"CAGR: {cagr * 100:.2f}%")
    print(f"MDD: {mdd * 100:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")

    # Compare with Buy & Hold
    bh_val = data["BuyHold_Cum"].iloc[-1]
    bh_cagr = (bh_val / initial_capital) ** (1 / years) - 1
    bh_mdd = (
        (data["BuyHold_Cum"] - data["BuyHold_Cum"].cummax())
        / data["BuyHold_Cum"].cummax()
    ).min()

    print("-" * 50)
    print(f"Benchmark (VTI Buy & Hold)")
    print(f"Final Value: ${bh_val:,.2f}")
    print(f"CAGR: {bh_cagr * 100:.2f}%")
    print(f"MDD: {bh_mdd * 100:.2f}%")
    print("-" * 50)

    # Save CSV
    data.to_csv("d:/gg/strategy_ma185_yield_results.csv")
    print("Saved details to d:/gg/strategy_ma185_yield_results.csv")


if __name__ == "__main__":
    run_strategy()
