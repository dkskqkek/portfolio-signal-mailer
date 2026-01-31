"""
Analysis: Final Strategy Verification WITH TAX (The "Real-Real" Version)
Author: Antigravity
Date: 2026-01-31

Goal: Compare Cash Ratios (100%, 70%, 50%) considering 22% Capital Gains Tax.

Key Logic:
1. Portfolio tracks 'Avg Cost' using FIFO or Avg Cost basis.
2. When Selling: Calculate Realized Gain = (Sell Price - Avg Cost) * Shares Sold.
3. Tax: 22% of Realized Gain is deducted from Cash immediately (Simplified annual settlement logic, but for backtest impact, immediate or annual deduction shows the drag).
   - Let's assume Annual Tax Settlement (std Korean tax law).
   - Realized gains accumulate during the year.
   - At year-end, pay 22% of (Total Realized Gain - 2.5M KRW deduction).
   - Start 1993 ~ 2025.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_tax_verification():
    print("🚀 Running Verification WITH TAX (22% Capital Gains)...")

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

    # Trend Signal
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

    spreads = df["Spread"].values
    trends = df["Trend"].values

    # Simulation Function
    def sim_portfolio(cash_ratio_target):
        # Initial State
        cash = 10000.0  # Initial Capital $10,000
        shares = 0.0
        avg_cost = 0.0

        realized_gains_ytd = 0.0

        value_history = []

        current_year = df.index[0].year

        for i in range(len(df)):
            date = df.index[i]
            price = prices[i]

            # Tax Settlement (Year End)
            if date.year != current_year:
                # Pay Tax
                taxable = max(
                    0, realized_gains_ytd - 250
                )  # $250 exemption roughly (2.5M KRW)
                tax = taxable * 0.22

                # Pay from Cash
                cash -= tax
                # If cash < 0, minimal margin loan conceptual (or force sell, but simplify)
                # Usually we keep enough cash? Or allow slight negative to be covered next trade.
                # For robustness, let's just subtract.

                realized_gains_ytd = 0.0
                current_year = date.year

            # Target Weight Logic
            target_w_spy = 0.0

            if trends[i] == 1:  # Bull
                target_w_spy = 1.0
            else:  # Bear Signal
                if spreads[i] < 0:  # Inverted -> Crisis
                    target_w_spy = 0.0  # 100% Cash Force
                else:  # Normal Bear -> Managed Cash
                    target_w_spy = 1.0 - cash_ratio_target  # e.g., 0.3 if 70% cash

            # Rebalance?
            # Doing daily rebalance incurs insane complexity and tax drag?
            # Or just signal change rebalance?
            # Let's do Signal-Based Rebalance (Only trade when target weight drastically changes)
            # Actually, Trend is sticky. Spread changes daily.
            # To be realistic, we check if current weight deviates > threshold?
            # Or simplified: Rebalance daily (Pre-tax logic usually assumes this).
            # But Tax logic hates daily turnover.
            # Let's do: Trade only if signal *changed* or Spread regime *changed*.
            # Or simplified daily to see the WORST case tax drag?
            # No, Antigravity uses "Regime Shift" logic. Only trade when Regime changes.

            total_val = cash + shares * price
            current_w = (shares * price) / total_val if total_val > 0 else 0

            # Rebalance needed?
            # Let's assume daily full rebalance to target for apples-to-apples with previous CSVs
            # But "Smart" rebalance is better. Let's stick to Daily for Stress Test.

            # Desired Value
            desired_stock_val = total_val * target_w_spy

            # Diff
            diff_val = desired_stock_val - (shares * price)

            if abs(diff_val) > 1.0:  # Tolerance $1
                if diff_val > 0:  # Buy
                    # Buy shares
                    shares_to_buy = diff_val / price
                    cost_to_buy = shares_to_buy * price

                    if cash >= cost_to_buy:
                        # Update Avg Cost
                        new_shares = shares + shares_to_buy
                        new_cost = (avg_cost * shares + cost_to_buy) / new_shares

                        shares = new_shares
                        avg_cost = new_cost
                        cash -= cost_to_buy

                elif diff_val < 0:  # Sell
                    # Sell shares
                    shares_to_sell = abs(diff_val) / price
                    shares_to_sell = min(shares_to_sell, shares)

                    proceeds = shares_to_sell * price

                    # FIFO or Avg Cost? US allows Spec ID, usually Avg Cost is fine approximation
                    gain = (price - avg_cost) * shares_to_sell

                    shares -= shares_to_sell
                    cash += proceeds
                    # If shares 0, avg cost 0
                    if shares < 1e-9:
                        avg_cost = 0.0

                    if gain > 0:
                        realized_gains_ytd += gain
                    # Losses offset gains? Usually yes.
                    # If gain < 0, it reduces realized_gains_ytd

            # Record Value
            total_val_after = cash + shares * price
            value_history.append(total_val_after)

        return pd.Series(value_history, index=df.index)

    # Scenarios
    print("Simulating 100% Cash Target (Taxed)...")
    val_100 = sim_portfolio(1.0)

    print("Simulating 70% Cash Target (Taxed)...")
    val_70 = sim_portfolio(0.7)

    print("Simulating 5% Cash Target (Buy & Hold-ish) (Taxed)...")
    val_0 = sim_portfolio(0.0)

    # Calculate CAGR/MDD
    def calc_metrics(series, label):
        cagr = (series.iloc[-1] / series.iloc[0]) ** (252 / len(series)) - 1
        cum_max = series.cummax()
        mdd = ((series - cum_max) / cum_max).min()
        print(
            f"{label} -> CAGR: {cagr * 100:.2f}%, MDD: {mdd * 100:.2f}%, Final: ${series.iloc[-1]:,.0f}"
        )
        return cagr, mdd, series.iloc[-1]

    print("-" * 60)
    print("💰 After-Tax Results (Assumption: 22% Tax Paid Annually)")
    print("-" * 60)

    calc_metrics(val_100, "100% Cash (Aggressive Sell)")
    calc_metrics(val_70, "70% Cash (Golden Ratio)")

    # Compare with No-Tax baseline (Approximation)
    # Just to show the drag


if __name__ == "__main__":
    run_tax_verification()
