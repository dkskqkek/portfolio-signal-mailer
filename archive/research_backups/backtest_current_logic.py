import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")


def fetch_data(tickers, start_date="2010-01-01"):
    print(f"Fetching data for {len(tickers)} tickers since {start_date}...")
    data = yf.download(tickers, start=start_date, progress=False, group_by="ticker")
    df = pd.DataFrame()
    for t in tickers:
        if t in data.columns.levels[0]:
            # Prefer Adj Close, fallback to Close
            if "Adj Close" in data[t].columns:
                df[t] = data[t]["Adj Close"]
            else:
                df[t] = data[t]["Close"]

    # Forward fill to handle different holidays/trading hours
    df = df.ffill()
    return df


def run_backtest():
    # 1. Configuration
    # Portfolio Weights within Core
    # SPY 20%, KOSPI 20%, GLD 15% -> Normalized: 20/55, 20/55, 15/55

    # Assets
    # Using EWY as KOSPI proxy (US Listed)
    core_assets = ["SPY", "EWY", "GLD"]

    # Tactical Universe
    def_pool = ["XLP", "XLU", "GLD", "FXY", "UUP", "DBC", "TLT", "IEF", "BIL", "SHY"]
    # Target risky asset
    RISKY_ASSET = "QLD"  # 2x Nasdaq

    # 2. Get Data (Start earliest meaningful for QLD which is mid-2006)
    all_tickers = list(set(core_assets + def_pool + [RISKY_ASSET, "SPY", "QQQ"]))
    df = fetch_data(all_tickers, start_date="2006-06-25")
    df = df.dropna(subset=["QQQ", "SPY"])

    # 3. Calculate Indicators
    # Dual SMA
    df["SMA110"] = df["QQQ"].rolling(window=110).mean()
    df["SMA250"] = df["QQQ"].rolling(window=250).mean()

    # Defensive Momentum (8 months = 168 trading days approx)
    lookback_mom = 168
    mom_df = df[def_pool].pct_change(lookback_mom)

    # 4. Simulation Loop

    # Monthly rebalance flag
    df["Month"] = df.index.month

    print("\nRunning simulation...")

    # Initialize Tactical Signal as DANGER (Defensive) by default to avoid accidental leverage during warmup
    signals = pd.Series(index=df.index, data="DANGER")

    # Hysteresis state
    curr_status = "DANGER"  # Default to Danger until proved Normal

    # Generate Signal Loop (Dual SMA)
    for i in range(250, len(df)):
        price = df["QQQ"].iloc[i]
        sma110 = df["SMA110"].iloc[i]
        sma250 = df["SMA250"].iloc[i]

        if np.isnan(sma110) or np.isnan(sma250):
            curr_status = "DANGER"
        elif price > sma110 and price > sma250:
            curr_status = "NORMAL"
        elif price < sma110 and price < sma250:
            curr_status = "DANGER"
        # else STAY (Hysteresis)

        signals.iloc[i] = curr_status

    # --- Vectorized Approach for Speed & Reliability ---
    # 1. Core Component
    df["Ret_Core"] = (
        (df["SPY"].pct_change() * 0.20)
        + (df["EWY"].pct_change() * 0.20)
        + (df["GLD"].pct_change() * 0.15)
    )

    # 2. Tactical Component
    # Create mask for signals (Shift 1 day for implementation lag)
    mask_normal = (signals == "NORMAL").shift(1).fillna(False)
    mask_danger = (signals == "DANGER").shift(1).fillna(True)  # Default danger

    # Return of QLD
    ret_qld = df[RISKY_ASSET].pct_change()

    # Return of Defensive
    # Resample to monthly for selection, then reindex to daily
    def_basket_ret = pd.Series(0.0, index=df.index)
    monthly_idx = df.resample("M").last().index

    print("Calculating defensive basket returns...")
    for i in range(len(monthly_idx) - 1):
        m_start = monthly_idx[i]
        m_end = monthly_idx[i + 1]

        # Selection at m_start
        try:
            loc_idx = df.index.get_indexer([m_start], method="pad")[0]
            past_prices = df[def_pool].iloc[loc_idx - 168]
            curr_prices = df[def_pool].iloc[loc_idx]
            moms = (curr_prices - past_prices) / past_prices

            # Select
            pos_moms = moms[moms > 0].sort_values(ascending=False)
            if len(pos_moms) == 0:
                selected = ["BIL"]
            else:
                selected = pos_moms.head(3).index.tolist()
        except:
            selected = ["BIL"]

        # Assign return for next month
        mask_period = (df.index > m_start) & (df.index <= m_end)

        if len(selected) > 0:
            # Check availability of selected tickers in period
            period_returns = df.loc[mask_period, selected].pct_change().mean(axis=1)
            def_basket_ret.loc[mask_period] = period_returns

    # Combine Tactical
    df["Ret_Tactical"] = 0.0
    df.loc[mask_normal, "Ret_Tactical"] = ret_qld.loc[mask_normal]
    df.loc[mask_danger, "Ret_Tactical"] = def_basket_ret.loc[mask_danger]

    # Total Strategy Return
    df["Ret_Strategy"] = df["Ret_Core"] + (df["Ret_Tactical"] * 0.45)

    # Clean up early NaN
    df = df.dropna(subset=["Ret_Strategy"])

    # Cumulative Index
    df["Strategy_Idx"] = (1 + df["Ret_Strategy"]).cumprod() * 100000
    df["Benchmark_Idx"] = (1 + df["SPY"].pct_change()).cumprod() * 100000

    # Stats
    total_days = len(df)
    years = total_days / 252.0

    cagr = (df["Strategy_Idx"].iloc[-1] / df["Strategy_Idx"].iloc[0]) ** (1 / years) - 1

    # MDD
    roll_max = df["Strategy_Idx"].cummax()
    drawdown = (df["Strategy_Idx"] - roll_max) / roll_max
    mdd = drawdown.min()

    # Sharpe (Risk Free = 0.03)
    rf = 0.03
    excess_ret = df["Ret_Strategy"].mean() * 252 - rf
    vol = df["Ret_Strategy"].std() * np.sqrt(252)
    sharpe = excess_ret / vol

    # 2022 Return
    df["Year"] = df.index.year
    yearly = df.groupby("Year")["Ret_Strategy"].apply(lambda x: (1 + x).prod() - 1)
    ret_2022 = yearly.get(2022, 0.0)
    ret_2008 = yearly.get(2008, 0.0)

    print(f"\n[Backtest Results] {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"CAGR: {cagr * 100:.2f}%")
    print(f"MDD : {mdd * 100:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"2008 Return: {ret_2008 * 100:.2f}%")
    print(f"2022 Return: {ret_2022 * 100:.2f}%")
    print(f"Final Value: {df['Strategy_Idx'].iloc[-1]:,.0f} KRW (Start 100k)")

    print("\n[Yearly Returns]")
    print(yearly * 100)

    # Save to CSV
    if not os.path.exists("research"):
        os.makedirs("research")

    csv_path = "research/backtest_results_v3_1.csv"
    output_cols = [
        "QQQ",
        "SPY",
        "SMA110",
        "SMA250",
        "Ret_Strategy",
        "Strategy_Idx",
        "Benchmark_Idx",
    ]
    df[output_cols].to_csv(csv_path)
    print(f"\n[Saved] Detailed daily data saved to: {csv_path}")


if __name__ == "__main__":
    run_backtest()
