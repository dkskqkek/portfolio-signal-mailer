import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def fetch_data(tickers, start_date="2008-01-01"):
    print(f"Fetching data for {len(tickers)} tickers since {start_date}...")
    try:
        data = yf.download(tickers, start=start_date, progress=False, group_by="ticker")
    except Exception as e:
        print(f"Error checking download: {e}")
        return pd.DataFrame()

    df = pd.DataFrame()
    for t in tickers:
        if t in data.columns.levels[0]:
            if "Adj Close" in data[t].columns:
                df[t] = data[t]["Adj Close"]
            else:
                df[t] = data[t]["Close"]

    df = df.ffill()
    return df


def run_backtest():
    # 1. Configuration
    CORE_WEIGHT = 0.55
    TACTICAL_WEIGHT = 0.45

    # Portfolio Weights within Core (Optimized)
    # SPY 20%, SCHD 20% (Replacing KOSPI), GLD 15%
    W_CORE_SPY = 0.20
    W_CORE_SCHD = 0.20
    W_CORE_GLD = 0.15

    # Assets
    # Using SCHD. Note: SCHD inception is late 2011.
    # For backtest prior to 2011, we need a proxy. VIG (Vanguard Dividend Appreciation) is a good proxy since 2006.
    # Let's use VIG as proxy for SCHD before 2011 to allow 2008 backtest.
    core_assets = ["SPY", "VIG", "GLD", "SCHD"]

    def_pool = ["XLP", "XLU", "GLD", "FXY", "UUP", "DBC", "TLT", "IEF", "BIL", "SHY"]

    RISKY_ASSET = "QLD"
    BENCHMARK = "SPY"

    all_tickers = list(set(core_assets + def_pool + [RISKY_ASSET, BENCHMARK, "QQQ"]))

    # 2. Get Data
    df = fetch_data(all_tickers, start_date="2008-01-01")
    df = df.dropna(subset=["QQQ", "SPY"])

    # Patch SCHD with VIG for early data
    # Calculate factor between SCHD and VIG at transition to smooth splicing?
    # Simply using VIG directly as the "Dividend Growth Slot" for the whole period is cleaner for broad verification.
    # Let's use VIG for the entire period for simplicity of data continuity in this check.
    # SCHD is slightly more aggressive/yield-focused than VIG, but VIG is the standard proxy.
    DIVIDEND_TICKER = "VIG"

    # 3. Indicators
    df["SMA110"] = df["QQQ"].rolling(window=110).mean()
    df["SMA250"] = df["QQQ"].rolling(window=250).mean()

    # Defensive Momentum
    mom_df = df[def_pool].pct_change(168)

    # 4. Simulation
    # Core Component Daily Return
    # 20% SPY, 20% DIVIDEND, 15% GLD
    df["Ret_Core"] = (
        (df["SPY"].pct_change() * W_CORE_SPY)
        + (df[DIVIDEND_TICKER].pct_change() * W_CORE_SCHD)
        + (df["GLD"].pct_change() * W_CORE_GLD)
    )

    # Tactical Signal (Hysteresis)
    signals = pd.Series(index=df.index, data="NORMAL")
    curr_status = "NORMAL"

    # Loop for signal generation
    for i in range(250, len(df)):
        price = df["QQQ"].iloc[i]
        sma110 = df["SMA110"].iloc[i]
        sma250 = df["SMA250"].iloc[i]

        if price > sma110 and price > sma250:
            curr_status = "NORMAL"
        elif price < sma110 and price < sma250:
            curr_status = "DANGER"
        signals.iloc[i] = curr_status

    mask_normal = (signals == "NORMAL").shift(1).fillna(False)
    mask_danger = (signals == "DANGER").shift(1).fillna(False)

    # Tactical Returns
    ret_qld = df[RISKY_ASSET].pct_change()

    # Defensive Returns (Monthly Rebal Top 3)
    def_basket_ret = pd.Series(0.0, index=df.index)
    monthly_idx = df.resample("M").last().index

    print("Calculating defensive basket (Optimized)...")
    for i in range(len(monthly_idx) - 1):
        m_start = monthly_idx[i]
        m_end = monthly_idx[i + 1]

        try:
            loc_idx = df.index.get_indexer([m_start], method="pad")[0]
            past_prices = df[def_pool].iloc[loc_idx - 168]
            curr_prices = df[def_pool].iloc[loc_idx]
            moms = (curr_prices - past_prices) / past_prices

            pos_moms = moms[moms > 0].sort_values(ascending=False)
            if len(pos_moms) == 0:
                selected = ["BIL"]
            else:
                selected = pos_moms.head(3).index.tolist()
        except:
            selected = ["BIL"]

        mask_period = (df.index > m_start) & (df.index <= m_end)
        if len(selected) > 0:
            period_returns = df.loc[mask_period, selected].pct_change().mean(axis=1)
            def_basket_ret.loc[mask_period] = period_returns

    # Combine
    df["Ret_Tactical"] = 0.0
    df.loc[mask_normal, "Ret_Tactical"] = ret_qld.loc[mask_normal]
    df.loc[mask_danger, "Ret_Tactical"] = def_basket_ret.loc[mask_danger]

    # Strategy Total
    # Note: Weights sum to 0.55 + 0.45 = 1.0
    # Ret_Core calculated above is already weighted sum (e.g. 0.2 * ret).
    # BUT, to mix with Tactical (0.45 * ret), we need to just sum them IF Ret_Core is weighted contribution.
    # Wait, Ret_Core definition: (SPY% * 0.2 + VIG% * 0.2 + GLD% * 0.15)
    # Total Portfolio Ret = Ret_Core + (Ret_Tactical * 0.45)
    # Logic check:
    # Portfolio = 0.2 SPY + 0.2 VIG + 0.15 GLD + 0.45 Tactical
    # Sum of weights = 0.2+0.2+0.15+0.45 = 1.0. Correct.

    df["Ret_Strategy"] = df["Ret_Core"] + (df["Ret_Tactical"] * TACTICAL_WEIGHT)
    df = df.dropna(subset=["Ret_Strategy"])

    # Stats
    df["Strategy_Idx"] = (1 + df["Ret_Strategy"]).cumprod() * 100000

    total_days = len(df)
    years = total_days / 252
    cagr = (df["Strategy_Idx"].iloc[-1] / df["Strategy_Idx"].iloc[0]) ** (1 / years) - 1

    roll_max = df["Strategy_Idx"].cummax()
    mdd = ((df["Strategy_Idx"] - roll_max) / roll_max).min()

    rf = 0.03
    sharpe = (df["Ret_Strategy"].mean() * 252 - rf) / (
        df["Ret_Strategy"].std() * np.sqrt(252)
    )

    print(f"\n[Optimized Backtest Results (KOSPI -> VIG/SCHD)]")
    print(f"CAGR: {cagr * 100:.2f}%")
    print(f"MDD : {mdd * 100:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")

    # 2008 & 2022 Performance check
    df["Year"] = df.index.year
    yearly = (
        df.groupby("Year")["Ret_Strategy"].apply(lambda x: (1 + x).prod() - 1) * 100
    )
    print("\n[Yearly Returns]")
    print(yearly)


if __name__ == "__main__":
    run_backtest()
