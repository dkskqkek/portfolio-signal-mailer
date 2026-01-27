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
    # Core: SPY 20% + SCHD 20% + GLD 15% = 55%
    # Tactical: QLD 45%

    W_CORE_SPY = 0.20
    W_CORE_SCHD = 0.20
    W_CORE_GLD = 0.15
    TACTICAL_WEIGHT = 0.45

    core_assets = ["SPY", "VIG", "GLD", "SCHD"]  # VIG as early proxy for SCHD
    def_pool = ["XLP", "XLU", "GLD", "FXY", "UUP", "DBC", "TLT", "IEF", "BIL", "SHY"]
    RISKY_ASSET = "QLD"
    BENCHMARK = "SPY"

    all_tickers = list(set(core_assets + def_pool + [RISKY_ASSET, BENCHMARK, "QQQ"]))

    df = fetch_data(all_tickers, start_date="2008-01-01")
    df = df.dropna(subset=["QQQ", "SPY"])

    DIVIDEND_TICKER = "VIG"  # Proxy

    # 2. NEW SIGNAL LOGIC: FAST TREND
    # Replaced 110/250 with Single SMA 120 (approx 6 months) for faster cut-loss
    # or a "Price < SMA 200 AND Returns < 0" check.
    # Let's try: SMA 120 (Standard "Bull/Bear" divider is often 200, but 120 is tactical)
    # Actually, let's use a specific "Crash Protection":
    # IF QQQ < SMA 150 -> DEFENSIVE. (Faster than 200, filters noise better than 50)

    SMA_FAST = 130  # 6 months approx (21*6 = 126)
    df["SMA_Trend"] = df["QQQ"].rolling(window=SMA_FAST).mean()

    # Core Returns
    df["Ret_Core"] = (
        (df["SPY"].pct_change() * W_CORE_SPY)
        + (df[DIVIDEND_TICKER].pct_change() * W_CORE_SCHD)
        + (df["GLD"].pct_change() * W_CORE_GLD)
    )

    # Generate Signal
    # Condition: Close > SMA 130 -> BULL (QLD)
    # else -> BEAR (Defensive)

    signals = pd.Series(index=df.index, data="NORMAL")
    signals = np.where(df["QQQ"] > df["SMA_Trend"], "NORMAL", "DANGER")
    signals = pd.Series(signals, index=df.index)

    # Add Hysteresis: Only switch if 3 days consecutive?
    # For this proof of concept, simple Day-End signal is fine.

    mask_normal = (signals == "NORMAL").shift(1).fillna(False)
    mask_danger = (signals == "DANGER").shift(1).fillna(False)

    # Returns
    ret_qld = df[RISKY_ASSET].pct_change()

    # Defensive Basket
    def_basket_ret = pd.Series(0.0, index=df.index)
    monthly_idx = df.resample("M").last().index

    print("Calculating defensive basket...")
    for i in range(len(monthly_idx) - 1):
        m_start = monthly_idx[i]
        m_end = monthly_idx[i + 1]
        try:
            loc_idx = df.index.get_indexer([m_start], method="pad")[0]
            past_prices = df[def_pool].iloc[loc_idx - 168]  # 8 month momentum remains
            curr_prices = df[def_pool].iloc[loc_idx]
            moms = (curr_prices - past_prices) / past_prices
            pos_moms = moms[moms > 0].sort_values(ascending=False)
            selected = (
                ["BIL"] if len(pos_moms) == 0 else pos_moms.head(3).index.tolist()
            )
        except:
            selected = ["BIL"]

        mask_period = (df.index > m_start) & (df.index <= m_end)
        if len(selected) > 0:
            def_basket_ret.loc[mask_period] = (
                df.loc[mask_period, selected].pct_change().mean(axis=1)
            )

    # Strategy Return
    df["Ret_Tactical"] = 0.0
    df.loc[mask_normal, "Ret_Tactical"] = ret_qld.loc[mask_normal]
    df.loc[mask_danger, "Ret_Tactical"] = def_basket_ret.loc[mask_danger]

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

    print(f"\n[Improved Logic (SMA {SMA_FAST} + SCHD)] Result:")
    print(f"CAGR: {cagr * 100:.2f}%")
    print(f"MDD : {mdd * 100:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")

    # Yearly
    df["Year"] = df.index.year
    yearly = (
        df.groupby("Year")["Ret_Strategy"].apply(lambda x: (1 + x).prod() - 1) * 100
    )
    print("\n[Yearly Returns]")
    print(yearly)


if __name__ == "__main__":
    run_backtest()
