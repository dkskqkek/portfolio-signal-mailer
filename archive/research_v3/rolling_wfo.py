import yfinance as yf
import pandas as pd
import numpy as np


def fetch_data():
    print("Fetching QQQ Data (1995-2026)...")
    qqq = yf.download("QQQ", start="1995-01-01", end="2026-02-01", progress=False)

    if isinstance(qqq.columns, pd.MultiIndex):
        if "Adj Close" in qqq.columns.get_level_values(0):
            qqq = qqq["Adj Close"]["QQQ"].to_frame(name="Close")
        else:
            qqq = qqq["Close"]["QQQ"].to_frame(name="Close")
    else:
        col = "Adj Close" if "Adj Close" in qqq.columns else "Close"
        qqq = qqq[[col]].rename(columns={col: "Close"})

    return qqq.ffill()


def optimize_params(df_train):
    """Find Best Sharpe Params in Training Window"""
    best_sharpe = -999
    best_params = (110, 250)

    # Grid: Fast 80-150, Slow 180-300 (Step 20 for speed)
    fast_range = range(80, 151, 10)
    slow_range = range(180, 301, 20)

    for f in fast_range:
        for s in slow_range:
            if f >= s:
                continue

            # Vectorized Backtest
            ma_f = df_train["Close"].rolling(f).mean()
            ma_s = df_train["Close"].rolling(s).mean()

            # Simple Dual SMA Logic
            signal = (df_train["Close"] > ma_f) & (df_train["Close"] > ma_s)
            ret = signal.shift(1) * (df_train["Close"].pct_change() * 2)  # 2x Lev Proxy

            # Sharpe
            ann_ret = ret.mean() * 252
            ann_vol = ret.std() * np.sqrt(252)
            if ann_vol == 0:
                sharpe = 0
            else:
                sharpe = (ann_ret - 0.04) / ann_vol

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (f, s)

    return best_params, best_sharpe


def run_rolling_wfo():
    df = fetch_data()
    if df.empty:
        print("No data.")
        return

    print("\n[Rolling Walk-Forward Analysis]")
    print("Training Window: 5 Years | Test Window: 1 Year")
    print("-" * 60)
    print(
        f"{'Year':<6} | {'Best Params':<12} | {'IS Sharpe':<10} | {'OOS Sharpe':<10} | {'WFE':<6} | {'Trade #':<8}"
    )
    print("-" * 60)

    results = []

    start_year = 2005
    end_year = 2026

    for year in range(start_year, end_year):
        # Define Periods
        train_start = f"{year - 5}-01-01"
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        # 1. In-Sample Optimization
        train_df = df.loc[train_start:train_end]
        if len(train_df) < 500:
            continue

        best_params, is_sharpe = optimize_params(train_df)

        # 2. Out-of-Sample Test
        test_df = df.loc[test_start:test_end]
        if len(test_df) < 100:
            continue

        f, s = best_params
        ma_f = test_df["Close"].rolling(f).mean()
        ma_s = test_df["Close"].rolling(s).mean()

        # We need pre-rolled data for MA calc, so we actually need to slice wider for calculation
        # But for simplicity, let's assume we pass enough history or use the main DF with mask

        # Better: Calculate indicators on full DF, then slice
        full_ma_f = df["Close"].rolling(f).mean()
        full_ma_s = df["Close"].rolling(s).mean()

        test_condition = (df.index >= test_start) & (df.index <= test_end)
        sub_price = df.loc[test_condition, "Close"]
        sub_ma_f = full_ma_f.loc[test_condition]
        sub_ma_s = full_ma_s.loc[test_condition]

        signal = (sub_price > sub_ma_f) & (sub_price > sub_ma_s)

        # Count Trades (Change from 0 to 1 or 1 to 0)
        # Signal is boolean. diff().abs().sum()
        trades = signal.astype(int).diff().abs().sum()

        ret = signal.shift(1) * (sub_price.pct_change() * 2)
        ann_ret = ret.mean() * 252
        ann_vol = ret.std() * np.sqrt(252)
        if ann_vol == 0:
            oos_sharpe = 0
        else:
            oos_sharpe = (ann_ret - 0.04) / ann_vol

        wfe = oos_sharpe / is_sharpe if is_sharpe > 0 else 0

        print(
            f"{year:<6} | {str(best_params):<12} | {is_sharpe:<10.2f} | {oos_sharpe:<10.2f} | {wfe:<6.2f} | {trades:<8}"
        )

        results.append(
            {"Year": year, "Params": best_params, "WFE": wfe, "Trades": trades}
        )

    print("-" * 60)
    avg_wfe = np.mean([r["WFE"] for r in results])
    print(f"Average WFE: {avg_wfe:.2f}")

    # Drift Analysis
    print("\n[Parameter Drift Check]")
    params = [r["Params"] for r in results]
    fasts = [p[0] for p in params]
    slows = [p[1] for p in params]

    print(f"Fast SMA Range: {min(fasts)} ~ {max(fasts)} (Avg: {np.mean(fasts):.1f})")
    print(f"Slow SMA Range: {min(slows)} ~ {max(slows)} (Avg: {np.mean(slows):.1f})")


if __name__ == "__main__":
    run_rolling_wfo()
