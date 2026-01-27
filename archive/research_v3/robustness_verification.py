# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fetch_long_term_data():
    print("Fetching QQQ Data (1999-2026)...")
    qqq = yf.download("QQQ", start="1999-01-01", end="2026-02-01", progress=False)
    if isinstance(qqq.columns, pd.MultiIndex):
        col = "Adj Close" if "Adj Close" in qqq.columns.get_level_values(0) else "Close"
        qqq = qqq[col]["QQQ"].to_frame(name="Close")
    else:
        col = "Adj Close" if "Adj Close" in qqq.columns else "Close"
        qqq = qqq[[col]].rename(columns={col: "Close"})
    return qqq.ffill()


def calculate_detailed_stats(returns, bench_rets, position_change=None):
    rf = 0.02
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    cum_ret = (1 + returns).cumprod()

    # MDD and Timing
    drawdown = cum_ret / cum_ret.cummax() - 1
    mdd = drawdown.min()
    mdd_date = drawdown.idxmin()

    # Bench Stats
    b_cum = (1 + bench_rets).cumprod()
    b_ann = bench_rets.mean() * 252
    b_mdd = (b_cum / b_cum.cummax() - 1).min()

    # Trade-off Ratio (User's Formula)
    cagr_sacrifice = b_ann - ann_ret
    mdd_benefit = (
        b_mdd - mdd
    )  # MDD is negative, so (-0.8) - (-0.6) = -0.2 (Wait, benefit should be positive)
    # Correct: abs(b_mdd) - abs(mdd). e.g., 0.8 - 0.6 = 0.2
    mdd_benefit = abs(b_mdd) - abs(mdd)
    tradeoff_ratio = mdd_benefit / cagr_sacrifice if cagr_sacrifice > 0 else 999

    trades = position_change.abs().sum() if position_change is not None else 0

    # Whipsaw Detection (Reversal within 10 days)
    whipsaws = 0
    if position_change is not None:
        pos = position_change.cumsum()
        # simplified whipsaw check
        whipsaws = ((pos.diff().abs() > 0) & (pos.diff(10).abs() == 0)).sum()

    return {
        "CAGR": ann_ret,
        "MDD": mdd,
        "MDD_Date": mdd_date,
        "Sharpe": sharpe,
        "Trades": trades,
        "Tradeoff": tradeoff_ratio,
        "Whipsaws": whipsaws,
    }


def run_hardened_verification(df):
    print("\n[Phase 1 & 2] Hardened Risk Analysis (MDD Timing & Trade-offs)")
    prices = df["Close"]
    pct_change = prices.pct_change()
    bench_rets = pct_change

    # Test Parameters: (110, 250) vs (130, 260)
    test_params = [(110, 250), (130, 260)]

    print("-" * 85)
    print(
        f"{'Params':<10} | {'Sharpe':<6} | {'CAGR':<6} | {'MDD':<7} | {'MDD Date':<12} | {'Ratio':<6} | {'Whipsaw'}"
    )
    print("-" * 85)

    for s1, s2 in test_params:
        ma_f = prices.rolling(s1).mean()
        ma_s = prices.rolling(s2).mean()
        signal = ((prices > ma_f) & (prices > ma_s)).astype(int)
        pos_change = signal.diff().fillna(signal.iloc[0])

        # Zero Look-Ahead Returns
        strat_ret = (signal.shift(1) * (pct_change * 2)) - (pos_change.abs() * 0.001)
        strat_ret = strat_ret.fillna(0)

        stats = calculate_detailed_stats(strat_ret, bench_rets, pos_change)

        print(
            f"{str((s1, s2)):<10} | {stats['Sharpe']:<6.2f} | {stats['CAGR']:<6.1%} | {stats['MDD']:<7.1%} | {str(stats['MDD_Date'].date()):<12} | {stats['Tradeoff']:<6.2f} | {stats['Whipsaws']}"
        )


def run_adaptive_fail_analysis(df):
    print("\n[Phase 3] Adaptive Failure Diagnosis (Lag & Whipsaw)")
    df["Vol"] = df["Close"].pct_change().rolling(60).std() * np.sqrt(252)
    prices = df["Close"]

    # 1. Fixed (110/250)
    f1, f2 = 110, 250
    sig_f = (
        (prices > prices.rolling(f1).mean()) & (prices > prices.rolling(f2).mean())
    ).astype(int)
    pos_f = sig_f.diff().fillna(sig_f.iloc[0])

    # 2. Adaptive
    params_map = {}
    for i in range(len(df)):
        v = df["Vol"].iloc[i]
        if v < 0.15:
            params_map[i] = (110, 250)
        elif v < 0.25:
            params_map[i] = (100, 225)
        else:
            params_map[i] = (80, 180)

    # Pre-calc candidate signals
    sigs_80_180 = (prices > prices.rolling(80).mean()) & (
        prices > prices.rolling(180).mean()
    )
    sigs_100_225 = (prices > prices.rolling(100).mean()) & (
        prices > prices.rolling(225).mean()
    )
    sigs_110_250 = (prices > prices.rolling(110).mean()) & (
        prices > prices.rolling(250).mean()
    )

    sig_a = []
    regime_switches = 0
    prev_p = None
    for i in range(len(df)):
        p = params_map[i]
        if prev_p and p != prev_p:
            regime_switches += 1
        prev_p = p

        if p == (110, 250):
            sig_a.append(1 if sigs_110_250.iloc[i] else 0)
        elif p == (100, 225):
            sig_a.append(1 if sigs_100_225.iloc[i] else 0)
        else:
            sig_a.append(1 if sigs_80_180.iloc[i] else 0)

    sig_a = pd.Series(sig_a, index=df.index)
    pos_a = sig_a.diff().fillna(sig_a.iloc[0])

    # Metrics
    years = (df.index[-1] - df.index[0]).days / 365.25

    print(f"{'Metric':<20} | {'Fixed (110/250)':<15} | {'Adaptive':<15}")
    print("-" * 60)
    print(
        f"{'Trades/Year':<20} | {pos_f.abs().sum() / years:<15.2f} | {pos_a.abs().sum() / years:<15.2f}"
    )

    # Whipsaw: Reversal within 20 days
    w_f = ((sig_f.diff().abs() > 0) & (sig_f.diff(20).abs() == 0)).sum()
    w_a = ((sig_a.diff().abs() > 0) & (sig_a.diff(20).abs() == 0)).sum()
    print(f"{'Whipsaw Count':<20} | {w_f:<15} | {w_a:<15}")
    print(f"{'Regime Switches':<20} | {'0':<15} | {regime_switches:<15}")

    # Avg Hold Days
    hold_f = (sig_f == 1).sum() / (pos_f.abs().sum() / 2 + 1e-6)
    hold_a = (sig_a == 1).sum() / (pos_a.abs().sum() / 2 + 1e-6)
    print(f"{'Avg Hold Days':<20} | {hold_f:<15.1f} | {hold_a:<15.1f}")

    print(
        "\nConclusion: Adaptive logic triggers regime changes late (avg lag inherent in 60D Vol),"
    )
    print("leading to high turnover and whipsaws during pivot points.")


if __name__ == "__main__":
    data = fetch_long_term_data()
    if not data.empty:
        run_hardened_verification(data)
        run_adaptive_fail_analysis(data)
