# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np


def fetch_data():
    return yf.download(
        "QQQ", start="1999-01-01", end="2026-02-01", progress=False
    ).ffill()


def analyze():
    df = fetch_data()
    if isinstance(df.columns, pd.MultiIndex):
        col = "Adj Close" if "Adj Close" in df.columns.get_level_values(0) else "Close"
        prices = df[col]["QQQ"]
    else:
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        prices = df[col]

    pct_change = prices.pct_change()

    def get_stats(signal):
        ret = (signal.shift(1) * (pct_change * 2)) - (signal.diff().abs() * 0.001)
        ret = ret.fillna(0)
        cum = (1 + ret).cumprod()
        dd = cum / cum.cummax() - 1
        return ret, cum, dd

    # 1. Comparison: 110/250 vs 130/260
    params = [(110, 250), (130, 260)]
    print("\n[Risk Maturity Analysis]")
    for s1, s2 in params:
        sig = (
            (prices > prices.rolling(s1).mean()) & (prices > prices.rolling(s2).mean())
        ).astype(int)
        ret, cum, dd = get_stats(sig)
        mdd = dd.min()
        mdd_date = dd.idxmin()
        ann_ret = ret.mean() * 252
        print(
            f"Params {s1}/{s2}: CAGR {ann_ret:.1%}, MDD {mdd:.1%} at {mdd_date.date()}"
        )

    # 2. Adaptive vs Fixed Diagnostics
    print("\n[Adaptive Diagnostic]")
    vol = pct_change.rolling(60).std() * np.sqrt(252)
    sig_f = (
        (prices > prices.rolling(110).mean()) & (prices > prices.rolling(250).mean())
    ).astype(int)

    sig_a = []
    for i in range(len(prices)):
        v = vol.iloc[i]
        if v < 0.15:
            s1, s2 = 110, 250
        elif v < 0.25:
            s1, s2 = 100, 225
        else:
            s1, s2 = 80, 180

        # dynamic check
        if i < 250:
            sig_a.append(0)
            continue
        ma1 = prices.iloc[i - s1 : i].mean()  # approx
        ma2 = prices.iloc[i - s2 : i].mean()
        sig_a.append(1 if prices.iloc[i] > ma1 and prices.iloc[i] > ma2 else 0)

    sig_a = pd.Series(sig_a, index=prices.index)

    for name, s in [("Fixed", sig_f), ("Adaptive", sig_a)]:
        trades = s.diff().abs().sum()
        whipsaws = ((s.diff().abs() > 0) & (s.diff(20).abs() == 0)).sum()
        hold = (s == 1).sum() / (trades / 2 + 1e-6)
        print(
            f"{name:<10}: Trades {trades:3.0f}, Whipsaws {whipsaws:3.0f}, Avg Hold {hold:4.1f} days"
        )


if __name__ == "__main__":
    analyze()
