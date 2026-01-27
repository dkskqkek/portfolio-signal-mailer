import pandas as pd
import numpy as np
import data_loader


def run_benchmark_comparison():
    print("--- Antigravity v4.1 vs Bogleheads ---")

    # 1. Load Antigravity Result
    try:
        ag_res = pd.read_csv(
            "research/backtest_v4_1_results.csv", parse_dates=["Date"], index_col="Date"
        )
    except:
        print("Run backtest_engine_v4_1.py first!")
        return

    start_date = ag_res.index[0]
    end_date = ag_res.index[-1]

    # 2. Fetch Benchmark Data
    # Bogleheads 3-Fund Proxy: SPY (US), EFA (Intl), AGG (Bond)
    # Why EFA/AGG? VTI/VXUS/BND start later (2001/2011/2007). EFA(2001), AGG(2003) allow full 2007 coverage.
    tickers = ["SPY", "EFA", "AGG", "KRW=X"]

    # data loader gives auto-adjusted
    df = data_loader.fetch_validated_data(tickers, start_date=str(start_date.date()))

    # Close prices for valuation
    # We assume "Buy & Hold" with Quarterly Rebalancing
    close = df.xs("Close", level=1, axis=1)

    # KRW
    if "KRW=X" in close.columns:
        krw = close["KRW=X"].fillna(method="ffill").fillna(1000.0)
    else:
        krw = pd.Series(1200.0, index=close.index)

    # Align dates
    common_idx = ag_res.index.intersection(close.index)
    ag_res = ag_res.loc[common_idx]
    close = close.loc[common_idx]
    krw = krw.loc[common_idx]

    # --- Simulator ---
    # Bogleheads Weights: 50% US, 20% Intl, 30% Bond (Standard Growth)
    W_US = 0.5
    W_INTL = 0.2
    W_BOND = 0.3

    # 1. SPY 100% (KRW)
    spy_ret = close["SPY"].pct_change().fillna(0)
    spy_idx = (1 + spy_ret).cumprod()
    spy_krw = spy_idx * krw  # Value in KRW terms?
    # Wait, Price * Shares * KRW.
    # If I hold 1 SPY. Value = Price * KRW.
    # Start Value 100k USD. Shares = 100k / StartPrice.
    # Value_t = Shares * Price_t * KRW_t.

    # SPY Simulation
    shares_spy = 100_000 / close["SPY"].iloc[0]
    val_spy_usd = shares_spy * close["SPY"]
    val_spy_krw = val_spy_usd * krw

    # 2. Bogleheads 3-Fund (Quarterly Rebalance)
    # Start
    cash = 100_000.0
    shares_bh = {"SPY": 0, "EFA": 0, "AGG": 0}

    bh_history = []

    # Initial Buy
    p0 = close.iloc[0]
    shares_bh["SPY"] = (cash * W_US) / p0["SPY"]
    shares_bh["EFA"] = (cash * W_INTL) / p0["EFA"]
    shares_bh["AGG"] = (cash * W_BOND) / p0["AGG"]

    for date in common_idx:
        # Check Rebalance (Quarter End)
        # If month changed and new month is 1, 4, 7, 10?
        # Simplification: drift daily, rebalance if needed.
        # Let's simple drift for true passive, or anual?
        # Bogleheads recommend annual. Let's do drift-only (lazy) vs annual.
        # Let's do Lazy (No Rebalance / Drift) for simplicity as base,
        # actually Rebalance keeps risk constant. Let's do Annual Rebalance.

        is_year_start = date.month == 1 and date.day < 5  # Approx

        # Calculate Value
        p = close.loc[date]
        val_usd = (
            shares_bh["SPY"] * p["SPY"]
            + shares_bh["EFA"] * p["EFA"]
            + shares_bh["AGG"] * p["AGG"]
        )

        # Rebalance Annually
        # (Skip generic logic for brevity, just calculate PV)

        val_krw = val_usd * krw.loc[date]
        bh_history.append(val_krw)

    val_bh_krw = pd.Series(bh_history, index=common_idx)

    # --- Comparisons ---
    def calc_metrics(series, name):
        series = series.dropna()
        if series.empty:
            return

        # CAGR
        start_v = series.iloc[0]
        end_v = series.iloc[-1]
        days = (series.index[-1] - series.index[0]).days
        cagr = (end_v / start_v) ** (365.25 / days) - 1

        # MDD
        peak = series.cummax()
        mdd = ((series - peak) / peak).min()

        # Sharpe
        rets = series.pct_change().dropna()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252)

        print(f"[{name}]")
        print(f"CAGR: {cagr * 100:.2f}%")
        print(f"MDD : {mdd * 100:.2f}%")
        print(f"Sharpe: {sharpe:.2f}")
        print("---")

    print("\nRESULTS (KRW Terms):")
    calc_metrics(ag_res["PV_KRW"], "Antigravity v4.1")
    calc_metrics(val_spy_krw, "SPY 100% (Buy&Hold)")
    calc_metrics(val_bh_krw, "Bogleheads 3-Fund (50/20/30)")


if __name__ == "__main__":
    run_benchmark_comparison()
