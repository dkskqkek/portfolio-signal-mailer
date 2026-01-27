import pandas as pd
import numpy as np
import data_loader


def run_backtest_v4():
    print("--- Antigravity v4.0: The Truth Engine (Robust) ---")

    # 1. Validation & Setup
    START_DATE = "2006-06-01"

    # Assets
    tickers = ["SPY", "QLD", "GLD", "EWY", "QQQ", "KRW=X"]
    def_pool = ["XLP", "XLU", "GLD", "FXY", "UUP", "DBC", "TLT", "IEF", "BIL", "SHY"]
    tickers += def_pool
    tickers = list(set(tickers))

    # Fetch Data
    try:
        df = data_loader.fetch_validated_data(tickers, start_date=START_DATE)
    except Exception as e:
        print(f"Data Fetch Failed: {e}")
        return

    # Extract prices
    close_prices = df.xs("Close", level=1, axis=1)
    open_prices = df.xs("Open", level=1, axis=1)

    # Extract KRW
    if "KRW=X" in close_prices.columns:
        krw = (
            close_prices["KRW=X"].fillna(method="ffill").fillna(1000.0)
        )  # Fallback 1000
    else:
        print("[Warn] KRW=X missing. Using 1200.")
        krw = pd.Series(1200.0, index=close_prices.index)

    # Indicators (QQQ based)
    sma110 = close_prices["QQQ"].rolling(110).mean()
    sma250 = close_prices["QQQ"].rolling(250).mean()

    # Def Momentum
    monthly_close = close_prices.resample("M").last()
    mom_monthly = monthly_close.pct_change(8)  # 8-month momentum

    # Simulation Config
    # Start sufficiently late to have SMA250
    trade_dates = close_prices.index
    start_idx = 260

    # State
    cash = 100_000.0
    shares = {t: 0.0 for t in tickers}

    # Cost
    COST_RATE = 0.0015

    # Weights
    W_CORE_SPY = 0.20
    W_CORE_KOSPI = 0.20
    W_CORE_GLD = 0.15
    W_TACTICAL = 0.45

    history = []

    print(f"Starting simulation loop from {trade_dates[start_idx]}...")
    prev_signal = "DANGER"

    for i in range(start_idx, len(trade_dates) - 1):
        today = trade_dates[i]
        next_day = trade_dates[i + 1]  # Execution Day

        # 1. Signal (Close T)
        q_price = close_prices.loc[today, "QQQ"]
        s1 = sma110.loc[today]
        s2 = sma250.loc[today]

        if pd.isna(s1) or pd.isna(s2):
            signal = "DANGER"
        elif q_price > s1 and q_price > s2:
            signal = "NORMAL"
        else:
            signal = "DANGER"

        # 2. Target Allocation
        # Def Selection logic
        last_m_idx = mom_monthly.index.asof(today - pd.Timedelta(days=1))
        selected_def = ["BIL"]

        if pd.notna(last_m_idx):
            try:
                row = mom_monthly.loc[last_m_idx]
                pos = row[def_pool].dropna()
                pos = pos[pos > 0].sort_values(ascending=False)
                if len(pos) > 0:
                    selected_def = pos.head(3).index.tolist()
            except:
                pass

        # Build Targets
        targets = {}
        targets["SPY"] = W_CORE_SPY
        targets["EWY"] = W_CORE_KOSPI
        targets["GLD"] = W_CORE_GLD

        if signal == "NORMAL":
            targets["QLD"] = W_TACTICAL
        else:
            w_each = W_TACTICAL / len(selected_def)
            for d in selected_def:
                targets[d] = targets.get(d, 0) + w_each

        # 3. Execution (Open T+1)
        # We rebalance if: Signal Change OR Month Change
        # For simplicity in this "Truth" engine, let's rebalance monthly for Core/Defensive
        # AND Daily for Tactical switching.

        is_new_month = today.month != trade_dates[i - 1].month
        needs_trade = (signal != prev_signal) or is_new_month

        # Current Value at Open T+1
        prices_exec = open_prices.loc[next_day]

        equity_open = cash
        for t, count in shares.items():
            if count > 0:
                p = prices_exec.get(t)
                if pd.notna(p):
                    equity_open += count * p
                else:
                    # Missing open price? Use Close T?
                    # If heavily traded e.g. QLD, Open should exist.
                    # If not, let's assume Close T.
                    p_fallback = close_prices.loc[today, t]
                    if pd.notna(p_fallback):
                        equity_open += count * p_fallback

        if needs_trade:
            # Sell All Logic (Virtual) to calc cost
            # Turnover approx
            curr_w = {}
            for t, count in shares.items():
                p = prices_exec.get(t)
                if pd.isna(p):
                    p = close_prices.loc[today, t]  # Fallback
                if pd.notna(p):
                    curr_w[t] = (count * p) / equity_open if equity_open > 0 else 0

            turnover = 0.0
            for t in tickers:
                tw = targets.get(t, 0.0)
                cw = curr_w.get(t, 0.0)
                turnover += abs(tw - cw)

            # Apply Cost
            cost = (
                (turnover / 2) * equity_open * COST_RATE
            )  # /2 because buy+sell=turnover, we pay on Vol.
            # wait, turnover sum abs diff = 200% if full switch.
            # Volume traded = sum(abs(diff)) / 2 * Equity?
            # No. If I sell 100% A and buy 100% B. Diff A = -1, Diff B = +1. Sum Abs = 2.
            # I sold 100k, Bought 100k. Total Vol = 200k.
            # So Cost = Sum(Abs(Diff)) * Equity * Cost_Rate ? No, usually cost is per side.
            # If rate is 0.15% per trade. 100k sell = 150 cost. 100k buy = 150 cost. Total 300.
            # Sum Abs Diff = 2. 2 * 100k * 0.0015 = 300. Correct.

            cost = turnover * equity_open * COST_RATE
            equity_clean = equity_open - cost

            if equity_clean < 0:
                print("BUST")
                break

            # Re-allocate
            cash = equity_clean
            shares = {t: 0.0 for t in tickers}

            for t, w in targets.items():
                if w > 0:
                    p = prices_exec.get(t)
                    if pd.isna(p):
                        p = close_prices.loc[today, t]

                    # BUY VALID CHECK
                    # If p is NaN (Asset not existed yet, e.g. BIL in 2006),
                    # we CANNOT buy. Keep in Cash.
                    if pd.notna(p) and p > 0:
                        amt = equity_clean * w
                        shares[t] = amt / p
                        cash -= amt
                    else:
                        # Asset missing -> Weight stays in Cash
                        # print(f"Skip {t} (No Price)")
                        pass

        # Update State
        prev_signal = signal

        # 4. valuation (Close T+1)
        prices_close_T1 = close_prices.loc[next_day]
        val_usd = cash
        for t, count in shares.items():
            if count > 0:
                p = prices_close_T1.get(t)
                if pd.isna(p):
                    p = close_prices.loc[today, t]  # Fallback
                if pd.notna(p):
                    val_usd += count * p

        k_r = krw.asof(next_day)
        val_krw = val_usd * k_r

        history.append({"Date": next_day, "PV_KRW": val_krw, "Signal": signal})

    res = pd.DataFrame(history).set_index("Date")
    res.to_csv("research/backtest_v4_results.csv")

    # Reports
    start_v = res["PV_KRW"].iloc[0]
    end_v = res["PV_KRW"].iloc[-1]
    days = (res.index[-1] - res.index[0]).days
    cagr = (end_v / start_v) ** (365.25 / days) - 1

    peak = res["PV_KRW"].cummax()
    mdd = ((res["PV_KRW"] - peak) / peak).min()

    print(f"\n[Antigravity v4.0 Final]")
    print(f"Range: {res.index[0].date()} -> {res.index[-1].date()}")
    print(f"CAGR: {cagr * 100:.2f}% (KRW)")
    print(f"MDD : {mdd * 100:.2f}%")
    print(f"End Valuation: {end_v:,.0f} KRW")


if __name__ == "__main__":
    run_backtest_v4()
