import pandas as pd
import numpy as np
import data_loader


def run_pool_comparison():
    print("--- Defensive Pool Comparison: Restricted vs Expanded ---")

    # 1. Setup
    START_DATE = "2007-06-01"

    # Core holdings (always present)
    core = ["SPY", "EWY", "GLD"]

    # Tactical Attack
    attack = ["QLD"]

    # Defensive Pool (Restricted - Current Logic)
    def_pool_restricted = [
        "XLP",
        "XLU",
        "GLD",
        "FXY",
        "UUP",
        "DBC",
        "TLT",
        "IEF",
        "BIL",
        "SHY",
        "SCHP",
        "GSY",
    ]

    # Expanded Pool (All tradable except QLD/QQQ - for momentum chase)
    # Include things like sector ETFs, commodities, intl that might have momentum
    expanded_extras = [
        "XLK",
        "XLV",
        "XLF",
        "XLE",
        "XLI",
        "XLB",
        "XLRE",
        "VNQ",
        "EEM",
        "EFA",
        "VEA",
        "AGG",
        "LQD",
        "HYG",
        "TIP",
    ]
    def_pool_expanded = list(set(def_pool_restricted + expanded_extras))

    # All tickers needed
    tickers = list(set(core + attack + ["QQQ", "KRW=X", "BIL"] + def_pool_expanded))

    try:
        df = data_loader.fetch_validated_data(tickers, start_date="2006-01-01")
    except Exception as e:
        print(f"Data Error: {e}")
        return

    close = df.xs("Close", level=1, axis=1)
    open_p = df.xs("Open", level=1, axis=1)

    # Signal based on QQQ
    sma110 = close["QQQ"].rolling(110).mean()
    sma250 = close["QQQ"].rolling(250).mean()

    # KRW
    if "KRW=X" in close.columns:
        krw = close["KRW=X"].fillna(method="ffill").fillna(1000.0)
    else:
        krw = pd.Series(1200.0, index=close.index)

    start_idx = 260
    trade_dates = close.index

    # Monthly Momentum
    monthly = close.resample("M").last()
    mom = monthly.pct_change(8)

    def simulate(def_pool, label):
        cash = 100_000.0
        shares = {t: 0.0 for t in tickers}
        history = []

        prev_signal = "DANGER"
        selected_def = ["BIL"]

        for i in range(start_idx, len(trade_dates) - 1):
            today = trade_dates[i]
            if today < pd.Timestamp(START_DATE):
                continue

            next_day = trade_dates[i + 1]

            # Monthly Selection
            is_month_end = today.month != next_day.month
            if is_month_end:
                m_idx = mom.index.asof(today)
                if pd.notna(m_idx):
                    try:
                        row = mom.loc[m_idx]
                        # Filter to def_pool only
                        available = [t for t in def_pool if t in row.index]
                        pos = row[available].dropna().sort_values(ascending=False)
                        # Positive momentum only
                        pos = pos[pos > 0]
                        if len(pos) > 0:
                            selected_def = pos.head(3).index.tolist()
                        else:
                            selected_def = ["BIL"]
                    except:
                        selected_def = ["BIL"]

            # Signal
            q = close.loc[today, "QQQ"]
            s1 = sma110.loc[today]
            s2 = sma250.loc[today]

            signal = "DANGER"
            if q > s1 and q > s2:
                signal = "NORMAL"

            # Target Allocation
            targets = {"SPY": 0.2, "EWY": 0.2, "GLD": 0.15}
            w_tac = 0.45

            if signal == "NORMAL":
                targets["QLD"] = w_tac
            else:
                w = w_tac / len(selected_def)
                for d in selected_def:
                    targets[d] = targets.get(d, 0) + w

            # Rebalance
            needs_trade = (signal != prev_signal) or is_month_end

            p_exec = open_p.loc[next_day]
            now_val = cash
            for t, c in shares.items():
                p = p_exec.get(t)
                if pd.isna(p):
                    p = close.loc[today, t] if t in close.columns else 0
                if pd.notna(p):
                    now_val += c * p

            if needs_trade:
                cash = now_val * (1 - 0.0015)
                shares = {t: 0.0 for t in tickers}

                for t, w in targets.items():
                    p = p_exec.get(t) if t in p_exec.index else None
                    if pd.isna(p):
                        p = close.loc[today, t] if t in close.columns else None
                    if pd.notna(p) and p > 0:
                        shares[t] = (cash * w) / p

                cash = 0

            prev_signal = signal

            # Record
            p_close = close.loc[next_day]
            val_usd = cash
            for t, c in shares.items():
                p = p_close.get(t) if t in p_close.index else None
                if pd.isna(p):
                    p = close.loc[today, t] if t in close.columns else 0
                val_usd += c * p

            val_krw = val_usd * krw.asof(next_day)
            history.append({"Date": next_day, "PV": val_krw})

        return pd.DataFrame(history).set_index("Date")

    print("Running Restricted Pool (Current Logic)...")
    res_restricted = simulate(def_pool_restricted, "Restricted")

    print("Running Expanded Pool (All Assets)...")
    res_expanded = simulate(def_pool_expanded, "Expanded")

    def metrics(df, name):
        s = df["PV"]
        cagr = (s.iloc[-1] / s.iloc[0]) ** (
            365.25 / (s.index[-1] - s.index[0]).days
        ) - 1
        peak = s.cummax()
        mdd = ((s - peak) / peak).min()
        vol = s.pct_change().std() * np.sqrt(252)
        sharpe = (cagr - 0.04) / vol if vol > 0 else 0
        print(f"[{name}]")
        print(
            f"  CAGR: {cagr * 100:.2f}% | MDD: {mdd * 100:.2f}% | Vol: {vol * 100:.1f}% | Sharpe: {sharpe:.2f}"
        )
        print(f"  End Value: {s.iloc[-1] / 1e8:.2f}억")

    print("\n" + "=" * 60)
    print("RESULTS (KRW)")
    print("=" * 60)
    metrics(res_restricted, "Restricted Pool (12 Defensive)")
    print("-" * 60)
    metrics(res_expanded, "Expanded Pool (27+ Assets)")


if __name__ == "__main__":
    run_pool_comparison()
