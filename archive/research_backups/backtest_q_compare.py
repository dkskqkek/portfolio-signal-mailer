import pandas as pd
import numpy as np
import data_loader


def run_leverage_comparison():
    print("--- Antigravity: QQQ (1x) vs QLD (2x) Attacker Comparison ---")

    # 1. Setup
    START_DATE = "2007-06-01"  # Post-BIL inception for simplicity, covering 2008
    tickers = ["SPY", "QLD", "QQQ", "GLD", "EWY", "KRW=X", "BIL"]
    # Defensive Pool
    def_pool = ["XLP", "XLU", "GLD", "FXY", "UUP", "DBC", "TLT", "IEF", "BIL", "SHY"]
    tickers += def_pool
    tickers = list(set(tickers))

    try:
        df = data_loader.fetch_validated_data(tickers, start_date="2006-01-01")
    except:
        return

    close = df.xs("Close", level=1, axis=1)
    open_p = df.xs("Open", level=1, axis=1)  # Execution price

    # Indicators based on QQQ (Signal Source is constant)
    sma110 = close["QQQ"].rolling(110).mean()
    sma250 = close["QQQ"].rolling(250).mean()

    # KRW
    if "KRW=X" in close.columns:
        krw = close["KRW=X"].fillna(method="ffill").fillna(1000.0)
    else:
        krw = pd.Series(1200.0, index=close.index)

    start_idx = 260
    trade_dates = close.index

    # --- Simulator Function ---
    def simulate(tactical_ticker):
        cash = 100_000.0
        shares = {t: 0.0 for t in tickers}
        history = []

        # Logic State
        prev_signal = "DANGER"
        selected_def = ["BIL"]

        # Monthly Mom
        monthly = close.resample("M").last()
        mom = monthly.pct_change(8)

        for i in range(start_idx, len(trade_dates) - 1):
            today = trade_dates[i]
            if today < pd.Timestamp(START_DATE):
                continue

            next_day = trade_dates[i + 1]

            # 1. Monthly Selection
            is_month_end = today.month != next_day.month
            if is_month_end:
                m_idx = mom.index.asof(today)
                if pd.notna(m_idx):
                    try:
                        row = mom.loc[m_idx]
                        pos = row[def_pool].dropna().sort_values(ascending=False)
                        pos = pos[pos > 0]
                        if len(pos) > 0:
                            selected_def = pos.head(3).index.tolist()
                        else:
                            selected_def = ["BIL"]
                    except:
                        pass

            # 2. Signal
            q = close.loc[today, "QQQ"]
            s1 = sma110.loc[today]
            s2 = sma250.loc[today]

            signal = "DANGER"
            if q > s1 and q > s2:
                signal = "NORMAL"

            # 3. Target
            targets = {"SPY": 0.2, "EWY": 0.2, "GLD": 0.15}
            w_tac = 0.45

            if signal == "NORMAL":
                targets[tactical_ticker] = w_tac
            else:
                w = w_tac / len(selected_def)
                for d in selected_def:
                    targets[d] = targets.get(d, 0) + w

            # 4. Rebalance Check
            needs_trade = (signal != prev_signal) or is_month_end

            # Value at T+1 Open
            p_exec = open_p.loc[next_day]
            now_val = cash
            for t, c in shares.items():
                p = p_exec.get(t)
                if pd.isna(p):
                    p = close.loc[today, t]
                if pd.notna(p):
                    now_val += c * p

            if needs_trade:
                # Sell All Sim
                cash = now_val * (1 - 0.0015)  # Cost approx
                shares = {t: 0.0 for t in tickers}

                for t, w in targets.items():
                    p = p_exec.get(t)
                    if pd.isna(p):
                        p = close.loc[today, t]
                    if pd.notna(p) and p > 0:
                        shares[t] = (cash * w) / p

                cash = 0  # Invested

            prev_signal = signal

            # Record Close Value (KRW)
            p_close = close.loc[next_day]
            val_usd = cash
            for t, c in shares.items():
                p = p_close.get(t)
                if pd.isna(p):
                    p = close.loc[today, t]
                val_usd += c * p

            # Tax Deferral ignore for comparison speed (NAV only)
            val_krw = val_usd * krw.asof(next_day)
            history.append({"Date": next_day, "PV": val_krw})

        return pd.DataFrame(history).set_index("Date")

    print("Running QQQ (1x)...")
    res_qqq = simulate("QQQ")
    print("Running QLD (2x)...")
    res_qld = simulate("QLD")

    # Calc Metrics
    def metrics(df, name):
        s = df["PV"]
        cagr = (s.iloc[-1] / s.iloc[0]) ** (
            365.25 / (s.index[-1] - s.index[0]).days
        ) - 1
        peak = s.cummax()
        mdd = ((s - peak) / peak).min()
        print(
            f"[{name}] CAGR: {cagr * 100:.2f}% | MDD: {mdd * 100:.2f}% | End: {s.iloc[-1] / 1e8:.2f}억"
        )

    print("\n--- RESULTS (KRW) ---")
    metrics(res_qqq, "Strategy with QQQ")
    metrics(res_qld, "Strategy with QLD")


if __name__ == "__main__":
    run_leverage_comparison()
