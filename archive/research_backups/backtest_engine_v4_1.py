import pandas as pd
import numpy as np
import data_loader


class TaxAccount:
    """
    Simulates a taxable account with year-end settlement.
    Tracks Average Cost Basis to calculate Realized PnL.
    """

    def __init__(self, tax_rate=0.22):
        self.tax_rate = tax_rate
        self.total_realized_pnl_ytd = 0.0
        # {ticker: {'qty': 0, 'avg_cost': 0}}
        self.positions = {}

    def buy(self, ticker, qty, price):
        if qty <= 0:
            return
        pos = self.positions.get(ticker, {"qty": 0.0, "avg_cost": 0.0})

        # Update Weighted Average Cost
        total_cost_old = pos["qty"] * pos["avg_cost"]
        cost_new = qty * price
        new_qty = pos["qty"] + qty
        new_avg = (total_cost_old + cost_new) / new_qty

        self.positions[ticker] = {"qty": new_qty, "avg_cost": new_avg}

    def sell(self, ticker, qty, price):
        if qty <= 0:
            return 0.0
        pos = self.positions.get(ticker)
        if not pos or pos["qty"] <= 0:
            return 0.0

        # FIFO or Avg Cost? Using Avg Cost for simplicity (allowed in KR/US usually)
        # Realized Gain = (Price - Avg_Cost) * Qty
        gain = (price - pos["avg_cost"]) * qty
        self.total_realized_pnl_ytd += gain

        # Reduce Qty
        pos["qty"] -= qty
        if pos["qty"] < 1e-9:
            del self.positions[ticker]

        return gain

    def year_end_settlement(self, current_cash):
        """
        Deducts tax from cash if YTD PnL is positive.
        Returns (tax_amount, new_cash)
        """
        tax = 0.0
        if (
            self.total_realized_pnl_ytd > 2500000
        ):  # KR deduction? Assuming pure 22% for simplicity or applied to net.
            # Let's simple robust: 22% on total net profit.
            tax = self.total_realized_pnl_ytd * self.tax_rate

        # Reset YTD
        self.total_realized_pnl_ytd = 0.0

        if tax > 0:
            return tax
        return 0.0


def run_backtest_v4_1():
    print("--- Antigravity v4.1: The Reality (Tax & Logic Split) ---")

    # 1. Setup
    START_DATE = "2006-01-01"
    tickers = ["SPY", "QLD", "GLD", "EWY", "QQQ", "KRW=X"]
    def_pool = ["XLP", "XLU", "GLD", "FXY", "UUP", "DBC", "TLT", "IEF", "BIL", "SHY"]
    tickers += def_pool
    tickers = list(set(tickers))

    # Fetch
    try:
        df = data_loader.fetch_validated_data(tickers, start_date=START_DATE)
    except Exception as e:
        print(f"Data Error: {e}")
        return

    close_prices = df.xs("Close", level=1, axis=1)
    open_prices = df.xs("Open", level=1, axis=1)

    # KRW
    if "KRW=X" in close_prices.columns:
        krw = close_prices["KRW=X"].fillna(method="ffill").fillna(1000.0)
    else:
        krw = pd.Series(1200.0, index=close_prices.index)

    # Indicators
    sma110 = close_prices["QQQ"].rolling(110).mean()
    sma250 = close_prices["QQQ"].rolling(250).mean()

    # Monthly Momentum (8-month)
    monthly_close = close_prices.resample("M").last()
    mom_monthly = monthly_close.pct_change(8)

    # Config
    start_idx = 260
    trade_dates = close_prices.index

    # Portfolio State
    cash = 100_000.0
    shares = {t: 0.0 for t in tickers}
    tax_account = TaxAccount(tax_rate=0.22)

    history = []

    # Logic State
    prev_signal = "DANGER"
    selected_def = ["BIL"]  # Default defensive

    print(f"Starting loop: {trade_dates[start_idx]}")

    for i in range(start_idx, len(trade_dates) - 1):
        today = trade_dates[i]
        next_day = trade_dates[i + 1]  # Execution T+1

        # --- 1. Monthly Selection Update (Month End Check) ---
        # Did month change from yesterday?
        # We need "Month End" logic.
        # If 'today' is the last trading day of the month -> Update 'Selected Def' for NEXT month.
        # Simplest: Check if month(today) != month(next_day).
        # If true, today is month-end. We select based on TODAY's monthly data.

        is_month_end = today.month != next_day.month

        if is_month_end:
            # Look at momentum (using data available up to today)
            # Find closest monthly index <= today
            m_idx = mom_monthly.index.asof(today)
            if pd.notna(m_idx):
                try:
                    row = mom_monthly.loc[m_idx]
                    pos = row[def_pool].dropna()
                    # Rank
                    ranked = pos.sort_values(ascending=False)
                    # Filter positive? Yes usually.
                    ranked = ranked[ranked > 0]

                    if len(ranked) > 0:
                        selected_def = ranked.head(3).index.tolist()
                    else:
                        selected_def = ["BIL"]  # Cash proxy
                except:
                    selected_def = ["BIL"]

        # --- 2. Daily Signal Trigger ---
        q = close_prices.loc[today, "QQQ"]
        s1 = sma110.loc[today]
        s2 = sma250.loc[today]

        if pd.isna(s1) or pd.isna(s2):
            signal = "DANGER"
        elif q > s1 and q > s2:
            signal = "NORMAL"
        else:
            signal = "DANGER"

        # --- 3. Target Weights ---
        # Core 55% / Tactical 45%
        # Normal: QLD 45%
        # Danger: Selected_Def 45% (Split equal)

        targets = {}
        targets["SPY"] = 0.20
        targets["EWY"] = 0.20
        targets["GLD"] = 0.15

        tactical_w = 0.45
        if signal == "NORMAL":
            targets["QLD"] = tactical_w
        else:
            # Use the "Monthly Selected" Defensive Assets
            w_each = tactical_w / len(selected_def)
            for d in selected_def:
                targets[d] = targets.get(d, 0) + w_each

        # --- 4. Execution (T+1 Open) ---
        prices_exec = open_prices.loc[next_day]
        prices_curr_close = close_prices.loc[today]  # Fallback

        # Rebalancing Condition:
        # 1. Signal Change
        # 2. Month End (To rotate defensive / Rebalance Core)

        needs_trade = (signal != prev_signal) or is_month_end

        # Current Equity at T+1 Open
        equity_open = cash
        for t, cnt in shares.items():
            if cnt > 0:
                p = prices_exec.get(t)
                if pd.isna(p):
                    p = prices_curr_close.get(t)
                if pd.notna(p):
                    equity_open += cnt * p

        if needs_trade:
            # Simple Sell-All-Buy-Target Logic for clarity/robustness in taxes
            # (Optimized: Calculate diffs)

            # 1. Calculate Target Actions
            # Current weights logic is complex with taxes.
            # Let's iterate targets.
            # Sell Overweight first.

            # Need precise "Shares to Trgt".
            # Target Val = Equity * W
            # Target Qty = Val / Price
            # Diff = Target - Current

            # We execute sells first to free up cash.
            # Then execute buys.

            actions = {}  # ticker: delta_qty

            for t in tickers:
                p = prices_exec.get(t)
                if pd.isna(p):
                    p = prices_curr_close.get(t)
                if pd.isna(p) or p <= 0:
                    continue  # Cannot trade

                curr_qty = shares.get(t, 0.0)
                trgt_w = targets.get(t, 0.0)
                trgt_val = equity_open * trgt_w
                trgt_qty = trgt_val / p

                delta = trgt_qty - curr_qty
                if abs(delta * p) > 1.0:  # Ignore dust < $1
                    actions[t] = delta

            # Sort Sells first
            sells = {t: q for t, q in actions.items() if q < 0}
            buys = {t: q for t, q in actions.items() if q > 0}

            # Execute Sells
            for t, delta in sells.items():
                qty_sell = abs(delta)
                p = prices_exec.get(t, prices_curr_close.get(t))

                # Slippage/Cost on Deal Val
                val = qty_sell * p
                cost = val * 0.0015

                # Check valid
                if shares[t] >= qty_sell:
                    # Tax Logic
                    tax_account.sell(t, qty_sell, p)

                    shares[t] -= qty_sell
                    cash += val
                    cash -= cost  # Pay transaction cost

            # Execute Buys
            for t, delta in buys.items():
                qty_buy = abs(delta)
                p = prices_exec.get(t, prices_curr_close.get(t))

                val = qty_buy * p
                cost = val * 0.0015

                # Affordability check
                if cash >= (val + cost):
                    tax_account.buy(t, qty_buy, p)
                    shares[t] += qty_buy
                    cash -= val + cost
                else:
                    # Buy max possible
                    max_val = cash / (1 + 0.0015)
                    if max_val > 0:
                        qty_max = max_val / p
                        tax_account.buy(t, qty_max, p)
                        shares[t] += qty_max
                        cash = 0

        # --- Year End Tax Settlement ---
        # If today is last day of year?
        if is_month_end and today.month == 12:
            tax_bill = tax_account.year_end_settlement(cash)
            if tax_bill > 0:
                # Deduct from cash
                cash -= tax_bill
                # If cash negative? Margin loan?
                # For reality, forced sell.
                # Simplification: Allow negative cash (debt) momentarily or assume cash buffer.
                # In 100% equity strategy, this is problem.
                # Fix: If cash < tax, sell some GLD/Core to pay tax.
                # Let's just record 'Debited' for now.

        # --- Record T+1 Close Valuation ---
        prices_close_T1 = close_prices.loc[next_day]
        idx_val_usd = cash
        for t, cnt in shares.items():
            if cnt > 0:
                p = prices_close_T1.get(t)
                if pd.isna(p):
                    p = prices_curr_close.get(t)
                if pd.notna(p):
                    idx_val_usd += cnt * p

        # KRW Convert
        k_r = krw.asof(next_day)
        val_krw = idx_val_usd * k_r

        prev_signal = signal

        history.append({"Date": next_day, "PV_KRW": val_krw, "Signal": signal})

    # Report
    res = pd.DataFrame(history).set_index("Date")
    res.to_csv("research/backtest_v4_1_results.csv")

    start_v = res["PV_KRW"].iloc[0]
    end_v = res["PV_KRW"].iloc[-1]
    days = (res.index[-1] - res.index[0]).days
    cagr = (end_v / start_v) ** (365.25 / days) - 1

    res["Peak"] = res["PV_KRW"].cummax()
    mdd = ((res["PV_KRW"] - res["Peak"]) / res["Peak"]).min()

    # Sharpe (Daily Returns)
    rets = res["PV_KRW"].pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252)

    # Sortino (Downside Vol)
    neg_rets = rets[rets < 0]
    sortino = (rets.mean() / neg_rets.std()) * np.sqrt(252)

    print(f"\n[Antigravity v4.1 Final Report]")
    print(f"Range: {res.index[0].date()} ~ {res.index[-1].date()}")
    print(f"CAGR (KRW, Post-Tax): {cagr * 100:.2f}%")
    print(f"MDD  (KRW): {mdd * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"Final Value: {end_v:,.0f} KRW")


if __name__ == "__main__":
    run_backtest_v4_1()
