"""
Script: Simulate FIRE Plan (Initial + DCA)
Author: Antigravity
Date: 2026-02-01

Scenario:
- Initial Capital: 200,000,000 KRW
- Monthly Contribution: 1,000,000 KRW
- Horizon: 10 Years
- Strategy: Antigravity V4 Final (Hybrid SCHG)
- Method: Bootstrap 10,000 runs

Assumption:
- Returns are applied directly to KRW (ignoring FX rate fluctuation or assuming neutral FX impact long-term).
- Monthly contribution is added every 21 trading days.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import locale

# Try to set locale for currency formatting, fallback if fails
try:
    locale.setlocale(locale.LC_ALL, "ko_KR.UTF-8")
except:
    pass


def format_krw(val):
    # Simple Korean format: 12억 3,450만원
    eok = int(val // 100000000)
    man = int((val % 100000000) // 10000)

    if eok > 0:
        return f"{eok}억 {man:,}만원"
    else:
        return f"{man:,}만원"


def run_fire_simulation():
    print("🚀 Starting FIRE Plan Simulation...")
    print("   - Initial: 200,000,000 KRW")
    print("   - Monthly:   1,000,000 KRW")
    print("   - Period:  10 Years")
    print("   - Logic:   V4 Hybrid SCHG")

    # 1. Get V4 Strategy Returns (Code Reuse from monte_carlo_v4_final.py logic)
    # Ideally we should import this, but for standalone robustness, let's re-fetch quickly.
    tickers = ["VTI", "SCHG", "^TNX", "^IRX"]
    data = yf.download(tickers, start="2010-01-01", end="2025-12-31", progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        df = data["Close"].copy() if "Close" in data.columns.levels[0] else data.copy()
    else:
        df = data.copy()
    df = df.ffill().dropna()

    # Strategy Calculation
    vti = df["VTI"]
    ma185 = vti.rolling(185).mean()
    upper = ma185 * 1.03
    lower = ma185 * 0.97

    trend = np.zeros(len(df))
    curr = 1
    for i in range(len(df)):
        if vti.iloc[i] > upper.iloc[i]:
            curr = 1
        elif vti.iloc[i] < lower.iloc[i]:
            curr = -1
        trend[i] = curr

    is_crisis = (df["^TNX"] - df["^IRX"]) < 0

    weights = []
    for i in range(len(df)):
        if trend[i] == 1:
            w = 1.0
        else:
            w = 0.0 if is_crisis.iloc[i] else 0.3
        weights.append(w)

    schg_ret = df["SCHG"].pct_change().fillna(0)
    strat_ret = schg_ret * pd.Series(weights, index=df.index).shift(1).fillna(0)

    # 2. Simulation (DCA Mode + XIRR + MDD)
    n_simulations = 10000
    years = 10
    days = 252 * years
    months = 12 * years

    initial_krw = 200_000_000
    monthly_krw = 1_000_000

    pool = strat_ret.values
    total_invested = initial_krw + (monthly_krw * months)

    print(f"\nRunning {n_simulations} simulations (calculating XIRR & MDD)...")

    # Results containers
    sim_final_values = []
    sim_xirrs = []
    sim_mdds = []

    # Pre-calculate cashflow factors for XIRR solver to speed up
    # Equation: Initial*(1+r)^N + Monthly * Sum((1+r)^(N-i)) - Final = 0
    # Let's use a simplified Monthly IRR solver (assuming regular monthly intervals)

    from scipy import optimize

    def solve_monthly_irr(final_val):
        # f(r) = P0*(1+r)^N + PMT * ((1+r)^N - 1)/r * (1+r) - FV = 0  (Prepaid annuity approx)
        # Actually in simulation we add monthly_krw every 21 days.
        # Let's treat it as: Period 0: Initial, Period 1..120: Monthly input.
        # But wait, equity grows daily.
        # Let's approximate XIRR by finding monthly rate 'rm' such that:
        # 200M * (1+rm)^120 + 1M * sum_{k=0 to 119} (1+rm)^(120 - k) = FinalValue
        # (Assuming contribution at start of month relative to that month's growth?)
        # In sim loop: 21 days check. So it's roughly end of month/start of next.
        # Let's use standard annuity due formula for approximation.
        # FV = PV*(1+r)^n + PMT * [ ((1+r)^n - 1)/r ]
        # We solve for r.

        def func(r):
            if r == 0:
                return initial_krw + monthly_krw * months - final_val
            # Regular Annuity Future Value (contributions at end of period)
            # Our sim adds 1M every 21 days.
            fv_annuity = monthly_krw * (((1 + r) ** months - 1) / r)
            fv_initial = initial_krw * ((1 + r) ** months)
            return (fv_initial + fv_annuity) - final_val

        try:
            rm = optimize.newton(func, 0.01, maxiter=50)
            return (1 + rm) ** 12 - 1  # Annualized XIRR
        except:
            return 0.0

    for i in range(n_simulations):
        # Sample random path
        path = np.random.choice(pool, size=days, replace=True)

        equity = initial_krw
        equity_curve = [equity]

        for day, ret in enumerate(path):
            # Apply Return
            equity *= 1 + ret

            # Apply Monthly Contribution (approx every 21 days)
            if (day + 1) % 21 == 0:
                equity += monthly_krw

            equity_curve.append(equity)

        sim_final_values.append(equity)

        # MDD Calc
        ec = np.array(equity_curve)
        cummax = np.maximum.accumulate(ec)
        drawdown = (ec - cummax) / cummax
        mdd = drawdown.min()
        sim_mdds.append(mdd)

        # XIRR Calc
        xirr = solve_monthly_irr(equity)
        sim_xirrs.append(xirr)

        if (i + 1) % 1000 == 0:
            print(f"Progress: {i + 1}/{n_simulations}")

    sim_final_values = np.array(sim_final_values)
    sim_xirrs = np.array(sim_xirrs)
    sim_mdds = np.array(sim_mdds)

    # 3. Report
    mean_val = np.mean(sim_final_values)
    worst_5_val = np.percentile(sim_final_values, 5)
    best_5_val = np.percentile(sim_final_values, 95)

    mean_xirr = np.mean(sim_xirrs)
    worst_5_xirr = np.percentile(sim_xirrs, 5)

    mean_mdd = np.mean(sim_mdds)
    worst_5_mdd = np.percentile(sim_mdds, 5)

    print("\n" + "=" * 60)
    print("💰 FIRE Plan Result (10 Years) - XIRR & MDD")
    print(f"   Total Invested Principal: {format_krw(total_invested)}")
    print("=" * 60)

    print(f"1. Expectancy (Average):")
    print(f"   ▶ Final Asset: {format_krw(mean_val)}")
    print(f"   ▶ XIRR (Annual): {mean_xirr:.2%} (True ROI)")
    print(f"   ▶ MDD: {mean_mdd:.2%} (Avg Drawdown)")

    print(f"\n2. Worst Case (Bottom 5%):")
    print(f"   ▶ Final Asset: {format_krw(worst_5_val)}")
    print(f"   ▶ XIRR (Annual): {worst_5_xirr:.2%}")
    print(f"   ▶ MDD: {worst_5_mdd:.2%} (Crash Depth)")

    print(f"\n3. Best Case (Top 5%):")
    print(f"   ▶ Final Asset: {format_krw(best_5_val)}")
    print("=" * 60)

    # Save simple log
    with open("d:/gg/fire_plan_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Mean Asset: {mean_val}\n")
        f.write(f"Mean XIRR: {mean_xirr}\n")
        f.write(f"Mean MDD: {mean_mdd}\n")
        f.write(f"Worst Asset: {worst_5_val}\n")
        f.write(f"Worst XIRR: {worst_5_xirr}\n")
        f.write(f"Worst MDD: {worst_5_mdd}\n")


if __name__ == "__main__":
    run_fire_simulation()
