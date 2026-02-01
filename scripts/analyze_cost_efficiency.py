"""
Script: Analyze Cost Efficiency (V4 vs MAMA Lite)
Author: Antigravity
Date: 2026-02-01

Goal: Quantify the impact of trading frequency on net performance.
      - Commission + Slippage: 0.12% one-way (0.24% round trip)
      - Period: 2020-01-01 ~ 2025-12-31
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import logging

# Add paths for MAMA logic
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "signal_mailer"))

try:
    from signal_mailer.mama_lite_predictor import MAMAPredictor
except ImportError:
    sys.path.append("d:/gg")
    sys.path.append("d:/gg/signal_mailer")
    from signal_mailer.mama_lite_predictor import MAMAPredictor

# Silence loggers
logging.getLogger("MAMAPredictor").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)


def calculate_metrics(daily_returns):
    cagr = (1 + daily_returns).prod() ** (252 / len(daily_returns)) - 1
    mdd = (
        (1 + daily_returns)
        .cumprod()
        .div((1 + daily_returns).cumprod().cummax())
        .sub(1)
        .min()
    )
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    return cagr, mdd, sharpe


def run_cost_analysis():
    print("🚀 Analyzing Cost Efficiency: V4 vs MAMA Lite (2020-2025)...")

    # Cost Assumption: 0.1% Slippage + 0.025% Comm = 0.125% one way -> 0.25% Round Trip
    COST_BPS = 25
    cost_rate = COST_BPS / 10000.0

    # 1. Load Data
    start_date = "2019-01-01"
    end_date = "2025-12-31"
    tickers = ["SPY", "VTI", "SCHG", "BIL", "^VIX", "^TNX", "^IRX"]

    print(f"Fetching data...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        df = data["Close"].copy() if "Close" in data.columns.levels[0] else data.copy()
    else:
        df = data.copy()
    df = df.ffill().dropna()

    analysis_df = df.loc["2020-01-01":].copy()

    # 2. V4 Signal Generation
    vti = df["VTI"]
    ma185 = vti.rolling(185).mean()
    upper = ma185 * 1.03
    lower = ma185 * 0.97

    # Crisis
    is_crisis = pd.Series(False, index=df.index)
    if "^TNX" in df.columns and "^IRX" in df.columns:
        is_crisis = (df["^TNX"] - df["^IRX"]) < 0

    v4_signals = pd.Series(index=df.index, dtype=int)
    curr = 1
    for i in range(len(df)):
        price = vti.iloc[i]
        if price > upper.iloc[i]:
            curr = 1
        elif price < lower.iloc[i]:
            curr = 0

        if is_crisis.iloc[i]:
            final = 0
        else:
            final = curr
        v4_signals.iloc[i] = final

    v4_signals = v4_signals.loc["2020-01-01":]

    # 3. MAMA Lite Signal Generation
    print("Calculating MAMA Lite Signals...")
    predictor = MAMAPredictor(
        config_path=os.path.join(parent_dir, "signal_mailer", "config.yaml")
    )

    features = pd.DataFrame(index=df.index)
    features["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df[
        "^VIX"
    ].rolling(252).std()
    features["tnx_mom"] = df["^TNX"].pct_change(20)
    features["spy_mom"] = df["SPY"].pct_change(60)

    valid_indices = features.dropna().index
    feat_clean = features.loc[valid_indices]

    X_srl = predictor.scaler.transform(feat_clean)
    regime_labels = predictor.kmeans.predict(X_srl)

    # Bull Logic
    regime_series = pd.Series(regime_labels, index=valid_indices)
    bull_id = predictor.bull_regime_id
    is_bull = (regime_series == bull_id).astype(int)
    bull_prob = is_bull.rolling(5).mean()

    mama_signals = (bull_prob >= 0.5).astype(int)
    mama_signals = mama_signals.reindex(analysis_df.index).fillna(0).astype(int)

    # 4. Backtest with Cost
    # Asset: SCHG (Aggressive) vs BIL (Defensive)
    schg_ret = analysis_df["SCHG"].pct_change().fillna(0)
    bil_ret = analysis_df["BIL"].pct_change().fillna(0)

    results = []

    for name, signals in [("V4 (Rule)", v4_signals), ("MAMA (AI)", mama_signals)]:
        equity = 1.0
        equity_curve = [1.0]

        trades = 0
        curr_pos = signals.iloc[0]  # Initial position

        # Assume we start with position aligned (no cost for first entry)

        for i in range(1, len(analysis_df)):
            date = analysis_df.index[i]
            ret_chk = schg_ret.iloc[i] if curr_pos == 1 else bil_ret.iloc[i]

            # Position Change?
            new_pos = signals.iloc[i]
            cost = 0.0

            if new_pos != curr_pos:
                # Trade occurred!
                # Sell old, Buy new (or Cash)
                # Apply Round Trip cost relative to portfolio value
                # (Sell entire portfolio, Buy entire portfolio -> Cost on turnover)
                # Let's say cost is on transaction value.
                # Switching 100% equity: Cost = equity * rate
                cost = cost_rate
                trades += 1
                curr_pos = new_pos

            # Apply return then subtract cost
            equity = equity * (1 + ret_chk) * (1 - cost)
            equity_curve.append(equity)

        final_eq = equity_curve[-1]

        # Zero Cost Baseline (for comparison)
        # Just simple signal calc without cost deduction loop (vectorized)
        pos_shift = signals.shift(1).fillna(signals.iloc[0])
        simple_ret = schg_ret * pos_shift + bil_ret * (1 - pos_shift)
        cagr_gross, mdd_gross, _ = calculate_metrics(simple_ret)

        # Net Metrics using equity curve
        # CAGR
        days = len(analysis_df)
        cagr_net = (final_eq) ** (252 / days) - 1

        # MDD
        ec = np.array(equity_curve)
        mdd_net = (ec / np.maximum.accumulate(ec) - 1).min()

        # Sharpe (approx from daily changes of equity curve)
        daily_chg = pd.Series(equity_curve).pct_change().dropna()
        sharpe_net = daily_chg.mean() / daily_chg.std() * np.sqrt(252)

        cost_drag = cagr_gross - cagr_net

        results.append(
            {
                "Strategy": name,
                "Trades": trades,
                "Net CAGR": cagr_net,
                "Net MDD": mdd_net,
                "Sharpe": sharpe_net,
                "Cost Drag": cost_drag,
                "Return/Trade": (cagr_net * 100) / trades if trades > 0 else 0,
            }
        )

    # 5. Output
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("💰 Cost Efficiency Analysis (2020-2025)")
    lines.append(
        f"   Assumption: Slippage+Comm = {COST_BPS * 2 / 100:.2f}% (Round Trip)"
    )
    lines.append("=" * 80)
    lines.append(
        f"{'Metric':<15} | {'V4 (The Tank)':<15} | {'MAMA (The Sprinter)':<20} | {'Diff'}"
    )
    lines.append("-" * 80)

    r_v4 = results[0]
    r_mama = results[1]

    lines.append(
        f"{'Total Trades':<15} | {r_v4['Trades']:<15} | {r_mama['Trades']:<20} | {r_mama['Trades'] - r_v4['Trades']:+d} times"
    )
    lines.append(
        f"{'Avg Trades/Yr':<15} | {r_v4['Trades'] / 6:.1f}{'':<12} | {r_mama['Trades'] / 6:.1f}{'':<17} |"
    )
    lines.append("-" * 80)
    lines.append(
        f"{'Net CAGR':<15} | {r_v4['Net CAGR']:.2%}{'':<9} | {r_mama['Net CAGR']:.2%}{'':<14} | {r_v4['Net CAGR'] - r_mama['Net CAGR']:+.2%}"
    )
    lines.append(
        f"{'Net MDD':<15} | {r_v4['Net MDD']:.2%}{'':<9} | {r_mama['Net MDD']:.2%}{'':<14} | {r_v4['Net MDD'] - r_mama['Net MDD']:+.2%}"
    )
    lines.append(
        f"{'Sharpe':<15} | {r_v4['Sharpe']:.2f}{'':<11} | {r_mama['Sharpe']:.2f}{'':<16} | {r_v4['Sharpe'] - r_mama['Sharpe']:+.2f}"
    )
    lines.append("-" * 80)
    lines.append(
        f"{'Cost Drag':<15} | {r_v4['Cost Drag']:.2%}{'':<9} | {r_mama['Cost Drag']:.2%}{'':<14} | {r_mama['Cost Drag'] - r_v4['Cost Drag']:.2%} (Lost to Fees)"
    )

    lines.append("=" * 80)

    final_output = "\n".join(lines)
    print(final_output)

    with open("d:/gg/cost_efficiency_result.txt", "w", encoding="utf-8") as f:
        f.write(final_output)


if __name__ == "__main__":
    run_cost_analysis()
