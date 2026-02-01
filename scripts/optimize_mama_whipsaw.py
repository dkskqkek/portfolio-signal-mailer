"""
Script: Optimize MAMA Whipsaw Reduction
Author: Antigravity
Date: 2026-02-01

Goal: Reduce false signals (whipsaw) in MAMA strategy.
Hypotheses:
1. Base: 5-day SMA, Threshold 0.5 (Current)
2. Longer Smooth: 10-day / 20-day SMA
3. Hysteresis: Buy > 0.6, Sell < 0.4 (Requires strong conviction)
4. Confirmation: AI Bear Signal only valid if Price < MA120 (Trend Filter)
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import logging

# Add paths
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


def backtest_signal(signals, prices, defensive_prices, cost_rate=0.0025):
    """Simple vectorized backtest with cost"""
    equity = 1.0
    equity_curve = [1.0]
    trades = 0
    curr_pos = signals.iloc[0]

    asset_ret = prices.pct_change().fillna(0)
    defensive_ret = defensive_prices.pct_change().fillna(0)

    for i in range(1, len(signals)):
        ret = asset_ret.iloc[i] if curr_pos == 1 else defensive_ret.iloc[i]

        new_pos = signals.iloc[i]
        cost = 0.0
        if new_pos != curr_pos:
            cost = cost_rate
            trades += 1
            curr_pos = new_pos

        equity = equity * (1 + ret) * (1 - cost)
        equity_curve.append(equity)

    return np.array(equity_curve), trades


def calculate_stats(equity_curve, trades):
    final = equity_curve[-1]
    days = len(equity_curve)
    cagr = (final) ** (252 / days) - 1

    # MDD
    ec = np.array(equity_curve)
    mdd = (ec / np.maximum.accumulate(ec) - 1).min()

    return cagr, mdd, trades


def run_optimization():
    print("🚀 Optimizing MAMA Whipsaws (2020-2025)...")

    # 1. Load Data
    start_date = "2019-01-01"
    end_date = "2025-12-31"
    tickers = ["SPY", "SCHG", "BIL", "^VIX", "^TNX", "^IRX"]

    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        df = data["Close"].copy() if "Close" in data.columns.levels[0] else data.copy()
    else:
        df = data.copy()
    df = df.ffill().dropna()

    analysis_df = df.loc["2020-01-01":].copy()

    # 2. Get Raw AI Probabilities
    print("Generating AI Probabilities...")
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

    bull_id = predictor.bull_regime_id
    is_bull_raw = (pd.Series(regime_labels, index=valid_indices) == bull_id).astype(int)

    # 3. Test Variants
    results = []

    # A. Smoothing Test
    windows = [5, 10, 20, 60]
    for w in windows:
        # Prob = Rolling Mean of Raw Bull Regime (0 or 1)
        prob = is_bull_raw.rolling(w).mean()
        # Signal: Prob >= 0.5
        sig = (prob >= 0.5).astype(int).reindex(analysis_df.index).fillna(0).astype(int)

        eq, tr = backtest_signal(
            sig, analysis_df["SCHG"], analysis_df["BIL"], cost_rate=0.0025
        )
        cagr, mdd, _ = calculate_stats(eq, tr)

        results.append(
            {
                "Method": f"Smooth {w}d",
                "Trades": tr,
                "CAGR": cagr,
                "MDD": mdd,
                "Note": "Base" if w == 5 else "Slower",
            }
        )

    # B. Hysteresis Test (on 10d smooth)
    # Buy > 0.6, Sell < 0.4
    prob_10 = is_bull_raw.rolling(10).mean().reindex(analysis_df.index).fillna(0.5)
    sig_hyst = pd.Series(index=analysis_df.index, dtype=int)
    curr = 1
    for i in range(len(analysis_df)):
        p = prob_10.iloc[i]
        if p > 0.6:
            curr = 1
        elif p < 0.4:
            curr = 0
        # else: keep curr
        sig_hyst.iloc[i] = curr

    eq, tr = backtest_signal(
        sig_hyst, analysis_df["SCHG"], analysis_df["BIL"], cost_rate=0.0025
    )
    cagr, mdd, _ = calculate_stats(eq, tr)
    results.append(
        {
            "Method": "Hysteresis (0.6/0.4)",
            "Trades": tr,
            "CAGR": cagr,
            "MDD": mdd,
            "Note": "Wait for conviction",
        }
    )

    # C. Trend Confirmation (Hybrid)
    # AI Signal (Base 5d) AND Price > MA120
    base_sig = (
        (is_bull_raw.rolling(5).mean() >= 0.5)
        .astype(int)
        .reindex(analysis_df.index)
        .fillna(0)
    )
    spy_ma120 = analysis_df["SPY"].rolling(120).mean()
    price_trend = (analysis_df["SPY"] > spy_ma120).astype(int)

    sig_confirm = pd.Series(index=analysis_df.index, dtype=int)
    curr = 1
    for i in range(len(analysis_df)):
        ai_bull = base_sig.iloc[i] == 1
        trend_bull = price_trend.iloc[i] == 1

        # New Logic: Ignore AI Sell if Trend is Bull
        if curr == 1:
            # Sell If: (AI says Sell) AND (Trend is Broken)
            # -> If Trend is Bull, we IGNORE AI Sell (Whipsaw Filter)
            if (not ai_bull) and (not trend_bull):
                curr = 0
            # If AI says Sell but Trend is Bull -> Hold (Avoid Whipsaw)
        else:
            # Buy If: AI says Buy (Aggressive Entry)
            if ai_bull:
                curr = 1

        sig_confirm.iloc[i] = curr

    eq, tr = backtest_signal(
        sig_confirm, analysis_df["SCHG"], analysis_df["BIL"], cost_rate=0.0025
    )
    cagr, mdd, _ = calculate_stats(eq, tr)
    results.append(
        {
            "Method": "AI + Trend Force",
            "Trades": tr,
            "CAGR": cagr,
            "MDD": mdd,
            "Note": "Sell only if AI & Trend agree",
        }
    )

    # Output
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("🧪 MAMA Anti-Whipsaw Optimization (2020-2025)")
    lines.append("   Benchmark: Cost 0.25% RT, Asset SCHG")
    lines.append("=" * 80)
    lines.append(
        f"{'Method':<20} | {'Trades':<8} | {'CAGR':<8} | {'MDD':<8} | {'Note'}"
    )
    lines.append("-" * 80)

    for r in results:
        lines.append(
            f"{r['Method']:<20} | {r['Trades']:<8} | {r['CAGR']:.2%}   | {r['MDD']:.2%}   | {r['Note']}"
        )

    lines.append("=" * 80)

    final_output = "\n".join(lines)
    print(final_output)

    # Save
    with open("d:/gg/mama_optimize_result.txt", "w", encoding="utf-8") as f:
        f.write(final_output)


if __name__ == "__main__":
    run_optimization()
