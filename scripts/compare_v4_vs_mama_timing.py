"""
Script: Compare Buy/Sell Timing (V4 vs MAMA Lite)
Author: Antigravity
Date: 2026-02-01

Goal: Visualize and list the exact dates when V4 and MAMA Lite switched regimes (Bull <-> Bear).
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
    # Fallback if running from scripts folder
    sys.path.append("d:/gg")
    sys.path.append("d:/gg/signal_mailer")
    from signal_mailer.mama_lite_predictor import MAMAPredictor

# Silence loggers
logging.getLogger("MAMAPredictor").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)


def get_v4_signal(df):
    """Calculate V4 Signal (Rule-based)"""
    vti = df["VTI"]
    ma185 = vti.rolling(185).mean()
    upper = ma185 * 1.03
    lower = ma185 * 0.97

    signals = pd.Series(index=df.index, dtype=int)
    curr = 1  # 1: Bull, 0: Bear/Crisis

    # Crisis Logic (Immediate Inversion)
    if "^TNX" in df.columns and "^IRX" in df.columns:
        spread = df["^TNX"] - df["^IRX"]
        is_crisis = spread < 0
    else:
        is_crisis = pd.Series(False, index=df.index)

    for i in range(len(df)):
        price = vti.iloc[i]

        # MA Logic (Buffer)
        if price > upper.iloc[i]:
            curr = 1
        elif price < lower.iloc[i]:
            curr = 0

        # Crisis Override
        if is_crisis.iloc[i]:
            final_sig = 0  # Crisis = Cash
        else:
            final_sig = curr  # Bear = Cash (or partial) -> Treat as "Risk Off"

        signals.iloc[i] = final_sig

    return signals


def run_comparison():
    print("🚀 Comparing V4 vs MAMA Lite Timing (2020-2025)...")

    # 1. Load Data
    start_date = "2019-01-01"  # Need buffer for MA185
    end_date = "2025-12-31"
    tickers = ["SPY", "VTI", "^VIX", "^TNX", "^IRX"]

    print(f"Fetching data for {tickers}...")
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        df = data["Close"].copy() if "Close" in data.columns.levels[0] else data.copy()
    else:
        df = data.copy()
    df = df.ffill().dropna()

    # 2. V4 Signals
    v4_signals = get_v4_signal(df)
    v4_signals = v4_signals.loc["2020-01-01":]  # Cut to analysis period

    # 3. MAMA Lite Signals
    # Need to instantiate predictor and run regime detection simulation
    print("Running MAMA Lite Regime Inference...")
    predictor = MAMAPredictor(
        config_path=os.path.join(parent_dir, "signal_mailer", "config.yaml")
    )

    mama_signals = pd.Series(index=df.index, dtype=int)
    # To save time, we vectorise SRL feature calc (Predictor has code for it)
    # but Predictor uses KMeans on normalized features.

    # Re-implement SRL feature calc loop or use predictor's hidden method if possible.
    # predictor.update_regime_history takes a DF. But it appends to history.
    # We want historical array.

    # Let's manually calculate SRL features and predict using predictor.kmeans
    features = pd.DataFrame(index=df.index)
    features["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df[
        "^VIX"
    ].rolling(252).std()
    features["tnx_mom"] = df["^TNX"].pct_change(20)
    features["spy_mom"] = df["SPY"].pct_change(60)

    # Drop NaNs
    valid_indices = features.dropna().index
    feat_clean = features.loc[valid_indices]

    # Predict
    X_srl = predictor.scaler.transform(feat_clean)
    regime_labels = predictor.kmeans.predict(X_srl)

    # Create Bull Logic (Smoothing Window 5)
    regime_series = pd.Series(regime_labels, index=valid_indices)

    # Bull Regime ID
    bull_id = predictor.bull_regime_id

    # Rolling 5-day prob
    is_bull_raw = (regime_series == bull_id).astype(int)
    bull_prob = is_bull_raw.rolling(5).mean()

    # MAMA Logic: Stock Weight = Bull Prob. If < 0.2 -> Defensive.
    # Treat Weight >= 0.5 as "Bull Signal" (Risk On), < 0.5 as "Bear Signal" (Risk Off)
    mama_sig_raw = (bull_prob >= 0.5).astype(int)
    mama_signals.loc[valid_indices] = mama_sig_raw
    mama_signals = mama_signals.loc["2020-01-01":].fillna(0).astype(int)

    # 4. Compare
    events = []

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("📅 Signal Change Comparison (Risk Off = Bear/Crisis)")
    output_lines.append("=" * 80)
    output_lines.append(f"{'Date':<12} | {'Strategy':<10} | {'Action':<15} | {'Note'}")
    output_lines.append("-" * 80)

    # Iterate dates
    common_idx = v4_signals.index.intersection(mama_signals.index)

    v4_curr = v4_signals.iloc[0]
    mama_curr = mama_signals.iloc[0]

    for date in common_idx:
        d_str = date.strftime("%Y-%m-%d")

        # V4 Change
        v4_new = v4_signals.loc[date]
        if v4_new != v4_curr:
            action = "🔴 SELL" if v4_new == 0 else "🟢 BUY"
            output_lines.append(
                f"{d_str:<12} | V4         | {action:<15} | Risk On/Off"
            )
            v4_curr = v4_new
            events.append({"date": date, "strat": "V4", "action": v4_new})

        # MAMA Change
        mama_new = mama_signals.loc[date]
        if mama_new != mama_curr:
            action = "🔴 SELL" if mama_new == 0 else "🟢 BUY"
            output_lines.append(
                f"{d_str:<12} | MAMA Lite  | {action:<15} | Risk On/Off"
            )
            mama_curr = mama_new
            events.append({"date": date, "strat": "MAMA", "action": mama_new})

    output_lines.append("-" * 80)

    # Summary of efficiency
    # Check 2020 COVID and 2022 Bear
    output_lines.append("\n🧐 Major Crisis Analysis:")

    # 2020 COVID
    covid_start = pd.Timestamp("2020-02-15")
    covid_end = pd.Timestamp("2020-04-01")
    output_lines.append(f"1. COVID-19 ({covid_start.date()} ~ {covid_end.date()})")

    # Find last sell before/during
    v4_covid = [
        e
        for e in events
        if e["strat"] == "V4"
        and e["action"] == 0
        and "2020-01" <= e["date"].strftime("%Y-%m") <= "2020-03"
    ]
    mama_covid = [
        e
        for e in events
        if e["strat"] == "MAMA"
        and e["action"] == 0
        and "2020-01" <= e["date"].strftime("%Y-%m") <= "2020-03"
    ]

    if v4_covid:
        output_lines.append(f"   - V4 Sold:   {v4_covid[-1]['date'].date()}")
    else:
        output_lines.append("   - V4: No Sell Signal")

    if mama_covid:
        output_lines.append(f"   - MAMA Sold: {mama_covid[-1]['date'].date()}")
    else:
        output_lines.append("   - MAMA: No Sell Signal")

    # 2022 Bear
    bear_start = pd.Timestamp("2022-01-01")
    output_lines.append(f"\n2. 2022 Inflation Bear ({bear_start.date()} ~ )")

    v4_bear = [
        e
        for e in events
        if e["strat"] == "V4"
        and e["action"] == 0
        and "2021-12" <= e["date"].strftime("%Y-%m") <= "2022-06"
    ]
    mama_bear = [
        e
        for e in events
        if e["strat"] == "MAMA"
        and e["action"] == 0
        and "2021-12" <= e["date"].strftime("%Y-%m") <= "2022-06"
    ]

    if v4_bear:
        output_lines.append(
            f"   - V4 Sold:   {v4_bear[0]['date'].date()}"
        )  # First sell
    else:
        output_lines.append("   - V4: No Sell Signal")

    if mama_bear:
        output_lines.append(
            f"   - MAMA Sold: {mama_bear[0]['date'].date()}"
        )  # First sell
    else:
        output_lines.append("   - MAMA: No Sell Signal")

    output_lines.append("=" * 80)

    final_output = "\n".join(output_lines)
    print(final_output)

    with open("d:/gg/comparison_result.txt", "w", encoding="utf-8") as f:
        f.write(final_output)


if __name__ == "__main__":
    run_comparison()
