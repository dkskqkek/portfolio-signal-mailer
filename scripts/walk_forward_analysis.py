"""
Script: Walk-Forward Analysis
Author: Antigravity
Date: 2026-02-01

Goal: Test consistency across different training windows
- Window Combinations: (5,2), (8,2), (10,2), (15,5)
- Method: Rolling Window (Train -> Test -> Slide)
- Metric: Win Rate, Average CAGR, Sharpe Distribution
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WalkForward")


def backtest_signal(signals, prices, defensive_prices, cost_rate=0.0025):
    """Simple backtest"""
    equity = 1.0
    trades = 0
    curr_pos = signals.iloc[0]

    asset_ret = prices.pct_change().fillna(0)
    defensive_ret = defensive_prices.pct_change().fillna(0)

    for i in range(1, len(signals)):
        ret = asset_ret.iloc[i] if curr_pos == 1 else defensive_ret.iloc[i]
        new_pos = signals.iloc[i]

        if new_pos != curr_pos:
            equity = equity * (1 - cost_rate)
            trades += 1
            curr_pos = new_pos

        equity = equity * (1 + ret)

    cagr = (equity) ** (252 / len(signals)) - 1 if len(signals) > 0 else 0
    return cagr, trades


def generate_mama_opt_signal(df, scaler, kmeans, bull_regime_id):
    """Generate MAMA Opt signal using OOS model"""
    # Features
    df["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df["^VIX"].rolling(
        252
    ).std()
    df["tnx_mom"] = df["^TNX"].pct_change(20)
    df["spy_mom"] = df["SPY"].pct_change(60)

    features = df[["vix_z", "tnx_mom", "spy_mom"]].dropna()

    # Transform
    X_scaled = scaler.transform(features)
    regime_labels = kmeans.predict(X_scaled)

    # Base signal
    is_bull_raw = (
        pd.Series(regime_labels, index=features.index) == bull_regime_id
    ).astype(int)
    mama_base_prob = is_bull_raw.rolling(5).mean()
    mama_base_sig = (
        (mama_base_prob >= 0.5).astype(int).reindex(df.index).fillna(0).astype(int)
    )

    # Trend Force
    spy_ma120 = df["SPY"].rolling(120).mean()
    trend_bull = (df["SPY"] > spy_ma120).astype(int)

    mama_opt_sig = pd.Series(index=df.index, dtype=int)
    curr = 1
    for i in range(len(df)):
        ai_bull = mama_base_sig.iloc[i] == 1
        trend_bull_now = trend_bull.iloc[i] == 1

        if curr == 1:
            if (not ai_bull) and (not trend_bull_now):
                curr = 0
        else:
            if ai_bull:
                curr = 1

        mama_opt_sig.iloc[i] = curr

    return mama_opt_sig


def walk_forward_analysis():
    """Run Walk-Forward Analysis"""

    logger.info("🚀 Walk-Forward Analysis Starting...")

    # Window configurations
    windows = [
        (5, 2, "5Y Train / 2Y Test"),
        (8, 2, "8Y Train / 2Y Test"),
        (10, 2, "10Y Train / 2Y Test"),
        (15, 5, "15Y Train / 5Y Test"),
    ]

    # Download full data (2004-2025)
    tickers = ["SPY", "SCHG", "BIL", "^VIX", "^TNX", "^IRX"]
    data = yf.download(tickers, start="2004-01-01", end="2025-12-31", progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        df_full = (
            data["Close"].copy() if "Close" in data.columns.levels[0] else data.copy()
        )
    else:
        df_full = data.copy()

    df_full = df_full.ffill().dropna()

    logger.info(f"Downloaded {len(df_full)} days of data")

    all_results = []

    for train_years, test_years, window_name in windows:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing Window: {window_name}")
        logger.info(f"{'=' * 60}")

        window_results = []

        # Sliding window
        start_year = 2004
        end_year = 2025

        current_year = start_year
        window_id = 1

        while current_year + train_years + test_years <= end_year:
            train_start = f"{current_year}-01-01"
            train_end = f"{current_year + train_years - 1}-12-31"
            test_start = f"{current_year + train_years}-01-01"
            test_end = f"{current_year + train_years + test_years - 1}-12-31"

            logger.info(
                f"  Window {window_id}: Train={train_start} to {train_end}, Test={test_start} to {test_end}"
            )

            # Train
            df_train = df_full.loc[train_start:train_end].copy()

            if len(df_train) < 252:
                logger.warning(f"    ⚠️ Insufficient training data, skipping...")
                current_year += 2
                window_id += 1
                continue

            # Calculate features
            df_train["vix_z"] = (
                df_train["^VIX"] - df_train["^VIX"].rolling(252).mean()
            ) / df_train["^VIX"].rolling(252).std()
            df_train["tnx_mom"] = df_train["^TNX"].pct_change(20)
            df_train["spy_mom"] = df_train["SPY"].pct_change(60)

            features_train = df_train[["vix_z", "tnx_mom", "spy_mom"]].dropna()

            if len(features_train) < 100:
                current_year += 2
                window_id += 1
                continue

            # Train Scaler & KMeans
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(features_train)

            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            regime_labels_train = kmeans.fit_predict(X_train_scaled)

            # Identify Bull
            features_train_copy = features_train.copy()
            features_train_copy["spy_ret"] = (
                df_train["SPY"].pct_change().reindex(features_train.index)
            )
            features_train_copy["regime"] = regime_labels_train

            regime_spy_ret = features_train_copy.groupby("regime")["spy_ret"].mean()
            bull_regime_id = int(regime_spy_ret.idxmax())

            # Test
            df_test = df_full.loc[test_start:test_end].copy()

            if len(df_test) < 50:
                current_year += 2
                window_id += 1
                continue

            # Generate signal
            signal = generate_mama_opt_signal(df_test, scaler, kmeans, bull_regime_id)

            # Backtest
            cagr, trades = backtest_signal(signal, df_test["SCHG"], df_test["BIL"])

            logger.info(f"    Result: CAGR={cagr:.2%}, Trades={trades}")

            window_results.append(
                {
                    "window_config": window_name,
                    "window_id": window_id,
                    "train_period": f"{train_start} to {train_end}",
                    "test_period": f"{test_start} to {test_end}",
                    "cagr": cagr,
                    "trades": trades,
                }
            )

            # Slide
            current_year += 2
            window_id += 1

        # Summary for this window config
        if window_results:
            cagrs = [r["cagr"] for r in window_results]
            win_rate = sum(1 for c in cagrs if c > 0) / len(cagrs)
            avg_cagr = np.mean(cagrs)

            all_results.append(
                {
                    "window_config": window_name,
                    "num_windows": len(window_results),
                    "win_rate": win_rate,
                    "avg_cagr": avg_cagr,
                    "worst_cagr": min(cagrs),
                    "best_cagr": max(cagrs),
                }
            )

            logger.info(f"  Summary: Win Rate={win_rate:.1%}, Avg CAGR={avg_cagr:.2%}")

    # Output
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("🔄 Walk-Forward Analysis Results")
    lines.append("=" * 80)
    lines.append(
        f"{'Window Config':<20} | {'Windows':<8} | {'Win Rate':<10} | {'Avg CAGR':<10} | {'Best':<8} | {'Worst'}"
    )
    lines.append("-" * 80)

    for r in all_results:
        lines.append(
            f"{r['window_config']:<20} | {r['num_windows']:<8} | {r['win_rate']:.1%}      | {r['avg_cagr']:.2%}     | {r['best_cagr']:.2%}  | {r['worst_cagr']:.2%}"
        )

    lines.append("=" * 80)
    lines.append("\n✅ Target: Win Rate > 60%")

    for r in all_results:
        status = "✅ PASS" if r["win_rate"] > 0.6 else "⚠️ FAIL"
        lines.append(f"   {r['window_config']}: {r['win_rate']:.1%} - {status}")

    lines.append("=" * 80)

    final_output = "\n".join(lines)
    print(final_output)

    # Save
    with open("d:/gg/walk_forward_result.txt", "w", encoding="utf-8") as f:
        f.write(final_output)

    logger.info("✅ Results saved to d:/gg/walk_forward_result.txt")


if __name__ == "__main__":
    walk_forward_analysis()
