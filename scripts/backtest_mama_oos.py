"""
Script: Out-of-Sample Backtest (2020-2025)
Author: Antigravity
Date: 2026-02-01

Goal: Test MAMA strategies on completely unseen data
- Model: kmeans_model_oos.pkl (trained on 2004-2019)
- Test Period: 2020-01-01 ~ 2025-12-31
- Strategies: MAMA Lite (Base), MAMA Opt (AI + Trend Force), V4, Buy & Hold
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BacktestOOS")

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "data", "gnn")

MODEL_PATH = os.path.join(data_dir, "kmeans_model_oos.pkl")


def backtest_signal(signals, prices, defensive_prices, cost_rate=0.0025):
    """Simple vectorized backtest with transaction cost"""
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

    # Sharpe
    daily_ret = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (
        (daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if daily_ret.std() > 0
        else 0
    )

    return cagr, mdd, sharpe, trades


def run_oos_backtest():
    """Run Out-of-Sample Backtest"""

    logger.info("🚀 Out-of-Sample Backtest (2020-2025)...")

    # 1. Load OOS Model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ OOS Model not found at {MODEL_PATH}")
        logger.error("Please run train_kmeans_oos.py first")
        return

    model_data = joblib.load(MODEL_PATH)
    scaler = model_data["scaler"]
    kmeans = model_data["kmeans"]
    bull_regime_id = model_data["bull_regime_id"]

    logger.info(f"✅ Loaded OOS Model (Trained: {model_data['training_period']})")
    logger.info(f"Bull Regime: Cluster {bull_regime_id}")

    # 2. Download Test Data (2020-2025)
    test_start = "2020-01-01"
    test_end = "2025-12-31"

    tickers = ["SPY", "SCHG", "BIL", "^VIX", "^TNX", "^IRX"]
    data = yf.download(tickers, start=test_start, end=test_end, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        df = data["Close"].copy() if "Close" in data.columns.levels[0] else data.copy()
    else:
        df = data.copy()

    df = df.ffill().dropna()

    logger.info(f"Downloaded {len(df)} days of test data")

    # 3. Calculate Features (SAME as training)
    df["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df["^VIX"].rolling(
        252
    ).std()
    df["tnx_mom"] = df["^TNX"].pct_change(20)
    df["spy_mom"] = df["SPY"].pct_change(60)

    features = df[["vix_z", "tnx_mom", "spy_mom"]].dropna()

    # 4. Transform (NOT fit!) using OOS Scaler
    X_scaled = scaler.transform(features)  # Transform only, no fit!
    regime_labels = kmeans.predict(X_scaled)  # Predict only, no fit!

    logger.info("✅ Features transformed using OOS Scaler (Data Leakage prevented)")

    # 5. Generate Signals
    is_bull_raw = (
        pd.Series(regime_labels, index=features.index) == bull_regime_id
    ).astype(int)

    # A. MAMA Lite (Base): 5-day smooth, threshold 0.5
    mama_base_prob = is_bull_raw.rolling(5).mean()
    mama_base_sig = (
        (mama_base_prob >= 0.5).astype(int).reindex(df.index).fillna(0).astype(int)
    )

    # B. MAMA Opt (AI + Trend Force)
    spy_ma120 = df["SPY"].rolling(120).mean()
    trend_bull = (df["SPY"] > spy_ma120).astype(int)

    mama_opt_sig = pd.Series(index=df.index, dtype=int)
    curr = 1
    for i in range(len(df)):
        ai_bull = mama_base_sig.iloc[i] == 1
        trend_bull_now = trend_bull.iloc[i] == 1

        # Logic: Sell ONLY if (AI=0 AND Trend=0)
        if curr == 1:
            if (not ai_bull) and (not trend_bull_now):
                curr = 0
        else:
            if ai_bull:
                curr = 1

        mama_opt_sig.iloc[i] = curr

    # C. V4 Full Logic (CORRECTED)
    # Bull: 100% SCHG
    # Bear: 30% SCHG + 70% BIL
    # Crisis (Yield Inversion): 100% BIL

    vti = df["SPY"]  # Use SPY as proxy for VTI
    ma185 = vti.rolling(185).mean()
    upper = ma185 * 1.03
    lower = ma185 * 0.97

    # Yield Curve
    yield_spread = df["^TNX"] - df["^IRX"]
    is_inverted = (yield_spread < 0).astype(int)

    v4_position = pd.Series(index=df.index, dtype=float)  # 0.0-1.0 (asset ratio)
    curr_state = "bull"  # bull, bear, crisis

    for i in range(len(df)):
        price = vti.iloc[i]
        inverted = is_inverted.iloc[i] == 1

        # Determine state
        if inverted:
            curr_state = "crisis"
        elif price > upper.iloc[i]:
            curr_state = "bull"
        elif price < lower.iloc[i]:
            curr_state = "bear"
        # else: maintain current state

        # Set position
        if curr_state == "bull":
            v4_position.iloc[i] = 1.0  # 100% SCHG
        elif curr_state == "bear":
            v4_position.iloc[i] = 0.3  # 30% SCHG, 70% BIL
        else:  # crisis
            v4_position.iloc[i] = 0.0  # 100% BIL

    # 6. Backtest All Strategies (with weighted positions for V4)
    results = []

    # For binary strategies
    for name, sig in [
        ("Buy & Hold (SPY)", pd.Series(1, index=df.index)),
        ("MAMA Lite (Base)", mama_base_sig),
        ("MAMA Opt (AI+Trend)", mama_opt_sig),
    ]:
        eq, tr = backtest_signal(sig, df["SCHG"], df["BIL"], cost_rate=0.0025)
        cagr, mdd, sharpe, _ = calculate_stats(eq, tr)
        results.append(
            {"Strategy": name, "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Trades": tr}
        )

    # V4 with weighted position (needs special handling)
    equity_v4 = 1.0
    equity_curve_v4 = [1.0]
    trades_v4 = 0
    prev_pos = v4_position.iloc[0]

    asset_ret = df["SCHG"].pct_change().fillna(0)
    defensive_ret = df["BIL"].pct_change().fillna(0)

    for i in range(1, len(v4_position)):
        pos = v4_position.iloc[i]

        # Calculate weighted return
        ret = pos * asset_ret.iloc[i] + (1 - pos) * defensive_ret.iloc[i]

        # Transaction cost
        if abs(pos - prev_pos) > 0.01:  # Position changed
            equity_v4 = equity_v4 * (1 - 0.0025)
            trades_v4 += 1

        equity_v4 = equity_v4 * (1 + ret)
        equity_curve_v4.append(equity_v4)
        prev_pos = pos

    cagr_v4, mdd_v4, sharpe_v4, _ = calculate_stats(
        np.array(equity_curve_v4), trades_v4
    )
    results.append(
        {
            "Strategy": "V4 (Full Logic)",
            "CAGR": cagr_v4,
            "MDD": mdd_v4,
            "Sharpe": sharpe_v4,
            "Trades": trades_v4,
        }
    )

    # 7. Output
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("🧪 Out-of-Sample Backtest Results (2020-2025)")
    lines.append(f"   Model: {model_data['training_period']} (OOS)")
    lines.append("   Asset: SCHG / Defensive: BIL / Cost: 0.25% RT")
    lines.append("=" * 80)
    lines.append(
        f"{'Strategy':<25} | {'CAGR':<8} | {'MDD':<8} | {'Sharpe':<8} | {'Trades':<8}"
    )
    lines.append("-" * 80)

    for r in results:
        lines.append(
            f"{r['Strategy']:<25} | {r['CAGR']:.2%}   | {r['MDD']:.2%}   | {r['Sharpe']:.2f}     | {r['Trades']:<8}"
        )

    lines.append("=" * 80)

    final_output = "\n".join(lines)
    print(final_output)

    # Save
    with open("d:/gg/oos_backtest_result.txt", "w", encoding="utf-8") as f:
        f.write(final_output)

    logger.info("✅ Results saved to d:/gg/oos_backtest_result.txt")

    return results


if __name__ == "__main__":
    run_oos_backtest()
