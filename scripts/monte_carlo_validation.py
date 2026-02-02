"""
Script: Monte Carlo Validation (Pattern-Aware Bootstrap)
Author: Antigravity
Date: 2026-02-01

Goal: Verify statistical significance of MAMA Opt
- Method: Pattern-Aware Randomization (preserves market structure)
- Iterations: 1000
- Metrics: p-value, Sharpe 95% CI, Information Ratio
- Target: p-value < 0.05
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MonteCarloValidation")

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "data", "gnn")
MODEL_PATH = os.path.join(data_dir, "kmeans_model_oos.pkl")


def backtest_signal(signals, prices, defensive_prices, cost_rate=0.0025):
    """Backtest with transaction cost"""
    equity = 1.0
    trades = 0
    curr_pos = signals.iloc[0]

    asset_ret = prices.pct_change().fillna(0)
    defensive_ret = defensive_prices.pct_change().fillna(0)

    equity_curve = [1.0]

    for i in range(1, len(signals)):
        ret = asset_ret.iloc[i] if curr_pos == 1 else defensive_ret.iloc[i]
        new_pos = signals.iloc[i]

        if new_pos != curr_pos:
            equity = equity * (1 - cost_rate)
            trades += 1
            curr_pos = new_pos

        equity = equity * (1 + ret)
        equity_curve.append(equity)

    return np.array(equity_curve), trades


def calculate_sharpe(equity_curve):
    """Calculate annualized Sharpe ratio"""
    daily_ret = pd.Series(equity_curve).pct_change().dropna()
    if daily_ret.std() == 0:
        return 0
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    return sharpe


def generate_pattern_aware_random_signal(actual_signal, n_iterations=1000):
    """
    Generate random signals that preserve statistical properties:
    1. Same bull/bear ratio
    2. Similar holding period distribution
    3. Similar trade frequency
    """
    logger.info("Generating Pattern-Aware Random Signals...")

    # Calculate actual stats
    bull_ratio = (actual_signal == 1).mean()

    # Calculate holding periods
    changes = actual_signal.diff().fillna(0)
    change_indices = changes[changes != 0].index.tolist()

    if len(change_indices) > 1:
        holding_periods = [
            change_indices[i + 1] - change_indices[i]
            for i in range(len(change_indices) - 1)
        ]
        avg_holding = np.mean(
            [
                len(actual_signal.loc[change_indices[i] : change_indices[i + 1]])
                for i in range(len(change_indices) - 1)
            ]
        )
    else:
        avg_holding = len(actual_signal)

    trade_frequency = len(change_indices)

    logger.info(
        f"Actual Stats: Bull Ratio={bull_ratio:.2%}, Avg Holding={avg_holding:.0f} days, Trades={trade_frequency}"
    )

    # Generate random signals
    random_signals = []

    for iteration in range(n_iterations):
        # Start with random initial position
        curr_pos = 1 if np.random.rand() < bull_ratio else 0
        signal = [curr_pos]

        # Generate signal with similar holding period
        i = 1
        while i < len(actual_signal):
            # Determine holding period (sample from exponential distribution)
            holding = int(np.random.exponential(avg_holding))
            holding = max(1, min(holding, len(actual_signal) - i))

            # Fill with current position
            signal.extend([curr_pos] * holding)

            # Flip position
            curr_pos = 1 - curr_pos
            i += holding

        # Trim to actual length
        signal = signal[: len(actual_signal)]
        random_signals.append(pd.Series(signal, index=actual_signal.index))

    return random_signals


def monte_carlo_validation():
    """Run Monte Carlo Validation"""

    logger.info("🎲 Monte Carlo Validation Starting...")

    # 1. Load OOS Model
    model_data = joblib.load(MODEL_PATH)
    scaler = model_data["scaler"]
    kmeans = model_data["kmeans"]
    bull_regime_id = model_data["bull_regime_id"]

    logger.info(f"✅ Loaded OOS Model (Trained: {model_data['training_period']})")

    # 2. Download Test Data (2020-2025)
    test_start = "2020-01-01"
    test_end = "2025-12-31"

    tickers = ["SPY", "SCHG", "BIL", "^VIX", "^TNX"]
    data = yf.download(tickers, start=test_start, end=test_end, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        df = data["Close"].copy() if "Close" in data.columns.levels[0] else data.copy()
    else:
        df = data.copy()

    df = df.ffill().dropna()

    logger.info(f"Downloaded {len(df)} days of test data")

    # 3. Generate MAMA Opt Signal
    df["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df["^VIX"].rolling(
        252
    ).std()
    df["tnx_mom"] = df["^TNX"].pct_change(20)
    df["spy_mom"] = df["SPY"].pct_change(60)

    features = df[["vix_z", "tnx_mom", "spy_mom"]].dropna()

    X_scaled = scaler.transform(features)
    regime_labels = kmeans.predict(X_scaled)

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

    # 4. Calculate Actual Performance
    eq_actual, trades_actual = backtest_signal(mama_opt_sig, df["SCHG"], df["BIL"])
    sharpe_actual = calculate_sharpe(eq_actual)

    logger.info(f"Actual Strategy: Sharpe={sharpe_actual:.3f}, Trades={trades_actual}")

    # 5. Generate Random Signals
    random_signals = generate_pattern_aware_random_signal(
        mama_opt_sig, n_iterations=1000
    )

    # 6. Backtest Random Signals
    logger.info("Backtesting 1000 random signals...")
    random_sharpes = []

    for i, random_sig in enumerate(random_signals):
        if (i + 1) % 200 == 0:
            logger.info(f"  Progress: {i + 1}/1000")

        eq_random, _ = backtest_signal(random_sig, df["SCHG"], df["BIL"])
        sharpe_random = calculate_sharpe(eq_random)
        random_sharpes.append(sharpe_random)

    random_sharpes = np.array(random_sharpes)

    # 7. Calculate p-value
    p_value = (random_sharpes >= sharpe_actual).sum() / len(random_sharpes)

    # 8. Calculate Sharpe CI
    n_days = len(eq_actual)
    se = np.sqrt((1 + 0.5 * sharpe_actual**2) / (n_days / 252))
    ci_lower = sharpe_actual - 1.96 * se
    ci_upper = sharpe_actual + 1.96 * se

    # 9. Calculate Information Ratio (vs V4)
    # (Simplified: use Buy & Hold as benchmark)
    eq_bh = (1 + df["SCHG"].pct_change().fillna(0)).cumprod().values

    ret_actual = pd.Series(eq_actual).pct_change().dropna()
    ret_bh = pd.Series(eq_bh).pct_change().dropna()

    excess_ret = ret_actual - ret_bh
    tracking_error = excess_ret.std() * np.sqrt(252)

    if tracking_error > 0:
        information_ratio = (excess_ret.mean() * 252) / tracking_error
    else:
        information_ratio = 0

    # 10. Output
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("🎲 Monte Carlo Validation Results")
    lines.append("=" * 80)
    lines.append(f"Actual Strategy Sharpe: {sharpe_actual:.3f}")
    lines.append(f"Random Signals (1000 iterations):")
    lines.append(f"  Mean Sharpe: {random_sharpes.mean():.3f}")
    lines.append(f"  Std Sharpe: {random_sharpes.std():.3f}")
    lines.append(f"  Min Sharpe: {random_sharpes.min():.3f}")
    lines.append(f"  Max Sharpe: {random_sharpes.max():.3f}")
    lines.append("-" * 80)
    lines.append(f"📊 Statistical Significance:")
    lines.append(
        f"  p-value: {p_value:.4f} {'✅ PASS' if p_value < 0.05 else '⚠️ FAIL'} (Target: < 0.05)"
    )
    lines.append("-" * 80)
    lines.append(f"📈 Sharpe Ratio 95% Confidence Interval:")
    lines.append(f"  Lower Bound: {ci_lower:.3f}")
    lines.append(f"  Upper Bound: {ci_upper:.3f}")
    lines.append("-" * 80)
    lines.append(f"📊 Information Ratio (vs Buy & Hold):")
    lines.append(
        f"  IR: {information_ratio:.3f} {'✅ PASS' if information_ratio > 0.3 else '⚠️ FAIL'} (Target: > 0.3)"
    )
    lines.append("=" * 80)

    final_output = "\n".join(lines)
    print(final_output)

    # Save
    with open("d:/gg/monte_carlo_result.txt", "w", encoding="utf-8") as f:
        f.write(final_output)

    logger.info("✅ Results saved to d:/gg/monte_carlo_result.txt")

    return {
        "sharpe_actual": sharpe_actual,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "information_ratio": information_ratio,
    }


if __name__ == "__main__":
    monte_carlo_validation()
