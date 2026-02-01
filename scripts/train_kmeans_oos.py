"""
Script: Train KMeans Out-of-Sample (OOS)
Author: Antigravity
Date: 2026-02-01

Goal: Eliminate Look-ahead Bias
- Train Period: 2004-01-01 ~ 2019-12-31
- Test Period: 2020-01-01 ~ 2025-12-31 (Completely Unseen)
- Data Leakage Prevention: StandardScaler also trained ONLY on 2004-2019
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainKMeansOOS")

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, "data", "gnn")
os.makedirs(data_dir, exist_ok=True)

MODEL_PATH = os.path.join(data_dir, "kmeans_model_oos.pkl")


def train_kmeans_oos():
    """Train KMeans on 2004-2019 ONLY"""

    logger.info("🚀 Training KMeans Out-of-Sample Model...")

    # 1. Download Data (2004-2019 ONLY)
    train_start = "2004-01-01"
    train_end = "2019-12-31"

    logger.info(f"Training Period: {train_start} to {train_end}")

    tickers = ["^VIX", "^TNX", "SPY"]
    data = yf.download(tickers, start=train_start, end=train_end, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        df = data["Close"].copy() if "Close" in data.columns.levels[0] else data.copy()
    else:
        df = data.copy()

    df = df.ffill().dropna()

    logger.info(f"Downloaded {len(df)} days of data")

    # 2. Calculate Features
    df["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df["^VIX"].rolling(
        252
    ).std()
    df["tnx_mom"] = df["^TNX"].pct_change(20)
    df["spy_mom"] = df["SPY"].pct_change(60)

    features = df[["vix_z", "tnx_mom", "spy_mom"]].dropna()

    logger.info(f"Features prepared: {len(features)} samples")

    # 3. Train StandardScaler (CRITICAL: Train Period Only)
    scaler = StandardScaler()
    X_scaled = scaler.fit(features)  # fit on 2004-2019 ONLY
    X_scaled = scaler.transform(features)

    logger.info(
        "✅ StandardScaler trained on Train Period ONLY (Data Leakage prevented)"
    )

    # 4. Train KMeans
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    regime_labels = kmeans.fit_predict(X_scaled)

    logger.info(f"KMeans trained: {n_clusters} clusters")

    # 5. Identify Bull/Bear Regimes
    features_with_ret = features.copy()
    features_with_ret["spy_ret"] = df["SPY"].pct_change().reindex(features.index)
    features_with_ret["regime"] = regime_labels

    regime_spy_ret = features_with_ret.groupby("regime")["spy_ret"].mean()
    bull_regime_id = int(regime_spy_ret.idxmax())
    bear_regime_id = int(regime_spy_ret.idxmin())

    logger.info(
        f"Bull Regime: Cluster {bull_regime_id} (Avg SPY Return: {regime_spy_ret[bull_regime_id]:.4f})"
    )
    logger.info(
        f"Bear Regime: Cluster {bear_regime_id} (Avg SPY Return: {regime_spy_ret[bear_regime_id]:.4f})"
    )

    # 6. Save Model
    model_data = {
        "scaler": scaler,
        "kmeans": kmeans,
        "bull_regime_id": bull_regime_id,
        "bear_regime_id": bear_regime_id,
        "training_period": f"{train_start} to {train_end}",
        "n_clusters": n_clusters,
        "feature_names": ["vix_z", "tnx_mom", "spy_mom"],
    }

    joblib.dump(model_data, MODEL_PATH)
    logger.info(f"✅ Model saved to {MODEL_PATH}")

    # 7. Summary
    print("\n" + "=" * 80)
    print("✅ Out-of-Sample KMeans Model Training Complete")
    print("=" * 80)
    print(f"Training Period: {train_start} to {train_end}")
    print(f"Samples: {len(features)}")
    print(f"Bull Regime: Cluster {bull_regime_id}")
    print(f"Bear Regime: Cluster {bear_regime_id}")
    print(f"Saved: {MODEL_PATH}")
    print("=" * 80)

    return model_data


if __name__ == "__main__":
    train_kmeans_oos()
