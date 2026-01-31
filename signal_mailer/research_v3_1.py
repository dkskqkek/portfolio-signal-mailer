import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def prepare_v3_1_assets():
    # 1. Fetch Macro Data for KMeans freezing (20 years)
    macro_tickers = ["^VIX", "^TNX", "SPY"]
    print(f"Fetching macro data: {macro_tickers}")
    data = yf.download(macro_tickers, start="2004-01-01", progress=False)["Close"]
    data = data.ffill().dropna()

    # Calculate SRL Features
    df = data.copy()
    df["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df["^VIX"].rolling(
        252
    ).std()
    df["tnx_mom"] = df["^TNX"].pct_change(20)
    df["spy_mom"] = df["SPY"].pct_change(60)
    features = df[["vix_z", "tnx_mom", "spy_mom"]].dropna()

    # Train KMeans (deterministic for freezing)
    scaler = StandardScaler()
    X_srl = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
    kmeans.fit(X_srl)

    # Save Centroids and Scaler Params
    save_dir = r"d:\gg\data\srl"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save Centroids
    centroids_path = os.path.join(save_dir, "kmeans_centroids.json")
    with open(centroids_path, "w") as f:
        json.dump(kmeans.cluster_centers_.tolist(), f)

    # Save Scaler Params
    scaler_path = os.path.join(save_dir, "scaler_params.json")
    with open(scaler_path, "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)

    print(f"KMeans Centroids & Scaler saved to {save_dir}")

    # 2. Analyze Adjacency Matrix
    gnn_tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "NFLX",
        "AVGO",
    ]
    print(f"Fetching GNN tickers for correlation analysis: {gnn_tickers}")
    gnn_data = yf.download(gnn_tickers, start="2022-01-01", progress=False)["Close"]
    corr = gnn_data.pct_change().corr()

    corr_path = os.path.join(r"d:\gg\data\gnn", "correlation_matrix.csv")
    corr.to_csv(corr_path)
    print(f"Correlation matrix saved to {corr_path}")


if __name__ == "__main__":
    prepare_v3_1_assets()
