"""
Adjacency Matrix 동적 업데이트 (v3.2)
상관계수 기반 분기별 자동 업데이트
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

GNN_DATA_DIR = r"d:\gg\data\gnn"
ADJ_FILE = os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv")
ADJ_BACKUP = os.path.join(GNN_DATA_DIR, "adjacency_matrix_backup.csv")

# v4.0: Expanded tickers with Healthcare (JNJ) and Financials (V)
import yaml

# Load config to get dynamic tickers
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "signal_mailer",
    "config.yaml",
)
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

GNN_TICKERS = config.get("strategy_info", {}).get("gnn_tickers", [])
if not GNN_TICKERS:
    # Fallback if config load fails
    GNN_TICKERS = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "NFLX",
        "AVGO",
        "JNJ",
        "V",
    ]


def calculate_correlation_matrix(lookback_days=252):
    """Calculate correlation matrix from recent data."""
    print(f"Downloading {lookback_days} days of data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 50)

    # Strategy 1: Bulk Download
    try:
        data = yf.download(
            GNN_TICKERS, start=start_date, end=end_date, progress=False, threads=False
        )["Close"]
        if data.empty or len(data) < 10:
            raise ValueError("Bulk download failed (empty)")
    except Exception as e:
        print(
            f"Warning: Bulk download failed ({e}). Switching to iterative download..."
        )
        data_frames = {}
        import time

        for ticker in GNN_TICKERS:
            try:
                print(f"  Fetching {ticker}...", end="", flush=True)
                df = yf.Ticker(ticker).history(start=start_date, end=end_date)
                if not df.empty:
                    data_frames[ticker] = df["Close"]
                    print(" OK")
                else:
                    print(" Empty")
                time.sleep(1.5)  # Anti-rate limit delay
            except Exception as e:
                print(f" Error ({e})")

        if data_frames:
            data = pd.DataFrame(data_frames)
        else:
            raise ValueError("All download attempts failed.")

    data = data.ffill().dropna()

    # Save for other scripts to use (avoid rate limit)
    PRICE_FILE = os.path.join(GNN_DATA_DIR, "price_history.csv")
    data.to_csv(PRICE_FILE)
    print(f"Saved price data to {PRICE_FILE}")

    # Daily returns correlation
    returns = data.pct_change().dropna()
    corr_matrix = returns.corr()

    return corr_matrix


def correlation_to_adjacency(corr_matrix, threshold=0.5, include_self=True):
    """Convert correlation matrix to binary adjacency matrix.

    Args:
        corr_matrix: Correlation matrix
        threshold: Minimum correlation to establish connection
        include_self: Include self-loops (diagonal = 1)

    Returns:
        Binary adjacency matrix
    """
    adj = (corr_matrix.abs() >= threshold).astype(int)

    if include_self:
        np.fill_diagonal(adj.values, 1)
    else:
        np.fill_diagonal(adj.values, 0)

    return adj


def update_adjacency_matrix(threshold=0.5, lookback_days=252):
    """Update adjacency matrix based on current correlations."""
    print("=" * 60)
    print("Adjacency Matrix 동적 업데이트 (v3.2)")
    print("=" * 60)

    # 1. Backup current matrix
    if os.path.exists(ADJ_FILE):
        import shutil

        shutil.copy(ADJ_FILE, ADJ_BACKUP)
    print(f"\n[1] Backup existing matrix: {ADJ_BACKUP}")

    # 2. Calculate current correlations
    print(f"\n[2] Calculating correlations ({lookback_days} days)...")
    corr_matrix = calculate_correlation_matrix(lookback_days)

    # 3. Convert to adjacency
    print(f"\n[3] Converting to Adjacency (threshold: {threshold})...")
    new_adj = correlation_to_adjacency(corr_matrix, threshold)

    # 4. Compare with old matrix (only if same shape)
    if os.path.exists(ADJ_BACKUP):
        old_adj = pd.read_csv(ADJ_BACKUP, index_col=0)
        if old_adj.shape == new_adj.shape:
            changes = (new_adj.values != old_adj.values).sum()
            print(f"\n[4] 변경된 연결 수: {changes}")

            if changes > 0:
                print("\n변경 상세:")
                for i, ticker1 in enumerate(GNN_TICKERS):
                    for j, ticker2 in enumerate(GNN_TICKERS):
                        if i < j and new_adj.iloc[i, j] != old_adj.iloc[i, j]:
                            old_val = old_adj.iloc[i, j]
                            new_val = new_adj.iloc[i, j]
                            action = "추가" if new_val == 1 else "제거"
                            corr_val = corr_matrix.iloc[i, j]
                            print(
                                f"  {ticker1}-{ticker2}: {action} (상관계수: {corr_val:.3f})"
                            )
        else:
            print(
                f"\n[4] 티커 수 변경: {old_adj.shape[0]} -> {new_adj.shape[0]} (비교 생략)"
            )

    # 5. Save new matrix
    new_adj.to_csv(ADJ_FILE)
    print(f"\n[5] Saving new matrix: {ADJ_FILE}")

    # 6. Statistics
    total_connections = (new_adj.sum().sum() - len(GNN_TICKERS)) / 2
    possible_connections = len(GNN_TICKERS) * (len(GNN_TICKERS) - 1) / 2
    density = total_connections / possible_connections

    print(f"\nStatistics:")
    print(f"  연결 수: {int(total_connections)}/{int(possible_connections)}")
    print(f"  밀도: {density:.1%}")

    # Show connection details per ticker
    print(f"\nConnections per Ticker:")
    for ticker in GNN_TICKERS:
        connections = new_adj.loc[ticker].sum() - 1  # Exclude self
        connected_to = [
            t for t in GNN_TICKERS if t != ticker and new_adj.loc[ticker, t] == 1
        ]
        print(f"  {ticker}: {int(connections)}개 ({', '.join(connected_to)})")

    print("\n" + "=" * 60)
    print("Adjacency Matrix 업데이트 완료!")
    print("=" * 60)

    return new_adj, corr_matrix


if __name__ == "__main__":
    new_adj, corr_matrix = update_adjacency_matrix(threshold=0.5, lookback_days=252)

    # Save correlation matrix for reference
    corr_file = os.path.join(GNN_DATA_DIR, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_file)
    print(f"\n상관계수 행렬 저장: {corr_file}")
