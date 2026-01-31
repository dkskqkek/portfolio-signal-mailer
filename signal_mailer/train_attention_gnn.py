"""
Attention GNN 가중치 학습 스크립트 (v4.0)
Multi-head Attention + 11개 티커
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from attention_gnn import MultiHeadAttentionGNN

GNN_DATA_DIR = r"d:\gg\data\gnn"
WEIGHT_FILE = os.path.join(GNN_DATA_DIR, "gnn_weights.pth")
ADJ_FILE = os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv")

# v4.0: Expanded tickers
import yaml

# Load config to get dynamic tickers
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

GNN_TICKERS = config.get("strategy_info", {}).get("gnn_tickers", [])
if not GNN_TICKERS:
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


def calculate_features(prices: pd.Series) -> list:
    """Calculate 10 technical indicators for a single ticker."""
    if len(prices) < 252:
        return [0.0] * 10

    try:
        mom_22 = (prices.iloc[-1] / prices.iloc[-22]) - 1
        vol_21 = prices.pct_change().iloc[-21:].std()

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1] / 100

        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        macd_signal = (macd.iloc[-1] - signal.iloc[-1]) / prices.iloc[-1]

        sma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        bb_position = (prices.iloc[-1] - lower_band.iloc[-1]) / (
            upper_band.iloc[-1] - lower_band.iloc[-1] + 1e-10
        )
        bb_position = max(0, min(1, bb_position))

        volume_trend = 0.0
        high_52w = prices.iloc[-252:].max()
        high_ratio = prices.iloc[-1] / high_52w
        mom_5 = (prices.iloc[-1] / prices.iloc[-5]) - 1
        mom_60 = (prices.iloc[-1] / prices.iloc[-60]) - 1
        mom_divergence = mom_5 - mom_60

        features = [
            float(mom_22) if not np.isnan(mom_22) else 0.0,
            float(vol_21) if not np.isnan(vol_21) else 0.0,
            float(rsi) if not np.isnan(rsi) else 0.5,
            float(macd_signal) if not np.isnan(macd_signal) else 0.0,
            float(bb_position),
            float(volume_trend),
            float(high_ratio) if not np.isnan(high_ratio) else 1.0,
            float(mom_5) if not np.isnan(mom_5) else 0.0,
            float(mom_60) if not np.isnan(mom_60) else 0.0,
            float(mom_divergence) if not np.isnan(mom_divergence) else 0.0,
        ]
        return features
    except Exception:
        return [0.0] * 10


def load_adjacency():
    """Load normalized adjacency matrix."""
    adj_df = pd.read_csv(ADJ_FILE, index_col=0)
    A = torch.tensor(adj_df.values, dtype=torch.float32)
    A_hat = A + torch.eye(A.shape[0])
    D = torch.diag(torch.sum(A_hat, dim=1))
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    return torch.mm(torch.mm(D_inv_sqrt, A_hat), D_inv_sqrt)


def create_training_data(start_date="2014-01-01", end_date="2024-01-01"):
    """Create training data with features and forward returns as labels."""
    PRICE_FILE = os.path.join(GNN_DATA_DIR, "price_history.csv")
    if os.path.exists(PRICE_FILE):
        print(f"Loading data from {PRICE_FILE}...")
        data = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
        # Ensure filtered by date range if needed, or just use what we have
        data = data.sort_index()
        # Filter columns to only current GNN_TICKERS (in case file has old tickers)
        # Handle case where file might be missing some new tickers?
        # Assuming update_adjacency runs first, it should be up to date.
        available_tickers = [t for t in GNN_TICKERS if t in data.columns]
        if len(available_tickers) < len(GNN_TICKERS):
            print(
                "Warning: Price file missing some tickers. Fallback to download (risk rate limit)."
            )
            data = yf.download(
                GNN_TICKERS, start=start_date, end=end_date, progress=False
            )["Close"]
        else:
            data = data[available_tickers]
    else:
        print(f"Downloading data from {start_date} to {end_date}...")
        data = yf.download(GNN_TICKERS, start=start_date, end=end_date, progress=False)[
            "Close"
        ]

    data = data.ffill().dropna()

    print("Creating training samples...")
    X_list = []
    y_list = []

    window_size = 252
    forward_days = 22
    dates = data.index[window_size:-forward_days]

    for i, date in enumerate(dates):
        if i % 200 == 0:
            print(f"  Processing {i}/{len(dates)}...")

        idx = data.index.get_loc(date)
        hist_data = data.iloc[idx - window_size : idx + 1]

        node_feats = []
        for ticker in GNN_TICKERS:
            features = calculate_features(hist_data[ticker])
            node_feats.append(features)

        X_list.append(node_feats)

        forward_data = data.iloc[idx : idx + forward_days + 1]
        forward_returns = []
        for ticker in GNN_TICKERS:
            ret = (forward_data[ticker].iloc[-1] / forward_data[ticker].iloc[0]) - 1
            forward_returns.append(ret if not np.isnan(ret) else 0.0)

        ranks = np.argsort(np.argsort(forward_returns))
        scores = ranks / (len(GNN_TICKERS) - 1)
        y_list.append(scores.tolist())

    return X_list, y_list


def train_attention_gnn(X_list, y_list, adj_norm, epochs=50, lr=0.005):
    """Train Attention GNN model."""
    print(f"\nTraining Attention GNN...")
    print(f"  Samples: {len(X_list)}")
    print(f"  Epochs: {epochs}")
    print(f"  Heads: 4")

    model = MultiHeadAttentionGNN(
        in_features=10, hidden_features=16, out_features=1, num_heads=4, dropout=0.1
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for X, y in zip(X_list, y_list):
            x_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

            optimizer.zero_grad()
            output = model(x_tensor, adj_norm)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(X_list)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    return model, losses


if __name__ == "__main__":
    print("=" * 60)
    print("Attention GNN 학습 (v4.0: 4-head, 11 tickers)")
    print("=" * 60)

    adj_norm = load_adjacency()
    print(f"Adjacency shape: {adj_norm.shape}")

    X_list, y_list = create_training_data(
        start_date="2014-01-01", end_date="2024-01-01"
    )

    model, losses = train_attention_gnn(X_list, y_list, adj_norm, epochs=50, lr=0.005)

    torch.save(model.state_dict(), WEIGHT_FILE)
    print(f"\n[OK] Model saved to: {WEIGHT_FILE}")
    print(f"   Final Loss: {losses[-1]:.6f}")

    print("\n" + "=" * 60)
    print("Attention GNN 학습 완료!")
    print("=" * 60)
