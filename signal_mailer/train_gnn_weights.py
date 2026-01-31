"""
GNN 가중치 재학습 스크립트 (v3.2)
10개 기술지표 기반 GNN 최적화
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

GNN_DATA_DIR = r"d:\gg\data\gnn"
WEIGHT_FILE = os.path.join(GNN_DATA_DIR, "gnn_weights.pth")
ADJ_FILE = os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv")

GNN_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AVGO"]


class SimpleGCN(nn.Module):
    """v3.2: 10 input features"""

    def __init__(self, in_features=10, hidden_features=16, out_features=1):
        super(SimpleGCN, self).__init__()
        self.conv1 = nn.Linear(in_features, hidden_features)
        self.conv2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, adj):
        x = torch.mm(adj, x)
        x = F.relu(self.conv1(x))
        x = torch.mm(adj, x)
        x = self.conv2(x)
        return x


def calculate_features(prices: pd.Series) -> list:
    """Calculate 10 technical indicators for a single ticker."""
    if len(prices) < 252:
        return [0.0] * 10

    try:
        # 1. Momentum (22-day)
        mom_22 = (prices.iloc[-1] / prices.iloc[-22]) - 1

        # 2. Volatility (21-day)
        vol_21 = prices.pct_change().iloc[-21:].std()

        # 3. RSI (14-day)
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1] / 100

        # 4. MACD Signal
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        macd_signal = (macd.iloc[-1] - signal.iloc[-1]) / prices.iloc[-1]

        # 5. Bollinger Band Position
        sma_20 = prices.rolling(20).mean()
        std_20 = prices.rolling(20).std()
        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20
        bb_position = (prices.iloc[-1] - lower_band.iloc[-1]) / (
            upper_band.iloc[-1] - lower_band.iloc[-1] + 1e-10
        )
        bb_position = max(0, min(1, bb_position))

        # 6. Volume Trend (placeholder)
        volume_trend = 0.0

        # 7. 52-Week High Ratio
        high_52w = prices.iloc[-252:].max()
        high_ratio = prices.iloc[-1] / high_52w

        # 8. Short Momentum (5-day)
        mom_5 = (prices.iloc[-1] / prices.iloc[-5]) - 1

        # 9. Long Momentum (60-day)
        mom_60 = (prices.iloc[-1] / prices.iloc[-60]) - 1

        # 10. Momentum Divergence
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
    print(f"📥 Downloading data from {start_date} to {end_date}...")
    data = yf.download(GNN_TICKERS, start=start_date, end=end_date, progress=False)[
        "Close"
    ]
    data = data.ffill().dropna()

    print(f"📊 Creating training samples...")
    X_list = []  # Features
    y_list = []  # Labels (forward returns ranking)

    # Rolling window training data
    window_size = 252  # 1 year lookback
    forward_days = 22  # 1 month forward return

    dates = data.index[window_size:-forward_days]

    for i, date in enumerate(dates):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(dates)}...")

        # Get historical data up to this date
        idx = data.index.get_loc(date)
        hist_data = data.iloc[idx - window_size : idx + 1]

        # Calculate features for each ticker
        node_feats = []
        for ticker in GNN_TICKERS:
            features = calculate_features(hist_data[ticker])
            node_feats.append(features)

        X_list.append(node_feats)

        # Labels: Forward return ranking (higher return = higher score)
        forward_data = data.iloc[idx : idx + forward_days + 1]
        forward_returns = []
        for ticker in GNN_TICKERS:
            ret = (forward_data[ticker].iloc[-1] / forward_data[ticker].iloc[0]) - 1
            forward_returns.append(ret if not np.isnan(ret) else 0.0)

        # Convert to ranking scores (0 to 1)
        ranks = np.argsort(np.argsort(forward_returns))
        scores = ranks / (len(GNN_TICKERS) - 1)
        y_list.append(scores.tolist())

    return X_list, y_list


def train_gnn(X_list, y_list, adj_norm, epochs=100, lr=0.01):
    """Train GNN model."""
    print(f"\n🚀 Training GNN model...")
    print(f"  Samples: {len(X_list)}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")

    model = SimpleGCN(in_features=10, hidden_features=16, out_features=1)
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

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

    return model, losses


if __name__ == "__main__":
    print("=" * 60)
    print("GNN 가중치 재학습 (v3.2: 10 features)")
    print("=" * 60)

    # 1. Load adjacency matrix
    adj_norm = load_adjacency()

    # 2. Create training data (5 years)
    X_list, y_list = create_training_data(
        start_date="2014-01-01", end_date="2024-01-01"
    )

    # 3. Train model
    model, losses = train_gnn(X_list, y_list, adj_norm, epochs=100, lr=0.01)

    # 4. Save model
    torch.save(model.state_dict(), WEIGHT_FILE)
    print(f"\n✅ Model saved to: {WEIGHT_FILE}")

    # 5. Validate
    model.eval()
    test_input = torch.randn(9, 10)
    with torch.no_grad():
        output = model(test_input, adj_norm)
    print(f"\n📊 Validation:")
    print(f"  Final Loss: {losses[-1]:.6f}")
    print(f"  Output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("GNN 재학습 완료!")
    print("=" * 60)
