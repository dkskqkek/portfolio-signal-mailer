# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("GNNTrainer")

# Constants
GNN_DATA_DIR = r"d:\gg\data\gnn"
DATA_DIR = r"d:\gg\data\historical"
WEIGHT_FILE = os.path.join(GNN_DATA_DIR, "gnn_weights.pth")
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AVGO"]


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = nn.Linear(in_channels, 16)
        self.conv2 = nn.Linear(16, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # Propagation: A_hat * X (Simple smoothing)
        x = torch.matmul(adj, x)
        x = self.relu(self.conv1(x))
        x = torch.matmul(adj, x)
        x = self.conv2(x)
        return x


def prepare_training_data():
    adj_df = pd.read_csv(
        os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv"), index_col=0
    )
    feat_df = pd.read_csv(
        os.path.join(GNN_DATA_DIR, "node_features.csv"), index_col=0, parse_dates=True
    )

    adj = adj_df.values
    adj = adj + np.eye(adj.shape[0])
    rowsum = adj.sum(1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    adj_tensor = torch.FloatTensor(adj_norm)

    all_returns = {}
    for t in TICKERS:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        # Fix: Predict return from T to T+1 (next period)
        all_returns[t] = df["Close"].pct_change().shift(-1)

    label_df = pd.DataFrame(all_returns).reindex(feat_df.index).ffill().dropna()
    feat_df = feat_df.reindex(label_df.index)

    return adj_tensor, feat_df, label_df


def train():
    adj, feat_df, label_df = prepare_training_data()

    split_idx = int(len(feat_df) * 0.8)
    train_feats = feat_df.iloc[:split_idx]
    train_labels = label_df.iloc[:split_idx]
    val_feats = feat_df.iloc[split_idx:]
    val_labels = label_df.iloc[split_idx:]

    num_features = 2
    model = SimpleGCN(in_channels=num_features, out_channels=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    logger.info("Starting GNN Training with Bias Fix...")

    for epoch in range(100):
        model.train()
        train_loss = 0
        indices = np.random.permutation(len(train_feats))[:64]

        for idx in indices:
            day_feat = [
                [train_feats.iloc[idx][f"{t}_mom"], train_feats.iloc[idx][f"{t}_vol"]]
                for t in TICKERS
            ]
            x = torch.FloatTensor(day_feat)
            y = torch.FloatTensor(train_labels.iloc[idx].values).view(-1, 1)

            optimizer.zero_grad()
            out = model(x, adj)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            v_indices = np.random.permutation(len(val_feats))[:32]
            for v_idx in v_indices:
                v_feat = [
                    [
                        val_feats.iloc[v_idx][f"{t}_mom"],
                        val_feats.iloc[v_idx][f"{t}_vol"],
                    ]
                    for t in TICKERS
                ]
                vx = torch.FloatTensor(v_feat)
                vy = torch.FloatTensor(val_labels.iloc[v_idx].values).view(-1, 1)
                vout = model(vx, adj)
                v_loss = criterion(vout, vy)
                val_loss += v_loss.item()

        if (epoch + 1) % 20 == 0:
            logger.info(
                f"Epoch {epoch + 1}/100, Train Loss: {train_loss / 64:.6f}, Val Loss: {val_loss / 32:.6f}"
            )

    if not os.path.exists(GNN_DATA_DIR):
        os.makedirs(GNN_DATA_DIR)
    torch.save(model.state_dict(), WEIGHT_FILE)
    logger.info(f"Bias-free model saved to {WEIGHT_FILE}")


if __name__ == "__main__":
    train()
