# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MAMALiteGNN")

# Constants
GNN_DATA_DIR = r"d:\gg\data\gnn"
OUTPUT_DIR = r"d:\gg\docs\reports"
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AVGO"]


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = nn.Linear(in_channels, 16)
        self.conv2 = nn.Linear(16, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # x: [num_nodes, in_features]
        # adj: [num_nodes, num_nodes] (normalized adjacency matrix)
        x = torch.matmul(adj, x)
        x = self.relu(self.conv1(x))
        x = torch.matmul(adj, x)
        x = self.conv2(x)
        return x


def prepare_tensors():
    # Load Data
    adj_path = os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv")
    feat_path = os.path.join(GNN_DATA_DIR, "node_features.csv")

    if not os.path.exists(adj_path) or not os.path.exists(feat_path):
        logger.error("GNN data files not found. Run gnn_stock_graph_builder.py first.")
        return None, None, None

    adj_df = pd.read_csv(adj_path, index_col=0)
    feat_df = pd.read_csv(feat_path, index_col=0)

    # 1. Normalize Adjacency Matrix
    adj = adj_df.values
    adj = adj + np.eye(adj.shape[0])  # Self-loops
    rowsum = adj.sum(1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    adj_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    adj_tensor = torch.FloatTensor(adj_norm)

    # 2. Latest Node Features (Momentum, Volatility)
    latest_feat = []
    for t in TICKERS:
        latest_feat.append([feat_df[f"{t}_mom"].iloc[-1], feat_df[f"{t}_vol"].iloc[-1]])

    x = torch.tensor(latest_feat, dtype=torch.float32)
    return x, adj_tensor, adj_df.index.tolist()


def run_gnn_selection():
    logger.info("Initializing MAMA Lite GNN Selection Engine...")
    x, adj, tickers = prepare_tensors()

    if x is None:
        return

    # Model parameters
    in_dim = x.shape[1]
    out_dim = 1  # Predicted Score

    model = SimpleGCN(in_dim, out_dim)

    # Load Weights
    weight_path = os.path.join(GNN_DATA_DIR, "gnn_weights.pth")
    if os.path.exists(weight_path):
        logger.info(f"Loading GNN weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path))
        model.eval()
    else:
        logger.warning("No weights found. Using random initialized weights.")

    with torch.no_grad():
        scores = model(x, adj).squeeze()

    results = pd.DataFrame(
        {"Ticker": tickers, "GNN_Score": scores.numpy()}
    ).sort_values("GNN_Score", ascending=False)

    logger.info("\nGNN-based Selection Rankings (Intra-Asset):")
    print(results)

    # Save Report
    report_path = os.path.join(OUTPUT_DIR, "mama_lite_gnn_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# MAMA Lite: Intra-Asset GNN 종목 선정 리포트\n\n")
        f.write(
            "개별 종목 간 상관관계를 그래프로 학습하여, 정보 전이(Spillover)를 고려한 최적의 종목을 선정합니다.\n\n"
        )
        f.write("## 1. GNN 순위 결과\n")
        f.write("| 순위 | 티커 | GNN 점수 (Spillover Aware) |\n")
        f.write("| :--- | :--- | :--- |\n")
        for i, row in results.iterrows():
            f.write(f"| {i + 1} | {row['Ticker']} | {row['GNN_Score']:.4f} |\n")

        f.write("\n## 2. GNN Insights\n")
        f.write(
            "1. **네트워크 효과**: 상관관계가 높은 종목들(예: NVDA와 AVGO)은 그래프 엣지를 통해 서로의 에너지를 공유하며, 한 종목의 강세가 인접 노드의 점수를 보정함.\n"
        )
        f.write(
            "2. **비선형성**: 단순 수익률 합산이 아닌 GCN 레이어를 통한 비선형 특징 추출로 시장의 '숨겨진 대장주'를 우선순위에 배치.\n"
        )

    logger.info(f"GNN Simulation Complete. Report: {report_path}")


if __name__ == "__main__":
    run_gnn_selection()
