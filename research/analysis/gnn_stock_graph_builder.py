# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("StockGraphBuilder")

# Constants
DATA_DIR = r"d:\gg\data\historical"
GNN_DATA_DIR = r"d:\gg\data\gnn"
CORR_WINDOW = 60
CORR_THRESHOLD = 0.5
START_DATE = "2018-01-01"

# Representative Tech Universe for GNN
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AVGO"]


def load_universe_data():
    all_prices = {}
    for t in TICKERS:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            # Find price column
            for col in ["Close", "Adj Close", t, f"^{t}"]:
                if col in df.columns:
                    all_prices[t] = df[col]
                    break

    df_prices = pd.DataFrame(all_prices).ffill().loc[START_DATE:]
    return df_prices


def build_graph_and_features(df_prices):
    """
    Construct Adjacency Matrix and Node Features over time.
    For the prototype, we will build a 'Global Static Graph' or a 'Snapshot'.
    To make it GNN-ready, we often need (Time, Nodes, Features) tensor.
    """
    logger.info(f"Building Graph for {len(TICKERS)} stocks...")

    returns = df_prices.pct_change().dropna()
    correlation_matrix = returns.corr()

    # Adjacency Matrix based on Correlation
    adj = (correlation_matrix.abs() > CORR_THRESHOLD).astype(int)
    # Remove self-loops for GNN if needed, or keep for stability
    # np.fill_diagonal(adj.values, 0)

    # Node Features: 20-day momentum, 20-day volatility
    features = {}
    for t in TICKERS:
        mom = df_prices[t].pct_change(20)
        vol = df_prices[t].pct_change().rolling(20).std()
        features[f"{t}_mom"] = mom
        features[f"{t}_vol"] = vol

    df_features = pd.DataFrame(features).ffill().dropna()

    # Save for Phase 2: GNN Model
    if not os.path.exists(GNN_DATA_DIR):
        os.makedirs(GNN_DATA_DIR)

    adj.to_csv(os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv"))
    df_features.to_csv(os.path.join(GNN_DATA_DIR, "node_features.csv"))

    logger.info(f"Graph Saved. Edges count: {adj.sum().sum() - len(TICKERS)}")
    return adj, df_features


if __name__ == "__main__":
    prices = load_universe_data()
    if not prices.empty:
        build_graph_and_features(prices)
    else:
        logger.error("No data loaded. Check ticker files.")
