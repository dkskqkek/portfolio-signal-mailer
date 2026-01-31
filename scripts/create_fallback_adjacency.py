"""
Fallback Adjacency Matrix Generator (Sector-Based)
Used when correlation data cannot be fetched (e.g., YF Rate Limit).
Logic: Connects tickers within the same GICS Sector.
"""

import pandas as pd
import numpy as np
import yaml
import os

# Config & Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "signal_mailer", "config.yaml")
GNN_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "gnn")
ADJ_FILE = os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv")

# Sector Map for the Top 20 Candidates
SECTOR_MAP = {
    "NVDA": "Info Tech",
    "AAPL": "Info Tech",
    "MSFT": "Info Tech",
    "AVGO": "Info Tech",
    "GOOGL": "Comm Services",
    "META": "Comm Services",
    "NFLX": "Comm Services",
    "AMZN": "Cons Discret",
    "TSLA": "Cons Discret",
    "HD": "Cons Discret",
    "LLY": "Health Care",
    "UNH": "Health Care",
    "JNJ": "Health Care",
    "BRK-B": "Financials",
    "JPM": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "XOM": "Energy",
    "WMT": "Cons Staples",
    "COST": "Cons Staples",
    "PG": "Cons Staples",
}


def load_tickers_from_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("strategy_info", {}).get("gnn_tickers", [])


def create_sector_adjacency():
    print("Generating Sector-Based Adjacency Matrix (Fallback)...")
    tickers = load_tickers_from_config()
    n = len(tickers)

    adj = np.zeros((n, n), dtype=int)

    print(f"Universe: {n} tickers")

    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i == j:
                adj[i][j] = 1  # Self-loop
                continue

            s1 = SECTOR_MAP.get(t1, "Unknown")
            s2 = SECTOR_MAP.get(t2, "Other")

            # Connection Logic: Same Sector = 1
            if s1 == s2 and s1 != "Unknown":
                adj[i][j] = 1

    # Convert to DataFrame
    adj_df = pd.DataFrame(adj, index=tickers, columns=tickers)

    # Save
    os.makedirs(GNN_DATA_DIR, exist_ok=True)
    adj_df.to_csv(ADJ_FILE)
    print(f"Saved sector adjacency matrix to {ADJ_FILE}")

    # Stats
    density = (adj.sum() - n) / (n * (n - 1))
    print(f"Density: {density:.1%}")
    return adj_df


if __name__ == "__main__":
    create_sector_adjacency()
