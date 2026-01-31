# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class MAMATradingEnv(gym.Env):
    """
    Custom Environment for MAMA Lite RL Policy Optimization.
    State: [Regime_ID, Stock1_GNN, Stock2_GNN, Stock3_GNN, Current_Allocation_Type]
    Action: [0: High Alpha (Stocks), 1: Defensive (Bonds/Cash)]
    """

    def __init__(self, price_df, regime_df, gnn_model, adj_tensor, feat_df):
        super(MAMATradingEnv, self).__init__()

        self.price_df = price_df
        self.regime_df = regime_df
        self.gnn_model = gnn_model
        self.adj_tensor = adj_tensor
        self.feat_df = feat_df
        self.dates = regime_df.index.tolist()
        self.tickers = [
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

        # Track previous weights for transaction cost calculation
        self.prev_weights = np.array([0.0, 1.0])  # [Alpha_Weight, Defensive_Weight]

        # Action: 0 (Stay in Stocks), 1 (Stay in Defensive)
        self.action_space = spaces.Discrete(2)

        # Observation: [Regime, GNN1, GNN2, GNN3]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.current_step = 0
        self.history = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_reward = 0
        self.nav = 1.0
        self.history = []
        self.prev_weights = np.array([0.0, 1.0])

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        import torch

        date = self.dates[self.current_step]
        regime = float(self.regime_df.loc[date, "regime"])

        # DYNAMIC GNN Inference
        day_feat = []
        for t in self.tickers:
            day_feat.append(
                [self.feat_df.loc[date, f"{t}_mom"], self.feat_df.loc[date, f"{t}_vol"]]
            )

        x = torch.FloatTensor(day_feat)
        self.gnn_model.eval()
        with torch.no_grad():
            scores = self.gnn_model(x, self.adj_tensor).squeeze().numpy()

        # Current Top 3 dynamic scores
        self.current_gnn_scores = pd.DataFrame(
            {"Ticker": self.tickers, "GNN_Score": scores}
        )
        top_scores = np.sort(scores)[-3:][::-1]  # Top 3 descending

        obs = np.array([regime, *top_scores], dtype=np.float32)
        return obs

    def step(self, action):
        date = self.dates[self.current_step]

        # Current Action Weights (for transaction cost)
        curr_weights = np.array([1.0, 0.0]) if action == 0 else np.array([0.0, 1.0])

        # Calculate Return
        if action == 0:
            # High Alpha Mode: Use top stocks from DYNAMIC GNN scores
            top_tickers = self.current_gnn_scores.nlargest(3, "GNN_Score")[
                "Ticker"
            ].tolist()
            # Mean return of top GNN stocks
            stock_rets = [
                self.price_df.loc[date, f"{t}_ret"]
                for t in top_tickers
                if f"{t}_ret" in self.price_df.columns
            ]
            day_ret = (
                np.mean(stock_rets)
                if stock_rets
                else self.price_df.loc[date, "QQQ_ret"]
            )
        else:
            # Defensive Mode
            day_ret = (
                self.price_df.loc[date, "BIL_ret"] * 0.5
                + self.price_df.loc[date, "TLT_ret"] * 0.5
            )

        # Transaction Cost (0.1% per 100% turnover)
        turnover = np.abs(curr_weights - self.prev_weights).sum()
        tc = turnover * 0.001

        # Risk-Adjusted Reward (Sharpe-style)
        self.nav *= 1 + day_ret - tc
        self.history.append(day_ret)
        self.prev_weights = curr_weights

        # Volatility Penalty
        vol_penalty = 0.0
        if len(self.history) >= 20:
            vol_penalty = np.std(self.history[-20:]) * 0.5

        reward = day_ret - tc - vol_penalty
        self.total_reward += reward

        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        truncated = False

        next_obs = self._get_obs() if not done else np.zeros((4,), dtype=np.float32)

        return next_obs, reward, done, truncated, {"nav": self.nav}


def prepare_env_data():
    import os
    import yfinance as yf

    DATA_DIR = r"d:\gg\data\historical"
    START_DATE = "2018-01-01"
    # Unified list including Asset allocation + GNN tech universe
    TICKERS = [
        "SPY",
        "QQQ",
        "BIL",
        "TLT",
        "TNX",
        "VIX",
        "UUP",
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

    all_prices = {}

    def extract_close_local(df, ticker):
        cols = [c for c in df.columns if c != "Date"]
        if "Close" in cols:
            return df["Close"]
        for f in [ticker, f"^{ticker}", ticker.replace("^", "")]:
            if f in cols:
                return df[f]
        return None

    for t in TICKERS:
        path = os.path.join(DATA_DIR, f"{t.replace('^', '')}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            all_prices[t] = extract_close_local(df, t)
        else:
            # Fallback to yf
            try:
                data = yf.download(
                    t if "VIX" not in t else "^VIX", start=START_DATE, progress=False
                )
                if not data.empty:
                    # Handle MultiIndex
                    if isinstance(data.columns, pd.MultiIndex):
                        all_prices[t] = data["Close"].iloc[:, 0]
                    else:
                        all_prices[t] = data["Close"]
            except:
                pass

    df_prices = pd.DataFrame(all_prices).ffill().loc[START_DATE:]

    # Check if VIX and other keys exist
    for t in TICKERS:
        if t not in df_prices.columns:
            # Create dummy if missing just to avoid crash, but log it
            print(f"Warning: {t} not found in data. Using dummy.")
            df_prices[t] = 0.0

    df_rets = df_prices.pct_change().dropna()
    df_rets.columns = [f"{c}_ret" for c in df_rets.columns]

    # Load Regime
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    macro = pd.DataFrame(index=df_prices.index)
    if "VIX" in df_prices.columns:
        macro["vix_z"] = (
            df_prices["VIX"] - df_prices["VIX"].rolling(252).mean()
        ) / df_prices["VIX"].rolling(252).std()
    macro = macro.ffill().dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(macro)
    model = KMeans(n_clusters=4, n_init=10, random_state=42)
    regime_labels = model.fit_predict(X)
    df_regime = pd.DataFrame(regime_labels, index=macro.index, columns=["regime"])

    # Load Node Features for dynamic inference inside env
    GNN_DATA_DIR = r"d:\gg\data\gnn"
    feat_df = pd.read_csv(
        os.path.join(GNN_DATA_DIR, "node_features.csv"), index_col=0, parse_dates=True
    )

    # [Assertion] Synchronize Indices
    shared_idx = df_rets.index.intersection(df_regime.index).intersection(feat_df.index)
    assert len(shared_idx) > 0, (
        "Error: SRL Regime index and Returns index do not overlap!"
    )

    df_rets = df_rets.reindex(shared_idx)
    df_regime = df_regime.reindex(shared_idx)
    feat_df = feat_df.reindex(shared_idx)

    return df_rets, df_regime, feat_df
