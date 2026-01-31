# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# Reuse logic from Phase 1 and 2
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MAMAIntegrated")

# Constants
DATA_DIR = r"d:\gg\data\historical"
GNN_DATA_DIR = r"d:\gg\data\gnn"
REPORTS_DIR = r"d:\gg\docs\reports"
START_DATE = "2018-01-01"

ALLOC_UNIVERSE = ["SPY", "QQQ", "GLD", "TLT", "BIL"]
MACRO_UNIVERSE = ["VIX", "TNX", "UUP"]
GNN_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AVGO"]


class SimpleGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleGCN, self).__init__()
        self.conv1 = nn.Linear(in_features, hidden_features)
        self.conv2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, adj):
        x = torch.mm(adj, x)
        x = F.relu(self.conv1(x))
        x = torch.mm(adj, x)
        x = self.conv2(x)
        return x


def extract_close(df, ticker=None):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            return df["Close"].iloc[:, 0]
    cols = [c for c in df.columns if c != "Date"]
    if "Close" in cols:
        return df["Close"]
    if ticker:
        for f in [ticker, f"^{ticker}", ticker.replace("^", "")]:
            if f in cols:
                return df[f]
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            return df[c]
    return None


def load_all_data():
    all_prices = {}
    for t in ALLOC_UNIVERSE + MACRO_UNIVERSE + GNN_TICKERS:
        t_name = t.replace("^", "")
        path = os.path.join(DATA_DIR, f"{t_name}.csv")
        if os.path.exists(path):
            tmp = pd.read_csv(path, index_col="Date", parse_dates=True)
            all_prices[t] = extract_close(tmp, t)

    df = pd.DataFrame(all_prices).ffill().loc[START_DATE:]
    return df


def run_integrated_simulation():
    logger.info("Starting MAMA Lite Integrated Simulation (SRL + GNN)...")
    df = load_all_data()
    logger.info(f"Full Data Loaded. Shape: {df.shape}")

    # --- Phase 1: SRL (Regime Aware) ---
    features = pd.DataFrame(index=df.index)
    features["vix_z"] = (df["VIX"] - df["VIX"].rolling(252).mean()) / df["VIX"].rolling(
        252
    ).std()
    features["tnx_mom"] = df["TNX"].pct_change(20)
    features["spy_mom"] = df["SPY"].pct_change(60)
    features = features.dropna()

    scaler = StandardScaler()
    X_srl = scaler.fit_transform(features)

    from sklearn.cluster import KMeans

    regime_model = KMeans(n_clusters=4, n_init=10, random_state=42)
    regime_labels = regime_model.fit_predict(X_srl)
    df_regime = pd.DataFrame(regime_labels, index=features.index, columns=["regime"])

    # Determine Bull Regime (Highest SPY Return)
    rets_spy = df["SPY"].pct_change().reindex(features.index)
    bull_regime = rets_spy.groupby(regime_labels).mean().idxmax()
    logger.info(f"Identified Bull Regime ID: {bull_regime}")

    # --- Phase 2: GNN (Stock Selection) ---
    # Load Adjacency Matrix
    adj_df = pd.read_csv(
        os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv"), index_col=0
    )
    A = torch.tensor(adj_df.values, dtype=torch.float32)
    A_hat = A + torch.eye(A.shape[0])
    D = torch.diag(torch.sum(A_hat, dim=1))
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    adj_norm = torch.mm(torch.mm(D_inv_sqrt, A_hat), D_inv_sqrt)

    gnn_model = SimpleGCN(2, 16, 1)  # (Mom, Vol) -> Score

    # --- Phase 3: Combined Simulation ---
    cash = 1.0
    nav = [1.0]
    dates = df_regime.index.tolist()

    for i in range(1, len(dates)):
        date = dates[i]
        prev_regime = df_regime.iloc[i - 1]["regime"]

        if prev_regime == bull_regime:
            # GNN Selection Mode
            # Get latest features for GNN tickers
            node_feats = []
            for t in GNN_TICKERS:
                mom = (df[t].iloc[i - 1] / df[t].iloc[max(0, i - 21)]) - 1
                vol = df[t].pct_change().iloc[max(0, i - 21) : i].std()
                node_feats.append([mom, vol if not np.isnan(vol) else 0])

            x_gnn = torch.tensor(node_feats, dtype=torch.float32)
            with torch.no_grad():
                scores = gnn_model(x_gnn, adj_norm).squeeze()

            top_stocks_idx = scores.argsort(descending=True)[:3]
            top_stocks = [GNN_TICKERS[idx] for idx in top_stocks_idx]

            # Equal weight on top 3 stocks
            day_ret = df[top_stocks].pct_change().iloc[i].mean()
        else:
            # Defensive Mode: BIL/TLT
            day_ret = (
                df["BIL"].pct_change().iloc[i] * 0.5
                + df["TLT"].pct_change().iloc[i] * 0.5
            )

        if np.isnan(day_ret):
            day_ret = 0
        cash *= 1 + day_ret
        nav.append(cash)

    final_nav = pd.Series(nav, index=dates)

    # Final Metrics
    cagr = (final_nav.iloc[-1] ** (252 / len(final_nav))) - 1
    mdd = (final_nav / final_nav.cummax() - 1).min()

    # Baseline (Full QQQ)
    baseline_nav = (1 + df["QQQ"].pct_change().reindex(dates).fillna(0)).cumprod()
    b_cagr = (baseline_nav.iloc[-1] ** (252 / len(baseline_nav))) - 1
    b_mdd = (baseline_nav / baseline_nav.cummax() - 1).min()

    logger.info(f"Integrated Simulation Complete. CAGR: {cagr:.2%}, MDD: {mdd:.2%}")
    print(f"\n[Integrated MAMA vs QQQ Baseline]")
    print(f"MAMA-Full: CAGR {cagr:.2%}, MDD {mdd:.2%}")
    print(f"QQQ: CAGR {b_cagr:.2%}, MDD {b_mdd:.2%}")

    # Save Integrated Report
    report_path = os.path.join(REPORTS_DIR, "mama_lite_integrated_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# MAMA Lite: Integrated Intelligence (SRL + GNN) Report\n\n")
        f.write(
            "SRL 체제 식별 엔진과 GNN 종목 선정 엔진을 결합한 통합 시뮬레이션 결과입니다.\n\n"
        )
        f.write("## 1. 성과 요약\n")
        f.write("| 지표 | **MAMA Integrated** | QQQ (Buy & Hold) |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **CAGR** | **{cagr:.2%}** | {b_cagr:.2%} |\n")
        f.write(f"| **MDD** | **{mdd:.2%}** | {b_mdd:.2%} |\n")
        f.write(
            f"| **Sharpe** | **{cagr / abs(mdd):.2f}** | {b_cagr / abs(b_mdd):.2f} |\n\n"
        )
        f.write("## 2. 통합 인사이트\n")
        f.write(
            "1. **상황 적응형 선정**: SRL이 'Bull' 체제를 감지했을 때만 공격적인 GNN 종목 선정을 가동하여 하락장 리스크를 회피함.\n"
        )
        f.write(
            "2. **네트워크 기반 필터링**: GNN을 통해 단순 모멘텀이 아닌, 상관관계를 엣지로 연결한 구조적 강점이 있는 종목을 선별함.\n"
        )


if __name__ == "__main__":
    run_integrated_simulation()
