import os
import logging
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import yfinance as yf
from datetime import datetime, timedelta

# v4.0: Import Attention GNN
from attention_gnn import MultiHeadAttentionGNN

# Constants
GNN_DATA_DIR = r"d:\gg\data\gnn"
WEIGHT_FILE = os.path.join(GNN_DATA_DIR, "gnn_weights.pth")
ADJ_FILE = os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv")

# Tickers
# v4.0: Expanded tickers with Healthcare (JNJ) and Financials (V)
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
SRL_TICKERS = ["^VIX", "^TNX", "SPY"]  # Macro indicators
DEFENSIVE_TICKERS = ["BIL", "TLT"]

logger = logging.getLogger("MAMAPredictor")


class SimpleGCN(nn.Module):
    """Graph Convolutional Network for asset selection (v3.2: expanded features)"""

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


class MAMAPredictor:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "config.yaml"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        # v4.0: Load dynamic universe from config (expanded to 11 tickers)
        self.gnn_tickers = self.config.get("strategy_info", {}).get(
            "gnn_tickers",
            [
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
            ],
        )
        self.etf_universe = self.config.get("strategy_info", {}).get(
            "etf_universe",
            ["SPY", "QQQ", "IWM", "TLT", "IEF", "SHY", "GLD", "DBC", "BIL"],
        )

        self.device = torch.device("cpu")
        self.adj_norm = self._load_adjacency()
        self.gnn_model = self._load_gnn_model()

        # Load frozen KMeans model (v3.1)
        self.kmeans_data = self._load_kmeans_model()
        self.scaler = self.kmeans_data["scaler"]
        self.kmeans = self.kmeans_data["kmeans"]
        self.bull_regime_id = self.kmeans_data["bull_regime_id"]
        self.bear_regime_id = self.kmeans_data["bear_regime_id"]

        # Regime Smoothing (v3.1 Week 2)
        self.regime_history = []  # 최근 N일 체제 기록
        self.smoothing_window = 5  # 5일 이동평균

    def _load_adjacency(self):
        if not os.path.exists(ADJ_FILE):
            raise FileNotFoundError(f"Adjacency matrix not found at {ADJ_FILE}")
        adj_df = pd.read_csv(ADJ_FILE, index_col=0)
        A = torch.tensor(adj_df.values, dtype=torch.float32)
        A_hat = A + torch.eye(A.shape[0])
        D = torch.diag(torch.sum(A_hat, dim=1))
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        adj_norm = torch.mm(torch.mm(D_inv_sqrt, A_hat), D_inv_sqrt)
        return adj_norm.to(self.device)

    def _load_gnn_model(self):
        """v4.0: Load Multi-head Attention GNN"""
        # v4.0: Multi-head Attention GNN (4 heads)
        # 새 모델은 가중치 파일 없이 초기화 (추후 학습)
        model = MultiHeadAttentionGNN(
            in_features=10, hidden_features=16, out_features=1, num_heads=4, dropout=0.1
        ).to(self.device)

        # Try to load weights if exists
        if os.path.exists(WEIGHT_FILE):
            try:
                model.load_state_dict(
                    torch.load(WEIGHT_FILE, map_location=self.device, weights_only=True)
                )
                logger.info("Loaded Attention GNN weights from file")
            except Exception as e:
                logger.warning(
                    f"Could not load weights ({e}), using initialized weights"
                )

        model.eval()
        return model

    def _load_kmeans_model(self):
        """怨좎젙??KMeans 紐⑤뜽 濡쒕뱶 (v3.1)"""
        import joblib

        model_path = os.path.join(GNN_DATA_DIR, "kmeans_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"KMeans model not found at {model_path}. "
                f"Please run train_kmeans_model.py first."
            )

        model_data = joblib.load(model_path)
        logger.info(f"Loaded KMeans model (trained on {model_data['training_period']})")
        logger.info(f"Bull Regime: Cluster {model_data['bull_regime_id']}")

        return model_data

    def _calculate_gnn_features(self, df: pd.DataFrame, ticker: str) -> list:
        """Calculate 10 technical indicators for GNN input (v3.2).

        Features:
        1. Momentum (22-day): 22일 수익률
        2. Volatility (21-day): 21일 변동성
        3. RSI (14-day): 상대강도지수
        4. MACD Signal: MACD - Signal Line
        5. Bollinger Position: 가격이 밴드 내 어디에 있는지 (0~1)
        6. Volume Trend: 거래량 20일 이동평균 대비 증감률
        7. 52-Week High Ratio: 현재가 / 52주 최고가
        8. Short Momentum (5-day): 단기 모멘텀
        9. Long Momentum (60-day): 장기 모멘텀
        10. Momentum Divergence: 단기 - 장기 모멘텀 차이

        Returns:
            list: 10개의 정규화된 특성값
        """
        if ticker not in df.columns or len(df[ticker]) < 252:
            return [0.0] * 10

        prices = df[ticker].dropna()
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
            rsi = (100 - (100 / (1 + rs))).iloc[-1] / 100  # 0~1 정규화

            # 4. MACD Signal (12-26-9)
            ema_12 = prices.ewm(span=12).mean()
            ema_26 = prices.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            macd_signal = (macd.iloc[-1] - signal.iloc[-1]) / prices.iloc[
                -1
            ]  # 가격 대비 정규화

            # 5. Bollinger Band Position (20-day, 2 std)
            sma_20 = prices.rolling(20).mean()
            std_20 = prices.rolling(20).std()
            upper_band = sma_20 + 2 * std_20
            lower_band = sma_20 - 2 * std_20
            bb_position = (prices.iloc[-1] - lower_band.iloc[-1]) / (
                upper_band.iloc[-1] - lower_band.iloc[-1] + 1e-10
            )
            bb_position = max(0, min(1, bb_position))  # 0~1 클리핑

            # 6. Volume Trend (데이터 없으면 0)
            volume_trend = 0.0  # yfinance Close only 사용 시

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

        except Exception as e:
            logger.warning(f"Feature calculation failed for {ticker}: {e}")
            return [0.0] * 10

    def fetch_data(self, lookback_days=365):
        """Fetch data for SRL and GNN inference using yfinance."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 100)

        # Macro indicators (mandatory)
        macro_tickers = ["^VIX", "^TNX", "SPY"]
        all_tickers = list(set(self.gnn_tickers + macro_tickers + self.etf_universe))

        try:
            data = yf.download(
                all_tickers, start=start_date, end=end_date, progress=False
            )["Close"]
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            data = data.ffill().dropna()
            return data
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return pd.DataFrame()

    def update_regime_history(self, df=None, lookback_days=400):
        """일별 regime 추적 (v3.1 Smoothing 백테스트용)

        백테스트에서 매일 호출하여 regime_history를 업데이트합니다.
        실제 포트폴리오 예측은 하지 않고 regime만 추적합니다.

        Args:
            df: 이미 로드된 데이터프레임 (백테스트용). None이면 새로 fetch.
            lookback_days: df가 None일 때 사용할 lookback 기간
        """
        if df is None:
            df = self.fetch_data(lookback_days=lookback_days)
        else:
            df = df.copy()  # SettingWithCopyWarning 방지

        if df.empty:
            return

        # Macro indicators check
        for col in ["^VIX", "^TNX", "SPY"]:
            if col not in df.columns:
                return

        # Calculate SRL features
        df["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df[
            "^VIX"
        ].rolling(252).std()
        df["tnx_mom"] = df["^TNX"].pct_change(20)
        df["spy_mom"] = df["SPY"].pct_change(60)
        features = df[["vix_z", "tnx_mom", "spy_mom"]].dropna()

        if len(features) == 0:
            return

        # Predict regime
        X_srl = self.scaler.transform(features)
        regime_labels = self.kmeans.predict(X_srl)
        current_regime = regime_labels[-1]

        # Update history
        self.regime_history.append(current_regime)
        if len(self.regime_history) > self.smoothing_window:
            self.regime_history.pop(0)

    def predict_portfolio(self):
        """Main prediction function (v3.0)."""
        logger.info("Fetching market data...")
        df = self.fetch_data(lookback_days=400)

        if df.empty:
            logger.error("Failed to fetch data.")
            return {}

        # 1. SRL Regime Identification
        for col in ["^VIX", "^TNX", "SPY"]:
            if col not in df.columns:
                logger.error(f"Missing required macro ticker: {col}")
                return {}

        # Calculate Indicators
        df["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df[
            "^VIX"
        ].rolling(252).std()
        df["tnx_mom"] = df["^TNX"].pct_change(20)
        df["spy_mom"] = df["SPY"].pct_change(60)
        features = df[["vix_z", "tnx_mom", "spy_mom"]].dropna()

        # Use frozen KMeans model (v3.1)
        X_srl = self.scaler.transform(features)  # transform only, no fit
        regime_labels = self.kmeans.predict(X_srl)  # predict only, no fit
        current_regime = regime_labels[-1]

        # Regime Smoothing (v3.1 Week 2)
        # 체제 이력 업데이트
        self.regime_history.append(int(current_regime))
        if len(self.regime_history) > self.smoothing_window:
            self.regime_history.pop(0)

        # Bull 확률 계산 (5일 평균)
        bull_probability = sum(
            1 for r in self.regime_history if r == self.bull_regime_id
        ) / len(self.regime_history)

        logger.info(
            f"Current Regime: Cluster {current_regime} (Bull ID: {self.bull_regime_id})"
        )
        logger.info(
            f"Regime History (recent {len(self.regime_history)} days): {self.regime_history}"
        )
        logger.info(f"Bull Probability (5-day avg): {bull_probability:.2%}")

        # 연속적 배분 (v3.1 Week 2 Smoothing)
        stock_weight = bull_probability
        bond_weight = 1 - bull_probability

        logger.info(
            f"[SMOOTHING] Stock Weight: {stock_weight:.2%}, Bond Weight: {bond_weight:.2%}"
        )

        target_weights = {}

        # 주식 비중이 20% 이상일 때만 GNN 가동
        if stock_weight >= 0.2:
            logger.info(
                f"Stock Weight: {stock_weight:.2%} -> Engaging GNN (v3.2: 10 features)"
            )

            # v3.2: Calculate 10 technical indicators per ticker
            node_feats = []
            for t in self.gnn_tickers:
                features = self._calculate_gnn_features(df, t)
                node_feats.append(features)

            x_gnn = torch.tensor(node_feats, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                scores = self.gnn_model(x_gnn, self.adj_norm).squeeze()

            # Select Top 3
            top_indices = scores.argsort(descending=True)[:3]
            top_tickers = [self.gnn_tickers[i] for i in top_indices]
            logger.info(f"GNN Selection: {top_tickers}")

            # 주식 비중을 3개 종목에 균등 분배 (v3.1 Smoothing)
            weight_per_stock = float(stock_weight) / 3
            for t in top_tickers:
                target_weights[t] = float(weight_per_stock)

        else:
            logger.info(f"Stock Weight: {stock_weight:.2%} -> Too low, full defensive")

        # 채권 배분 (v3.1 Smoothing)
        if bond_weight > 0:
            target_weights["BIL"] = float(bond_weight) * 0.5
            target_weights["TLT"] = float(bond_weight) * 0.5

        return target_weights

    def get_current_regime(self):
        """
        Get current market regime classification.
        Returns: str - 'Bull', 'Bear', 'Crisis', or 'Neutral'
        """
        try:
            df = self.fetch_data(lookback_days=400)
            if df.empty:
                return "Unknown"

            # Calculate SRL features
            for col in ["^VIX", "^TNX", "SPY"]:
                if col not in df.columns:
                    return "Unknown"

            df["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df[
                "^VIX"
            ].rolling(252).std()
            df["tnx_mom"] = df["^TNX"].pct_change(20)
            df["spy_mom"] = df["SPY"].pct_change(60)

            features = df[["vix_z", "tnx_mom", "spy_mom"]].dropna()
            if features.empty:
                return "Unknown"

            # Fit and predict
            X_srl = self.scaler.fit_transform(features)
            regime_labels = self.kmeans.fit_predict(X_srl)
            current_regime = regime_labels[-1]

            # Classify regime based on characteristics
            features_with_ret = features.copy()
            features_with_ret["spy_ret"] = (
                df["SPY"].pct_change().reindex(features.index)
            )
            features_with_ret["regime"] = regime_labels

            regime_spy_ret = features_with_ret.groupby("regime")["spy_ret"].mean()
            bull_regime = regime_spy_ret.idxmax()
            bear_regime = regime_spy_ret.idxmin()

            # Get current VIX Z-score
            current_vix_z = features["vix_z"].iloc[-1]

            if current_regime == bull_regime:
                return "Bull"
            elif current_regime == bear_regime:
                if current_vix_z > 2.0:  # Extreme fear
                    return "Crisis"
                return "Bear"
            else:
                return "Neutral"

        except Exception as e:
            logger.error(f"Error getting regime: {e}")
            return "Unknown"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predictor = MAMAPredictor()
    weights = predictor.predict_portfolio()
    print("Target Portfolio Weights:", weights)
