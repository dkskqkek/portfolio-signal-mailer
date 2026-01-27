# -*- coding: utf-8 -*-
"""
Project JARVIS Engine (v1.1 Live Connector)
------------------------------------------
Antigravity 시스템을 위한 ML Sidecar 엔진
- 역할 1: 시장 국면 탐지 (Regime Classifier)
- 역할 2: 파라미터 최적화 (Parameter Tuner)
- 모드: Live Mode (제안된 파라미터를 jarvis_config.json에 저장)

Author: Antigravity AI Partner
Date: 2026-01-25
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ML Libraries
try:
    import optuna
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ 경고: optuna 또는 lightgbm이 설치되지 않았습니다.")

# =========================================================
# CONFIG & UTILS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "jarvis_engine.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] JARVIS: %(message)s",
)


class DataFetcher:
    """JARVIS용 학습 데이터 수집기"""

    @staticmethod
    def get_market_data(days=365 * 5):
        tickers = ["^VIX", "DX-Y.NYB", "^TNX", "QQQ", "SPY"]
        end = datetime.now()
        start = end - timedelta(days=days)

        # yfinance 2025/2026 호환성
        data = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
        df = pd.DataFrame()

        for t in tickers:
            if t in data.columns.get_level_values(0):
                cols = data[t]
                # 컬럼 이름 유연하게 처리
                price = cols["Close"] if "Close" in cols.columns else cols.iloc[:, 0]
                df[t] = price
            else:
                try:
                    df[t] = data[t]["Close"]
                except:
                    pass

        return df.ffill().dropna()


# =========================================================
# MODULE A: Regime Classifier
# =========================================================
class RegimeClassifier:
    def __init__(self):
        self.model = None

    def prepare_features(self, df):
        df = df.copy()
        df["vix_ma20"] = df["^VIX"].rolling(20).mean()
        df["vix_ratio"] = df["^VIX"] / df["vix_ma20"]
        df["tnx_chg"] = df["^TNX"].pct_change(20)
        df["dxy_chg"] = df["DX-Y.NYB"].pct_change(60)
        df["qqq_ma120"] = df["QQQ"].rolling(120).mean()
        df["qqq_dist"] = (df["QQQ"] / df["qqq_ma120"]) - 1
        return df.dropna()

    def create_labels(self, df, lookahead=20):
        future_ret = df["QQQ"].pct_change(lookahead).shift(-lookahead)
        future_mdd = (
            df["QQQ"]
            .rolling(lookahead)
            .apply(lambda x: (x.min() / x[0]) - 1)
            .shift(-lookahead)
        )

        # 0:Normal, 1:Choppy, 2:Crash
        conditions = [(future_mdd < -0.15), (future_ret.abs() < 0.03)]
        choices = [2, 1]
        df["label"] = np.select(conditions, choices, default=0)
        return df.dropna()

    def train(self, df):
        if not ML_AVAILABLE:
            return False
        X = df[["vix_ratio", "tnx_chg", "dxy_chg", "qqq_dist"]]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        self.model = lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, verbose=-1
        )
        self.model.fit(X_train, y_train)

        acc = accuracy_score(y_test, self.model.predict(X_test))
        logging.info(f"Regime Classifier 학습 완료. Accuracy: {acc:.2f}")
        return True

    def predict_current(self, df):
        if not self.model:
            return "Unknown", 0.0
        latest = df.iloc[[-1]][["vix_ratio", "tnx_chg", "dxy_chg", "qqq_dist"]]
        pred = self.model.predict(latest)[0]
        prob = self.model.predict_proba(latest)[0]
        regimes = {0: "Normal", 1: "Choppy", 2: "Crash"}
        return regimes.get(pred, "Unknown"), prob


# =========================================================
# MODULE B: Parameter Tuner (Optuna)
# =========================================================
class ParamTuner:
    def __init__(self):
        self.best_params = {}

    def optimize(self, df):
        if not ML_AVAILABLE:
            return None

        def objective(trial):
            # Guardrail 범위 내 탐색
            s1 = trial.suggest_int("s1", 110, 150)
            s2 = trial.suggest_int("s2", 200, 280)

            ma_fast = df["QQQ"].rolling(s1).mean()
            ma_slow = df["QQQ"].rolling(s2).mean()
            signal = (df["QQQ"] > ma_fast) & (df["QQQ"] > ma_slow)
            returns = df["QQQ"].pct_change().shift(-1)
            strategy_ret = signal * returns

            cagr = strategy_ret.mean() * 252
            mdd = (strategy_ret.cumsum() - strategy_ret.cumsum().cummax()).min()

            # Objective: (CAGR * 0.6) - (|MDD| * 0.4)
            score = (cagr * 0.6) - (abs(mdd) * 0.4)
            return score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        self.best_params = study.best_params
        return self.best_params


# =========================================================
# LIVE CONNECTOR
# =========================================================
def save_config(params, regime, prob):
    """JARVIS의 제안을 JSON으로 저장 (Sidecar -> Host)"""
    config_path = os.path.join(DATA_DIR, "jarvis_config.json")

    # 0:Normal, 1:Choppy, 2:Crash
    crash_prob = float(prob[2]) if len(prob) > 2 else 0.0

    payload = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "suggested_params": params,
        "market_regime": regime,
        "crash_probability": crash_prob,
        "confidence_score": float(np.max(prob)),  # 가장 높은 확률
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ JARVIS Config 저장 완료: {config_path}")
    logging.info(f"Config Saved: {payload}")


def run_jarvis():
    print(
        f"--- 🧠 JARVIS Engine (Live) Start [{datetime.now().strftime('%Y-%m-%d')}] ---"
    )

    if not ML_AVAILABLE:
        print("❌ ML 라이브러리 부재로 실행 중단.")
        return

    # 1. Data Fetch
    print("1. 데이터 수집 및 전처리...")
    fetcher = DataFetcher()
    raw_df = fetcher.get_market_data()

    # 2. Regime Analysis
    print("2. 시장 국면 분석 (LightGBM)...")
    clf = RegimeClassifier()
    feat_df = clf.prepare_features(raw_df)
    labeled_df = clf.create_labels(feat_df)

    regime = "Unknown"
    prob = [0, 0, 0]

    if clf.train(labeled_df):
        regime, prob = clf.predict_current(feat_df)
        print(f"   👉 국면 예측: {regime} (Crash Prob: {prob[2] * 100:.1f}%)")

    # 3. Parameter Tuning
    print("3. 파라미터 최적화 (Optuna)...")
    tuner = ParamTuner()
    best_params = tuner.optimize(raw_df)

    if best_params:
        print(
            f"   👉 최적 파라미터: Fast {best_params['s1']} / Slow {best_params['s2']}"
        )

        # 4. Save Config (The Handshake)
        save_config(best_params, regime, prob)

    print("✅ JARVIS Engine 실행 완료.")


if __name__ == "__main__":
    run_jarvis()
