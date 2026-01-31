import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import yfinance as yf
from datetime import datetime, timedelta
import os


def train_and_save_kmeans():
    """
    KMeans 모델을 히스토리 데이터(2004-2024)로 학습하고 저장.
    이후 predict만 수행하여 클러스터 레이블 일관성 확보.
    """
    print("🚀 Training KMeans Model with Historical Data...")

    # 1. 20년 히스토리 데이터 로드
    end_date = datetime(2024, 12, 31)
    start_date = datetime(2004, 1, 1)

    tickers = ["^VIX", "^TNX", "SPY"]
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]

    # 2. Feature Engineering
    data["vix_z"] = (data["^VIX"] - data["^VIX"].rolling(252).mean()) / data[
        "^VIX"
    ].rolling(252).std()
    data["tnx_mom"] = data["^TNX"].pct_change(20)
    data["spy_mom"] = data["SPY"].pct_change(60)

    features = data[["vix_z", "tnx_mom", "spy_mom"]].dropna()

    print(
        f"   Training period: {features.index[0].date()} to {features.index[-1].date()}"
    )
    print(f"   Total samples: {len(features)}")

    # 3. Scaler 학습 및 저장
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # 4. KMeans 학습 (고정 random_state)
    kmeans = KMeans(n_clusters=4, n_init=20, random_state=42)
    kmeans.fit(X_scaled)

    # 5. Bull Regime 식별
    features_with_ret = features.copy()
    features_with_ret["spy_ret"] = data["SPY"].pct_change().reindex(features.index)
    features_with_ret["regime"] = kmeans.labels_

    regime_spy_ret = features_with_ret.groupby("regime")["spy_ret"].mean()
    bull_regime_id = int(regime_spy_ret.idxmax())
    bear_regime_id = int(regime_spy_ret.idxmin())

    # 6. 모델 및 메타데이터 저장
    model_data = {
        "kmeans": kmeans,
        "scaler": scaler,
        "bull_regime_id": bull_regime_id,
        "bear_regime_id": bear_regime_id,
        "training_period": f"{start_date.date()} to {end_date.date()}",
        "regime_characteristics": regime_spy_ret.to_dict(),
    }

    save_dir = r"d:\gg\data\gnn"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "kmeans_model.pkl")
    joblib.dump(model_data, save_path)

    print(f"\n✅ KMeans Model Saved to {save_path}")
    print(
        f"   Bull Regime: Cluster {bull_regime_id} (Avg SPY Return: {regime_spy_ret[bull_regime_id]:.4f})"
    )
    print(
        f"   Bear Regime: Cluster {bear_regime_id} (Avg SPY Return: {regime_spy_ret[bear_regime_id]:.4f})"
    )

    # 7. 검증: 과거 레이블 분포
    print("\n📊 Historical Regime Distribution:")
    print(features_with_ret["regime"].value_counts().sort_index())

    return model_data


if __name__ == "__main__":
    train_and_save_kmeans()
