# -*- coding: utf-8 -*-
"""
ML 기반 시그널 정교화 엔진 (ml_signal_refiner.py)
- 목표: M1(기본 시그널)의 판정을 M2(ML 모델)가 매크로 문맥을 보고 필터링
- 적용 기법: Meta-Labeling (Marcos Lopez de Prado)
- 데이터: 기술적 지표 + 거시 경제 지표 (VIX, TNX, DXY 등)
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

# --- 1. 데이터 수집 설정 ---
TICKER_SPY = "SPY"
MACRO_TICKERS = {
    "VIX": "^VIX",        # 공포 지수
    "TNX": "^TNX",        # 미국 10년물 금리
    "DXY": "DX-Y.NYB",    # 달러 인덱스
    "TYVIX": "^TYVIX",    # 채권 변동성
    "GOLD": "GC=F"        # 금 선물
}
START_DATE = "2015-01-01" # 충분한 학습 데이터 확보

class MLSignalRefiner:
    def __init__(self, start_date=START_DATE):
        self.start_date = start_date
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.features = []

    def fetch_comprehensive_data(self):
        """SPY 가격 데이터 및 매크로 지표 수집"""
        print("Gathering data for ML training...")
        # SPY (Primary Ticker)
        spy = yf.download(TICKER_SPY, start=self.start_date, end=self.end_date)
        if hasattr(spy.columns, 'nlevels') and spy.columns.nlevels > 1:
            spy.columns = spy.columns.get_level_values(0)
        
        data = pd.DataFrame(index=spy.index)
        data['SPY_Close'] = spy['Close']
        data['SPY_Ret'] = spy['Close'].pct_change()
        
        # Macro Indicators
        for name, ticker in MACRO_TICKERS.items():
            m_data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            if hasattr(m_data.columns, 'nlevels') and m_data.columns.nlevels > 1:
                m_data.columns = m_data.columns.get_level_values(0)
            data[name] = m_data['Close']
        
        return data.fillna(method='ffill').dropna()

    def add_technical_features(self, df):
        """기술적 피처 추가"""
        df = df.copy()
        # Returns & Volatility
        df['Vol_15'] = df['SPY_Ret'].rolling(15).std()
        df['Vol_30'] = df['SPY_Ret'].rolling(30).std()
        df['MA_15_Gap'] = df['SPY_Close'] / df['SPY_Close'].rolling(15).mean() - 1
        
        # Momentum
        df['RSI'] = self.calculate_rsi(df['SPY_Close'])
        
        # Macro Changes
        df['TNX_Chg'] = df['TNX'].pct_change()
        df['DXY_Chg'] = df['DXY'].pct_change()
        df['VIX_Level'] = df['VIX']
        
        # YOY/MOM Gaps
        df['Yield_Curve'] = df['TNX'] - 2.0 # 단순화된 10Y-Target (실제 2Y 데이터 부족 시 대용)
        
        return df.dropna()

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_meta_labels(self, df, window=5):
        """
        Meta-Labeling (Triple Barrier 단순화):
        - M1 시그널(Danger)이 발생했을 때, 실제로 이후 window일 동안 수익이 났는가?
        - 여기서는 'DANGER 가동 상태에서 방어 성공 여부'를 라벨링
        """
        # M1 시그널 로직 재현 (15d MA 하위 25% or 30d Vol 상위 65%)
        log_ret = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
        ma15 = log_ret.rolling(15).mean()
        vol30 = log_ret.rolling(30).std()
        
        # Rolling Percentile (대략적 구현)
        df['M1_MA_Pct'] = ma15.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        df['M1_Vol_Pct'] = vol30.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        df['M1_Signal'] = ((df['M1_MA_Pct'] < 0.25) | (df['M1_Vol_Pct'] > 0.65)).astype(int)
        
        # Label Y: M1이 위험(1)이라고 했을 때, 향후 5일간 SPY가 실제로 하락했는가? (방어 성공=1, 불필요한 공포=0)
        # 즉, Forward Return이 마이너스면 '위험 판정이 옳았다'는 뜻
        forward_ret = df['SPY_Close'].shift(-window) / df['SPY_Close'] - 1
        df['Label'] = (forward_ret < -0.01).astype(int) # 1% 이상 하락했을 때만 '진짜 위험'으로 간주
        
        return df.dropna()

    def train_meta_model(self, df):
        """ML 모델 학습 - M1이 '위험'을 외친 시점들만 대상으로 학습"""
        # M1이 시그널을 준 샘플만 추출 (Meta-Labeling 핵심)
        meta_df = df[df['M1_Signal'] == 1].copy()
        
        X_cols = ['Vol_15', 'Vol_30', 'MA_15_Gap', 'RSI', 'TNX_Chg', 'DXY_Chg', 'VIX_Level']
        X = meta_df[X_cols]
        y = meta_df['Label']
        
        if len(y.unique()) < 2:
            print("Not enough labels for training ML. Using default rules.")
            return None
        
        # Time Series Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        print("\n[Meta-Labeler Performance]")
        print(classification_report(y_test, preds))
        
        self.features = X_cols
        return self.model

    def get_hybrid_signal(self, current_data_row):
        """
        최종 하이브리드 판정:
        1. M1이 NORMAL이면 -> NORMAL
        2. M1이 DANGER이면 -> ML(M2)에게 물어봄
        3. M2가 '진짜 위험(1)'이라고 하면 -> DANGER
        4. M2가 '가짜 공포(0)'라고 하면 -> NORMAL (필터링)
        """
        m1_danger = (current_data_row['M1_Signal'] == 1)
        if not m1_danger:
            return 0 # Normal
        
        # ML 필터링
        x_input = current_data_row[self.features].values.reshape(1, -1)
        m2_confidence = self.model.predict_proba(x_input)[0][1] # 위험할 확률
        
        # 보수적으로 0.5 이상이면 위험 수용
        return 1 if m2_confidence >= 0.5 else 0

def main():
    refiner = MLSignalRefiner()
    raw_df = refiner.fetch_comprehensive_data()
    df = refiner.add_technical_features(raw_df)
    df = refiner.create_meta_labels(df)
    
    model = refiner.train_meta_model(df)
    
    if model:
        # 결과 로그
        print(f"ML Feature Importance: {dict(zip(refiner.features, model.feature_importances_))}")
        
        # 하이브리드 시그널 생성
        df['Hybrid_Signal'] = df.apply(lambda row: refiner.get_hybrid_signal(row), axis=1)
        
        m1_total = df['M1_Signal'].sum()
        hybrid_total = df['Hybrid_Signal'].sum()
        reduced = (1 - hybrid_total / m1_total) * 100
        print(f"\n[Signal Filtering Summary]")
        print(f"M1 Original Danger Signals: {m1_total}")
        print(f"Hybrid Confirmed Danger Signals: {hybrid_total}")
        print(f"Noise Filtered: {reduced:.1f}%")
        
        # 데이터 저장 (백테스트 연동용)
        df.to_csv("ml_augmented_signals.csv")
        print("Final signals saved to ml_augmented_signals.csv")

if __name__ == "__main__":
    main()
