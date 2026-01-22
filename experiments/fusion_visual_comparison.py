# -*- coding: utf-8 -*-
"""
융합 모델 시계열 시그널 정밀 시각화 (Fusion Signal Visual Comparison)
- 4개 차트 구성:
  1. 기존 모델 (Optimized Basic) 시그널 마커
  2. 융합 모델 (Optimized Fusion) 시그널 마커
  3. 누적 수익률 비교 (성능 대조)
  4. 매수/매도 시점 정밀 비교 (필터링 확인)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt

# --- 1. 설정 및 데이터 수집 (EVAL_START 이후 집중) ---
TRAIN_START = "2015-01-01"
EVAL_START = "2020-05-21" # JEPI 상장 이후
END_DATE = datetime.now().strftime("%Y-%m-%d")

TICKERS = ['SPY', 'SCHD', 'QQQ', 'JEPI']
MACRO = {'VIX': '^VIX'}

def fetch_data():
    prices = {}
    for t in TICKERS + [MACRO['VIX']]:
        data = yf.download(t, start=TRAIN_START, end=END_DATE, progress=False, auto_adjust=False, multi_level_index=False)
        prices[t] = data['Close']
    return pd.DataFrame(prices).fillna(method='ffill').dropna()

# --- 2. 시그널 엔진 (기존 fusion_model_test 로직 복제) ---

def get_basic_danger(df_p):
    returns = np.log(df_p['SPY'] / df_p['SPY'].shift(1))
    ma15 = returns.rolling(15).mean()
    vol30 = returns.rolling(30).std()
    signals = []
    ma_hist, vol_hist = [], []
    for i in range(len(df_p)):
        m, v = ma15.iloc[i], vol30.iloc[i]
        if np.isnan(m) or np.isnan(v):
            signals.append(0)
            continue
        if len(ma_hist) > 100:
            p25 = np.percentile(ma_hist, 25)
            p65 = np.percentile(vol_hist, 65)
            signals.append(1 if (m < p25 or v > p65) else 0)
        else:
            signals.append(0)
        ma_hist.append(m)
        vol_hist.append(v)
    return pd.Series(signals, index=df_p.index)

def get_multifactor_score(df_p, lookback=126):
    close, vix = df_p['SPY'], df_p['^VIX']
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema_dist = (close - ema200) / ema200
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain/loss.replace(0, np.nan))).fillna(100)
    
    def get_score(series):
        z = (series - series.rolling(lookback).mean()) / (series.rolling(lookback).std() + 1e-6)
        return pd.Series(norm.cdf(z), index=series.index) * 100
        
    score = (get_score(ema_dist).apply(lambda x: 100-x) * 0.2 + 
             get_score(rsi).apply(lambda x: 100-x) * 0.4 + 
             get_score(vix) * 0.4).rolling(3).mean()
    return score

# --- 3. 데이터 준비 ---
df = fetch_data()
basic_danger = get_basic_danger(df)
mf_score = get_multifactor_score(df)

# 융합 시그널
fusion_signals = []
state = 0
for i in range(len(df)):
    b_danger = basic_danger.iloc[i]
    score = mf_score.iloc[i]
    if state == 0:
        if b_danger and score <= 40: state = 1
    else:
        if not b_danger or score >= 60: state = 0
    fusion_signals.append(state)
fusion_series = pd.Series(fusion_signals, index=df.index)

# 평가 기간 필터링
eval_df = df[df.index >= EVAL_START].copy()
eval_basic = basic_danger[basic_danger.index >= EVAL_START]
eval_fusion = fusion_series[fusion_series.index >= EVAL_START]

# --- 4. 4분할 시각화 ---
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# [Plot 1] 기존 모델 시그널 (Basic)
axes[0, 0].plot(eval_df['SPY'], color='lightgray', alpha=0.5, label='SPY Price')
# Sell Markers (0 -> 1)
s_sell = eval_basic[(eval_basic == 1) & (eval_basic.shift(1) == 0)]
axes[0, 0].scatter(s_sell.index, eval_df.loc[s_sell.index, 'SPY'], color='red', marker='v', label='Sell (JEPI Entry)', s=100, zorder=5)
# Buy Markers (1 -> 0)
s_buy = eval_basic[(eval_basic == 0) & (eval_basic.shift(1) == 1)]
axes[0, 0].scatter(s_buy.index, eval_df.loc[s_buy.index, 'SPY'], color='green', marker='^', label='Buy (QQQ Return)', s=100, zorder=5)
axes[0, 0].set_title("1. Optimized Basic (M1) Signal Markers", fontsize=14)
axes[0, 0].legend()

# [Plot 2] 융합 모델 시그널 (Fusion)
axes[0, 1].plot(eval_df['SPY'], color='lightgray', alpha=0.5, label='SPY Price')
# Sell Markers
f_sell = eval_fusion[(eval_fusion == 1) & (eval_fusion.shift(1) == 0)]
axes[0, 1].scatter(f_sell.index, eval_df.loc[f_sell.index, 'SPY'], color='darkred', marker='v', label='Sell (JEPI Entry)', s=100, zorder=5)
# Buy Markers
f_buy = eval_fusion[(eval_fusion == 0) & (eval_fusion.shift(1) == 1)]
axes[0, 1].scatter(f_buy.index, eval_df.loc[f_buy.index, 'SPY'], color='darkgreen', marker='^', label='Buy (QQQ Return)', s=100, zorder=5)
axes[0, 1].set_title("2. Optimized Fusion (M1+M2) Signal Markers", fontsize=14)
axes[0, 1].legend()

# [Plot 3] 시그널 비교 매트릭스 (필터링 확인)
axes[1, 0].fill_between(eval_basic.index, 0, eval_basic, color='red', alpha=0.3, label='Basic Danger')
axes[1, 0].fill_between(eval_fusion.index, 0, eval_fusion, color='blue', alpha=0.5, label='Fusion Danger')
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_yticklabels(['Normal', 'Danger'])
axes[1, 0].set_title("3. Signal Comparison (Area shows Danger periods)", fontsize=14)
axes[1, 0].legend()

# [Plot 4] 누적 수익 성능 비교 (단순 누적 수익률 기준)
cum_basic = (eval_df['SPY'].pct_change() * (1-eval_basic)).cumsum() # 단순화된 보유 수익률 대조
# 실제 백테스트 결과를 가져와야 하나 시각화를 위해 로직 재현
# (여기서는 이전 백테스트 결과의 경향성을 차트로 표시)
# 간단한 가상 지수화
axes[1, 1].plot((eval_df['SPY']/eval_df['SPY'].iloc[0]), color='gray', alpha=0.3, label='SPY Buy&Hold')
# 실제 전략 로그 재구현 (간이)
def get_cum_ret(prices, signals):
    rets = []
    curr = 1.0
    for i in range(1, len(prices)):
        day_ret = (prices.iloc[i] / prices.iloc[i-1]) - 1
        if signals.iloc[i-1] == 1: # Danger(JEPI/Cash)
            curr *= (1 + 0.0002) # JEPI 대용 일 0.02% 수익 가정
        else:
            curr *= (1 + day_ret)
        rets.append(curr)
    return pd.Series(rets, index=prices.index[1:])

axes[1, 1].plot(get_cum_ret(eval_df['SPY'], eval_basic), label='Basic Strategy', alpha=0.7)
axes[1, 1].plot(get_cum_ret(eval_df['SPY'], eval_fusion), label='Fusion Strategy', linewidth=2)
axes[1, 1].set_title("4. Strategy Wealth Evolution (Standardized)", fontsize=14)
axes[1, 1].legend()

plt.suptitle(f"The Power of Fusion: Technical vs Multi-Factor Context\n(Period: {EVAL_START} to {END_DATE})", fontsize=18, y=0.98)
plt.savefig('fusion_four_panel_comparison.png', dpi=120)
print("\nSuccess! 4-panel visual comparison generated: fusion_four_panel_comparison.png")
