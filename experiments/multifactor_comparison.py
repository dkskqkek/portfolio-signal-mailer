# -*- coding: utf-8 -*-
"""
사용자 멀티팩터 모델 vs 기존 최적화 기본 시그널 성과 비교
- 2020년~현재, 포트폴리오 환경 (SCHD/QQQ/JEPI/GLD/KOSPI)
- 사용자 모델을 '위험(Greed) 시 탈출' 로직으로 변환하여 비교
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt

# --- 1. 설정 및 데이터 수집 ---
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
TICKERS = ['SPY', 'SCHD', 'QQQ', 'JEPI', 'GLD', '^KS200']

def fetch_data():
    prices = {}
    for t in TICKERS + ['^VIX']:
        data = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False, multi_level_index=False)
        prices[t] = data['Adj Close']
    return pd.DataFrame(prices).fillna(method='ffill').dropna()

# --- 2. 사용자 멀티팩터 모델 (MarketTimingModel 로직 이식) ---
def calculate_multifactor_score(df, lookback=126):
    temp = pd.DataFrame(index=df.index)
    temp['Close'] = df['SPY']
    temp['VIX'] = df['^VIX']
    
    # 지표 계산 (사용자 코드 반영)
    temp['EMA_200'] = temp['Close'].ewm(span=200, adjust=False).mean()
    temp['EMA_Dist'] = (temp['Close'] - temp['EMA_200']) / temp['EMA_200']
    
    # RSI
    delta = temp['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    temp['RSI'] = 100 - (100 / (1 + gain/loss.replace(0, np.nan))).fillna(100)
    
    # Williams %R (대략적 구현, High/Low 데이터 대신 Close 변동성 활용 가능하나 사용자 로직 존중 위해 SPY 전체 데이터 다시 수집)
    # 여기서는 단순화를 위해 사용자 로직의 EMA_Dist, RSI, VIX 위주로 정규화 테스트
    
    def normalize(series, window, inverse=False):
        roll_mean = series.rolling(window=window).mean()
        roll_std = series.rolling(window=window).std()
        z = (series - roll_mean) / (roll_std + 1e-6)
        score = pd.Series(norm.cdf(z), index=series.index) * 100
        return 100 - score if inverse else score

    s_trend = normalize(temp['EMA_Dist'], lookback, inverse=True)
    s_mom = normalize(temp['RSI'], lookback, inverse=True)
    s_vol = normalize(temp['VIX'], lookback, inverse=False)
    
    # 사용자 가용 지표 기반 가중치 재조정 (WillR 제외 시)
    # EMA(15%), RSI(25%), VIX(30%), Breadth(30%)인데 Breadth 대안으로 RSI 가중치 상향
    score = (s_vol * 0.4 + s_mom * 0.4 + s_trend * 0.2).rolling(3).mean()
    
    # 사용자 판정: Score <= 20 (Greed) -> 위험(Danger), Score >= 80 (Fear) -> 정상(Normal)
    # 마켓 타이밍을 위해 "과열 국면에서 탈출" 하는 시그널로 활용
    mf_is_danger = []
    current_state = 0 # 0: Normal, 1: Danger
    for s in score:
        if s <= 30: # 과열(Greed) 시 위험 감지
            current_state = 1
        elif s >= 60: # 공포(Fear) 시 정상 복귀
            current_state = 0
        mf_is_danger.append(current_state)
        
    return pd.Series(mf_is_danger, index=df.index)

# --- 3. 기존 최적화 기본 시그널 (15d MA / 30d Vol) ---
def calculate_basic_signal(df):
    spy_ret = np.log(df['SPY'] / df['SPY'].shift(1))
    ma15 = spy_ret.rolling(15).mean()
    vol30 = spy_ret.rolling(30).std()
    
    signals = []
    hist_ma = []
    hist_vol = []
    for i in range(len(df)):
        if i < 30: 
            signals.append(0)
            if not np.isnan(ma15.iloc[i]): hist_ma.append(ma15.iloc[i])
            if not np.isnan(vol30.iloc[i]): hist_vol.append(vol30.iloc[i])
            continue
        
        p25 = np.nanpercentile(hist_ma, 25)
        p65 = np.nanpercentile(hist_vol, 65)
        
        danger = (ma15.iloc[i] < p25) or (vol30.iloc[i] > p65)
        signals.append(1 if danger else 0)
        
        hist_ma.append(ma15.iloc[i])
        hist_vol.append(vol30.iloc[i])
        
    return pd.Series(signals, index=df.index)

# --- 4. 백테스트 엔진 (포트폴리오 환경) ---
def run_backtest(df, signal_series, core_ticker='SCHD'):
    initial_cash = 100000
    weights = {'CORE': 0.38, 'DYNAMIC': 0.38, 'GOLD': 0.05, 'KOSPI': 0.19}
    
    # 수수료/슬리피지 0.1% 반영
    COST = 0.001
    
    p = df.iloc[0]
    shares = {
        core_ticker: (initial_cash * weights['CORE'] * (1-COST)) / p[core_ticker],
        'QQQ': (initial_cash * weights['DYNAMIC'] * (1-COST)) / p['QQQ'],
        'GLD': (initial_cash * weights['GOLD'] * (1-COST)) / p['GLD'],
        '^KS200': (initial_cash * weights['KOSPI'] * (1-COST)) / p['^KS200'],
        'JEPI': 0.0
    }
    
    history = []
    current_mode = 0 # 0: QQQ, 1: JEPI
    
    for i in range(len(df)):
        date = df.index[i]
        p_curr = df.iloc[i]
        is_danger = (signal_series.iloc[i] == 1)
        
        # 스위칭 로직
        if is_danger and current_mode == 0:
            # QQQ -> JEPI
            val = shares['QQQ'] * p_curr['QQQ'] * (1-COST)
            shares['JEPI'] = (val * (1-COST)) / p_curr['JEPI'] if p_curr['JEPI'] > 0 else 0
            shares['QQQ'] = 0
            current_mode = 1
        elif not is_danger and current_mode == 1:
            # JEPI -> QQQ
            val = shares['JEPI'] * p_curr['JEPI'] * (1-COST)
            shares['QQQ'] = (val * (1-COST)) / p_curr['QQQ']
            shares['JEPI'] = 0
            current_mode = 0
            
        # 총 가치 (배당은 이 비교 백테스트에서는 제외하거나 단순화)
        total_val = sum(shares[t] * p_curr[t] for t in shares)
        history.append(total_val)
        
    return pd.Series(history, index=df.index)

def analyze成果(series):
    days = (series.index[-1] - series.index[0]).days
    cagr = (series.iloc[-1] / series.iloc[0]) ** (365/days) - 1
    mdd = (series / series.cummax() - 1).min()
    daily_ret = series.pct_change()
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252))
    return {'Final': series.iloc[-1], 'CAGR': cagr*100, 'MDD': mdd*100, 'Sharpe': sharpe}

def main():
    print("데이터 수집 중...")
    df = fetch_data()
    
    print("시그널 생성 중...")
    sig_basic = calculate_basic_signal(df)
    sig_mf = calculate_multifactor_score(df)
    
    print("백테스트 실행 중...")
    res_basic = run_backtest(df, sig_basic)
    res_mf = run_backtest(df, sig_mf)
    
    stats_b = analyze成果(res_basic)
    stats_mf = analyze成果(res_mf)
    
    print("\n" + "="*70)
    print(f"{'Strategy':<25} | {'Final ($)':<12} | {'CAGR (%)':<10} | {'MDD (%)':<10} | {'Sharpe':<8}")
    print("-" * 75)
    print(f"{'Basic (MA/Vol Optimized)':<25} | {stats_b['Final']:>12,.0f} | {stats_b['CAGR']:>10.2f} | {stats_b['MDD']:>10.2f} | {stats_b['Sharpe']:>8.2f}")
    print(f"{'User Multifactor CDF':<25} | {stats_mf['Final']:>12,.0f} | {stats_mf['CAGR']:>10.2f} | {stats_mf['MDD']:>10.2f} | {stats_mf['Sharpe']:>8.2f}")
    print("="*70)
    
    # 결과 저장
    with open('multifactor_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"{'Strategy':<25} | {'Final ($)':<12} | {'CAGR (%)':<10} | {'MDD (%)':<10} | {'Sharpe':<8}\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'Basic (MA/Vol Optimized)':<25} | {stats_b['Final']:>12,.0f} | {stats_b['CAGR']:>10.2f} | {stats_b['MDD']:>10.2f} | {stats_b['Sharpe']:>8.2f}\n")
        f.write(f"{'User Multifactor CDF':<25} | {stats_mf['Final']:>12,.0f} | {stats_mf['CAGR']:>10.2f} | {stats_mf['MDD']:>10.2f} | {stats_mf['Sharpe']:>8.2f}\n")
        f.write("="*70 + "\n")
    
    plt.figure(figsize=(12, 7))
    plt.plot(res_basic, label='Basic (Optimized)')
    plt.plot(res_mf, label='User Multifactor (Greed-Exit)')
    plt.title("Portfolio Performance Comparison (2020-Present)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('multifactor_vs_basic_result.png')
    print("\nChart saved: multifactor_vs_basic_result.png")

if __name__ == "__main__":
    main()
