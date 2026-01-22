# -*- coding: utf-8 -*-
"""
정밀 비교우위 평가 분석 (Comparative Advantage Evaluation Analysis)
- 기존 모델 (Optimized Basic: 15d MA / 30d Vol)
- 사용자 모델 (Multifactor CDF Scoring: EMA, RSI, VIX)
- 기간: 2017-01-01 ~ 현재 (평가는 2020-01-01부터)
- 포트폴리오: SCHD(38%), QQQ/JEPI(38%), GOLD(5%), KOSPI(19%)
- 모든 실전 비용(0.1% 슬리피지, 연간 세수 22%) 반영
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt

# --- 1. 전역 설정 ---
START_TRAIN = "2015-01-01" # 지표 정규화를 위한 선행 데이터
START_EVAL = "2020-01-01"  # 실제 성과 비교 구간 (JEPI 상장 이후 로직 통일)
END_DATE = datetime.now().strftime("%Y-%m-%d")

TICKERS = ['SPY', 'SCHD', 'QQQ', 'JEPI', 'GLD', '^KS200']
MACRO = {'VIX': '^VIX'}

COST = 0.0011 # 슬리피지 + 수수료
TAX_RATE = 0.22
TAX_FREE_LIMIT_USD = 2500000 / 1450 # 약 $1,724

# --- 2. 데이터 엔진 ---
def fetch_robust_data():
    prices = {}
    divs = {}
    for t in TICKERS + [MACRO['VIX']]:
        data = yf.download(t, start=START_TRAIN, end=END_DATE, progress=False, auto_adjust=False, multi_level_index=False)
        prices[t] = data['Close']
        if t in TICKERS:
            divs[t] = data['Dividends'] if 'Dividends' in data.columns else pd.Series(0, index=data.index)
            
    df_p = pd.DataFrame(prices).fillna(method='ffill').dropna()
    df_d = pd.DataFrame(divs).reindex(df_p.index).fillna(0)
    return df_p, df_d

# --- 3. 시그널 엔진 A: Optimized Basic ---
def get_basic_signal(df_p):
    spy_ret = np.log(df_p['SPY'] / df_p['SPY'].shift(1))
    ma15 = spy_ret.rolling(15).mean()
    vol30 = spy_ret.rolling(30).std()
    
    signals = []
    # 누적 통계를 위한 리스트
    ma_hist = []
    vol_hist = []
    
    for i in range(len(df_p)):
        m = ma15.iloc[i]
        v = vol30.iloc[i]
        
        if np.isnan(m) or np.isnan(v):
            signals.append(0)
            continue
            
        # 임계값 계산 (동적 백테스트 재현)
        if len(ma_hist) > 100:
            p25 = np.percentile(ma_hist, 25)
            p65 = np.percentile(vol_hist, 65)
            danger = (m < p25) or (v > p65)
            signals.append(1 if danger else 0)
        else:
            signals.append(0)
            
        ma_hist.append(m)
        vol_hist.append(v)
        
    return pd.Series(signals, index=df_p.index)

# --- 4. 시그널 엔진 B: Multifactor CDF ---
def get_multifactor_signal(df_p, lookback=126):
    close = df_p['SPY']
    vix = df_p['^VIX']
    
    # 지표 산정
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema_dist = (close - ema200) / ema200
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain/loss.replace(0, np.nan))).fillna(100)
    
    # 정규화 함수
    def get_score(series, inv=False):
        m = series.rolling(lookback).mean()
        s = series.rolling(lookback).std()
        z = (series - m) / (s + 1e-6)
        score = pd.Series(norm.cdf(z), index=series.index) * 100
        return 100 - score if inv else score
        
    s_trend = get_score(ema_dist, inv=True)
    s_mom = get_score(rsi, inv=True)
    s_vol = get_score(vix, inv=False)
    
    # 가중치 (Breahth 제외하고 조정)
    score = (s_trend * 0.2 + s_mom * 0.4 + s_vol * 0.4).rolling(3).mean()
    
    # 판정 (사용자 로직: 80점 이상 매수=Fear, 20점 이하 매도=Greed)
    # 우리는 "위험 감지" 시그널이므로 Greed(점수 낮음) 시 위험(1)으로 판정
    signals = []
    state = 0 # 0: Normal, 1: Danger
    for s in score:
        if s <= 30: # 과열 국면 -> 위험
            state = 1
        elif s >= 60: # 공포 국면/조정 완료 -> 정상
            state = 0
        signals.append(state)
        
    return pd.Series(signals, index=df_p.index)

# --- 5. 백테스트 시뮬레이터 (통일) ---
def simulate(df_p, df_d, signals, name):
    # 평가 시작 구간으로 데이터 필터링은 하지만, 이전 상태는 유지해야 함
    # (실제로는 전체 기간을 돌리고 EVAL_START 이후만 분석)
    
    initial_cash = 100000
    shares = {t: 0.0 for t in TICKERS}
    
    # 초기 설정 (EVAL_START 시점의 가격으로 진입)
    eval_idx = df_p.index.get_loc(df_p.index[df_p.index >= START_EVAL][0])
    p0 = df_p.iloc[eval_idx]
    
    shares['SCHD'] = (initial_cash * 0.38 * (1-COST)) / p0['SCHD']
    shares['QQQ'] = (initial_cash * 0.38 * (1-COST)) / p0['QQQ']
    shares['GLD'] = (initial_cash * 0.05 * (1-COST)) / p0['GLD']
    shares['^KS200'] = (initial_cash * 0.19 * (1-COST)) / p0['^KS200']
    
    history = []
    current_mode = 0 # 0: QQQ, 1: JEPI
    yearly_profit = 0
    prev_year = p0.name.year
    trade_count = 0
    
    for i in range(eval_idx, len(df_p)):
        date = df_p.index[i]
        p = df_p.iloc[i]
        d = df_d.iloc[i]
        sig = signals.iloc[i]
        
        # 1. 세금 정산 (연말)
        if date.year != prev_year:
            if yearly_profit > TAX_FREE_LIMIT_USD:
                tax = (yearly_profit - TAX_FREE_LIMIT_USD) * TAX_RATE
                total_v = sum(shares[t] * p[t] for t in TICKERS)
                shares = {t: v * (1 - tax/total_v) for t, v in shares.items()}
            yearly_profit = 0
            prev_year = date.year
            
        # 2. 배당 재투자
        for t in TICKERS:
            if shares[t] > 0 and d[t] > 0:
                div_val = shares[t] * d[t] * (1 - 0.154) # 배당소득세
                shares[t] += (div_val * (1-COST)) / p[t]
                
        # 3. 스위칭 시그널 처리
        if sig == 1 and current_mode == 0: # Danger 진입 -> QQQ 매도, JEPI 매수
            val = shares['QQQ'] * p['QQQ'] * (1-COST)
            shares['JEPI'] = (val * (1-COST)) / p['JEPI']
            shares['QQQ'] = 0
            current_mode = 1
            trade_count += 1
        elif sig == 0 and current_mode == 1: # Normal 복귀 -> JEPI 매도, QQQ 매수
            val = shares['JEPI'] * p['JEPI'] * (1-COST)
            shares['QQQ'] = (val * (1-COST)) / p['QQQ']
            shares['JEPI'] = 0
            current_mode = 0
            trade_count += 1
            
        # 총 가치 기록
        total_val = sum(shares[t] * p[t] for t in TICKERS)
        history.append(total_val)
        if len(history) > 1:
            yearly_profit += (history[-1] - history[-2])
            
    return pd.Series(history, index=df_p.index[eval_idx:]), trade_count

# --- 6. 주 실행부 ---
def main():
    print("Fetching data from 2015...")
    df_p, df_d = fetch_robust_data()
    
    print("Generating Signals...")
    sig_basic = get_basic_signal(df_p)
    sig_mf = get_multifactor_signal(df_p)
    
    print("Simulating Strategies (2020-Present)...")
    res_b, trades_b = simulate(df_p, df_d, sig_basic, "Basic")
    res_mf, trades_mf = simulate(df_p, df_d, sig_mf, "Multifactor")
    
    def get_stats(series, trades):
        days = (series.index[-1] - series.index[0]).days
        cagr = (series.iloc[-1] / series.iloc[0]) ** (365/days) - 1
        mdd = (series / series.cummax() - 1).min()
        vol = series.pct_change().std() * np.sqrt(252)
        sharpe = (series.pct_change().mean() * 252) / (series.pct_change().std() * np.sqrt(252))
        return {
            'CAGR': cagr*100, 'MDD': mdd*100, 'Sharpe': sharpe, 
            'Vol': vol*100, 'Trades': trades, 'Final': series.iloc[-1]
        }
    
    s_b = get_stats(res_b, trades_b)
    s_mf = get_stats(res_mf, trades_mf)
    
    # 보고서 출력
    print("\n" + "="*80)
    print(f"{'Metric':<20} | {'Optimized Basic':<20} | {'Multifactor CDF'}")
    print("-" * 80)
    print(f"{'Final Value ($)':<20} | {s_b['Final']:>18,.0f} | {s_mf['Final']:>18,.0f}")
    print(f"{'CAGR (%)':<20} | {s_b['CAGR']:>18.2f} | {s_mf['CAGR']:>18.2f}")
    print(f"{'MDD (%)':<20} | {s_b['MDD']:>18.2f} | {s_mf['MDD']:>18.2f}")
    print(f"{'Sharpe Ratio':<20} | {s_b['Sharpe']:>18.2f} | {s_mf['Sharpe']:>18.2f}")
    print(f"{'Annual Vol (%)':<20} | {s_b['Vol']:>18.2f} | {s_mf['Vol']:>18.2f}")
    print(f"{'Trade Count':<20} | {s_b['Trades']:>18} | {s_mf['Trades']:>18}")
    print("="*80)
    
    # 파일 저장
    with open('comparative_advantage_evaluation.txt', 'w', encoding='utf-8') as f:
        f.write("Comparative Advantage Evaluation Analysis Report\n")
        f.write("="*80 + "\n")
        f.write(f"{'Metric':<20} | {'Optimized Basic':<20} | {'Multifactor CDF'}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Final Value ($)':<20} | {s_b['Final']:>18,.0f} | {s_mf['Final']:>18,.0f}\n")
        f.write(f"{'CAGR (%)':<20} | {s_b['CAGR']:>18.2f} | {s_mf['CAGR']:>18.2f}\n")
        f.write(f"{'MDD (%)':<20} | {s_b['MDD']:>18.2f} | {s_mf['MDD']:>18.2f}\n")
        f.write(f"{'Sharpe Ratio':<20} | {s_b['Sharpe']:>18.2f} | {s_mf['Sharpe']:>18.2f}\n")
        f.write(f"{'Annual Vol (%)':<20} | {s_b['Vol']:>18.2f} | {s_mf['Vol']:>18.2f}\n")
        f.write(f"{'Trade Count':<20} | {s_b['Trades']:>18} | {s_mf['Trades']:>18}\n")
        f.write("="*80 + "\n")

    # 차션 시각화
    plt.figure(figsize=(12, 8))
    plt.plot(res_b / res_b.iloc[0], label=f'Basic (MDD {s_b["MDD"]:.1f}%)', alpha=0.8)
    plt.plot(res_mf / res_mf.iloc[0], label=f'Multifactor (MDD {s_mf["MDD"]:.1f}%)', alpha=0.8)
    plt.title("Comparative Real-World Performance: Basic vs User Multifactor")
    plt.ylabel("Relative Value (Initial=1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('comparative_advantage_chart.png')
    print("\nAnalysis Complete. Files: comparative_advantage_evaluation.txt, comparative_advantage_chart.png")

if __name__ == "__main__":
    main()
