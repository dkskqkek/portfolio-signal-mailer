# -*- coding: utf-8 -*-
"""
최적화 융합 모델 검증 (Optimized Fusion Model Validation)
- 기존 모델 (Optimized Basic: 15d MA / 30d Vol)
- 융합 모델 (Fusion: Basic Danger + Multifactor Validator)
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

# --- 1. 설정 및 데이터 수집 ---
TRAIN_START = "2015-01-01"
EVAL_START = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

TICKERS = ['SPY', 'SCHD', 'QQQ', 'JEPI', 'GLD', '^KS200']
MACRO = {'VIX': '^VIX'}

COST = 0.0011 # 슬리피지 + 수수료
TAX_RATE = 0.22
TAX_FREE_LIMIT_USD = 2500000 / 1450

def fetch_data():
    prices = {}
    divs = {}
    for t in TICKERS + [MACRO['VIX']]:
        data = yf.download(t, start=TRAIN_START, end=END_DATE, progress=False, auto_adjust=False, multi_level_index=False)
        prices[t] = data['Close']
        if t in TICKERS:
            divs[t] = data['Dividends'] if 'Dividends' in data.columns else pd.Series(0, index=data.index)
    return pd.DataFrame(prices).fillna(method='ffill').dropna(), pd.DataFrame(divs).fillna(0)

# --- 2. 시그널 엔진 ---

def get_basic_danger(df_p):
    """기존 최적화 기본 시그널 (15d MA / 30d Vol)"""
    returns = np.log(df_p['SPY'] / df_p['SPY'].shift(1))
    ma15 = returns.rolling(15).mean()
    vol30 = returns.rolling(30).std()
    
    signals = []
    ma_hist = []
    vol_hist = []
    
    for i in range(len(df_p)):
        m, v = ma15.iloc[i], vol30.iloc[i]
        if np.isnan(m) or np.isnan(v):
            signals.append(False)
            continue
        
        if len(ma_hist) > 100:
            p25 = np.percentile(ma_hist, 25)
            p65 = np.percentile(vol_hist, 65)
            signals.append((m < p25) or (v > p65))
        else:
            signals.append(False)
        
        ma_hist.append(m)
        vol_hist.append(v)
    return pd.Series(signals, index=df_p.index)

def get_multifactor_score(df_p, lookback=126):
    """사용자 멀티팩터 CDF 스코어링"""
    close = df_p['SPY']
    vix = df_p['^VIX']
    
    # 지표 산정 (동일 로직)
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema_dist = (close - ema200) / ema200
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain/loss.replace(0, np.nan))).fillna(100)
    
    def get_score(series, inv=False):
        z = (series - series.rolling(lookback).mean()) / (series.rolling(lookback).std() + 1e-6)
        score = pd.Series(norm.cdf(z), index=series.index) * 100
        return 100 - score if inv else score
        
    score = (get_score(ema_dist, inv=True) * 0.2 + 
             get_score(rsi, inv=True) * 0.4 + 
             get_score(vix, inv=False) * 0.4).rolling(3).mean()
    
    return score

# --- 3. 백테스트 시뮬레이션 ---

def simulate(df_p, df_d, signal_series):
    initial_cash = 100000
    shares = {t: 0.0 for t in TICKERS}
    
    eval_idx = df_p.index.get_loc(df_p.index[df_p.index >= EVAL_START][0])
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
        p, d = df_p.iloc[i], df_d.iloc[i]
        sig = signal_series.iloc[i] # 1: Danger, 0: Normal
        
        # 세금 및 배당 (동일 로직)
        if date.year != prev_year:
            if yearly_profit > TAX_FREE_LIMIT_USD:
                shares = {t: v * (1 - ((yearly_profit - TAX_FREE_LIMIT_USD)*TAX_RATE / sum(shares[k]*p[k] for k in TICKERS))) for t, v in shares.items()}
            yearly_profit = 0
            prev_year = date.year
            
        for t in TICKERS:
            if shares[t] > 0 and d[t] > 0:
                shares[t] += (shares[t] * d[t] * (1-0.154) * (1-COST)) / p[t]
        
        # 스위칭
        if sig == 1 and current_mode == 0:
            val = shares['QQQ'] * p['QQQ'] * (1-COST)
            shares['JEPI'] = (val * (1-COST)) / p['JEPI']
            shares['QQQ'], current_mode, trade_count = 0, 1, trade_count + 1
        elif sig == 0 and current_mode == 1:
            val = shares['JEPI'] * p['JEPI'] * (1-COST)
            shares['QQQ'] = (val * (1-COST)) / p['QQQ']
            shares['JEPI'], current_mode, trade_count = 0, 0, trade_count + 1
            
        val_curr = sum(shares[t] * p[t] for t in TICKERS)
        history.append(val_curr)
        if len(history) > 1: yearly_profit += (history[-1] - history[-2])
        
    return pd.Series(history, index=df_p.index[eval_idx:]), trade_count

# --- 4. 메인 실행 ---

def main():
    print("Fetching data and calculating signals...")
    df_p, df_d = fetch_data()
    
    # 시그널 생성
    basic_danger = get_basic_danger(df_p)
    mf_score = get_multifactor_score(df_p)
    
    # 융합 로직: Basic이 Danger라고 할 때, MF 점수가 40점 이하(위험 동의)일 때만 탈출
    # 60점 이상이면 확실한 정상 상태로 복귀
    fusion_signals = []
    state = 0
    for i in range(len(df_p)):
        b_danger = basic_danger.iloc[i]
        score = mf_score.iloc[i]
        
        if state == 0:
            if b_danger and score <= 40: state = 1 # 이중 화인 시 탈출
        else:
            if not b_danger or score >= 60: state = 0 # 둘 중 하나라도 회복 시 신속 복귀
        fusion_signals.append(state)
        
    fusion_series = pd.Series(fusion_signals, index=df_p.index)
    
    print("Running Backtests...")
    res_b, trades_b = simulate(df_p, df_d, basic_danger.astype(int))
    res_f, trades_f = simulate(df_p, df_d, fusion_series)
    
    def analyze(series, trades):
        ret = series.pct_change().dropna()
        cagr = (series.iloc[-1] / series.iloc[0]) ** (252/len(series)) - 1
        mdd = (series / series.cummax() - 1).min()
        sharpe = (ret.mean() * 252) / (ret.std() * np.sqrt(252))
        return {'Final': series.iloc[-1], 'CAGR': cagr*100, 'MDD': mdd*100, 'Sharpe': sharpe, 'Trades': trades}

    s_b = analyze(res_b, trades_b)
    s_f = analyze(res_f, trades_f)
    
    print("\n" + "="*80)
    print(f"{'Metric':<20} | {'Optimized Basic':<20} | {'Fusion Model (Hybrid)'}")
    print("-" * 80)
    print(f"{'Final Value ($)':<20} | {s_b['Final']:>20,.0f} | {s_f['Final']:>20,.0f}")
    print(f"{'CAGR (%)':<20} | {s_b['CAGR']:>20.2f} | {s_f['CAGR']:>20.2f}")
    print(f"{'MDD (%)':<20} | {s_b['MDD']:>20.2f} | {s_f['MDD']:>20.2f}")
    print(f"{'Sharpe Ratio':<20} | {s_b['Sharpe']:>20.2f} | {s_f['Sharpe']:>20.2f}")
    print(f"{'Trade Count':<20} | {s_b['Trades']:>20} | {s_f['Trades']:>20}")
    print("="*80)
    
    with open('fusion_performance_report.txt', 'w', encoding='utf-8') as f:
        f.write("Optimized Fusion Model Performance Report\n")
        f.write("="*80 + "\n")
        f.write(f"{'Metric':<20} | {'Optimized Basic':<20} | {'Fusion Model (Hybrid)'}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Final Value ($)':<20} | {s_b['Final']:>20,.0f} | {s_f['Final']:>20,.0f}\n")
        f.write(f"{'CAGR (%)':<20} | {s_b['CAGR']:>20.2f} | {s_f['CAGR']:>20.2f}\n")
        f.write(f"{'MDD (%)':<20} | {s_b['MDD']:>20.2f} | {s_f['MDD']:>20.2f}\n")
        f.write(f"{'Sharpe Ratio':<20} | {s_b['Sharpe']:>20.2f} | {s_f['Sharpe']:>20.2f}\n")
        f.write(f"{'Trade Count':<20} | {s_b['Trades']:>20} | {s_f['Trades']:>20}\n")
        f.write("="*80 + "\n")

    plt.figure(figsize=(12, 8))
    plt.plot(res_b / res_b.iloc[0], label='Basic (M1)', alpha=0.7)
    plt.plot(res_f / res_f.iloc[0], label='Fusion (M1+M2)', linewidth=2)
    plt.title("Performance Comparison: Optimized Basic vs Fusion Model")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fusion_performance_chart.png')
    print("\nFiles generated: fusion_performance_report.txt, fusion_performance_chart.png")

if __name__ == "__main__":
    main()
