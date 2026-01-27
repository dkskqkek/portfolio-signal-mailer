# -*- coding: utf-8 -*-
"""
SCHD vs 배당성장 ETF 경쟁군 정밀 비교 백테스트
- 대상: SCHD, VIG (안정성), DGRO (성장밸런스), DGRW (품질성장), FDVV (알파/기술)
- 전략: 융합 모델 (Sentinel + Validator) 적용
- 기간: 2020-05-21 (JEPI 상장) ~ 현재
- 비용: 연간 양도세(22%), 배당세(15.4%), 슬리피지(0.1%) 반영
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt

# --- 1. 설정 및 데이터 수집 ---
TRAIN_START = "2015-01-01"
# JEPI 상장 이후 안정적인 비교를 위해 시작일을 2020-06-01로 설정
EVAL_START = "2020-05-22" 
END_DATE = datetime.now().strftime("%Y-%m-%d")

# 비교할 코어 자산들
CORES = ['SCHD', 'VIG', 'DGRO', 'DGRW', 'FDVV']
# 보조 자산 (전략 고정)
FIXED = ['QQQ', 'JEPI', 'GLD', '^KS200']
MACRO = {'VIX': '^VIX'}

COST = 0.0011 # 슬리피지 + 수수료 (0.11%)
TAX_RATE = 0.22
USD_KRW = 1450 # 일관된 계산을 위해 고정 환율 사용
TAX_FREE_LIMIT_USD = 2500000 / USD_KRW

def fetch_data():
    tickers = list(set(CORES + FIXED + ['SPY'] + [MACRO['VIX']]))
    prices = {}
    divs = {}
    for t in tickers:
        print(f"Downloading {t}...")
        data = yf.download(t, start=TRAIN_START, end=END_DATE, progress=False, auto_adjust=False, multi_level_index=False)
        prices[t] = data['Close']
        if t not in [MACRO['VIX'], 'SPY']:
            divs[t] = data['Dividends'] if 'Dividends' in data.columns else pd.Series(0, index=data.index)
    return pd.DataFrame(prices).fillna(method='ffill').dropna(), pd.DataFrame(divs).fillna(0)

# --- 2. 시그널 엔진 (융합 모델 복제) ---

def get_fusion_signals(df_p):
    # Sentinel (M1)
    returns = np.log(df_p['SPY'] / df_p['SPY'].shift(1))
    ma15 = returns.rolling(15).mean()
    vol30 = returns.rolling(30).std()
    
    # Validator (M2)
    close = df_p['SPY']
    vix = df_p['^VIX']
    ema200 = close.ewm(span=200, adjust=False).mean()
    ema_dist = (close - ema200) / ema200
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain/loss.replace(0, np.nan))).fillna(100)
    
    def get_cdf_score(series, lookback=126, inv=False):
        z = (series - series.rolling(lookback).mean()) / (series.rolling(lookback).std() + 1e-6)
        score = pd.Series(norm.cdf(z), index=series.index) * 100
        return 100 - score if inv else score
        
    mf_score = (get_cdf_score(ema_dist, inv=True) * 0.2 + 
                get_cdf_score(rsi, inv=True) * 0.4 + 
                get_cdf_score(vix, inv=False) * 0.4).rolling(3).mean()
    
    fusion_signals = []
    state = 0
    ma_hist, vol_hist = [], []
    
    for i in range(len(df_p)):
        m, v = ma15.iloc[i], vol30.iloc[i]
        score = mf_score.iloc[i]
        
        # M1 Danger 결정
        m1_danger = False
        if len(ma_hist) > 100:
            p25 = np.percentile(ma_hist, 25)
            p65 = np.percentile(vol_hist, 65)
            m1_danger = (m < p25) or (v > p65)
        
        # 융합 로직
        if state == 0:
            if m1_danger and score <= 40: state = 1
        else:
            if not m1_danger or score >= 60: state = 0
            
        fusion_signals.append(state)
        if not np.isnan(m) and not np.isnan(v):
            ma_hist.append(m)
            vol_hist.append(v)
            
    return pd.Series(fusion_signals, index=df_p.index)

# --- 3. 백테스트 시뮬레이션 ---

def backtest_etf(core_ticker, df_p, df_d, signals):
    initial_cash = 100000
    eval_start_date = df_p[df_p.index >= EVAL_START].index[0]
    eval_idx = df_p.index.get_loc(eval_start_date)
    
    p0 = df_p.iloc[eval_idx]
    
    # 포트폴리오 구성: Core(38%), QQQ/JEPI(38%), Gold(5%), KS200(19%)
    shares = {
        core_ticker: (initial_cash * 0.38 * (1-COST)) / p0[core_ticker],
        'QQQ': (initial_cash * 0.38 * (1-COST)) / p0['QQQ'],
        'GLD': (initial_cash * 0.05 * (1-COST)) / p0['GLD'],
        '^KS200': (initial_cash * 0.19 * (1-COST)) / p0['^KS200'],
        'JEPI': 0.0
    }
    
    history = []
    current_mode = 0 # 0: QQQ, 1: JEPI
    yearly_profit = 0
    prev_year = p0.name.year
    trade_count = 0
    
    all_active_tickers = [core_ticker, 'QQQ', 'JEPI', 'GLD', '^KS200']
    
    for i in range(eval_idx, len(df_p)):
        date = df_p.index[i]
        p, d = df_p.iloc[i], df_d.iloc[i]
        sig = signals.iloc[i]
        
        # 연간 세금 정산
        if date.year != prev_year:
            if yearly_profit > TAX_FREE_LIMIT_USD:
                tax = (yearly_profit - TAX_FREE_LIMIT_USD) * TAX_RATE
                total_val = sum(shares[t] * p[t] for t in all_active_tickers)
                tax_ratio = tax / total_val
                for t in all_active_tickers:
                    shares[t] *= (1 - tax_ratio)
            yearly_profit = 0
            prev_year = date.year
            
        # 배당 재투자
        for t in all_active_tickers:
            if shares[t] > 0 and d[t] > 0:
                shares[t] += (shares[t] * d[t] * (1-0.154) * (1-COST)) / p[t]
        
        # 전략적 스위칭 (QQQ <-> JEPI)
        if sig == 1 and current_mode == 0:
            val = shares['QQQ'] * p['QQQ'] * (1-COST)
            shares['JEPI'] = (val * (1-COST)) / p['JEPI']
            shares['QQQ'], current_mode, trade_count = 0, 1, trade_count + 1
        elif sig == 0 and current_mode == 1:
            val = shares['JEPI'] * p['JEPI'] * (1-COST)
            shares['QQQ'] = (val * (1-COST)) / p['QQQ']
            shares['JEPI'], current_mode, trade_count = 0, 0, trade_count + 1
            
        curr_val = sum(shares[t] * p[t] for t in all_active_tickers)
        history.append(curr_val)
        if len(history) > 1:
            yearly_profit += (history[-1] - history[-2])
            
    return pd.Series(history, index=df_p.index[eval_idx:]), trade_count

# --- 4. 메인 실행 ---

def main():
    df_p, df_d = fetch_data()
    fusion_signals = get_fusion_signals(df_p)
    
    results = {}
    
    print("\nStarting Comparative Backtests...")
    for core in CORES:
        print(f"Evaluating Core: {core}...")
        res_series, trades = backtest_etf(core, df_p, df_d, fusion_signals)
        
        # 성과 분석
        rets = res_series.pct_change().dropna()
        cagr = (res_series.iloc[-1] / res_series.iloc[0]) ** (252/len(res_series)) - 1
        mdd = (res_series / res_series.cummax() - 1).min()
        sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
        
        results[core] = {
            'Series': res_series,
            'Final': res_series.iloc[-1],
            'CAGR': cagr * 100,
            'MDD': mdd * 100,
            'Sharpe': sharpe,
            'Trades': trades
        }
        
    # 결과 출력 및 저장
    print("\n" + "="*95)
    print(f"{'ETF Ticker':<10} | {'Final Value ($)':<18} | {'CAGR (%)':<10} | {'MDD (%)':<10} | {'Sharpe':<8} | {'Trades':<6}")
    print("-" * 95)
    sorted_cores = sorted(results.items(), key=lambda x: x[1]['CAGR'], reverse=True)
    
    with open('schd_alternatives_report.txt', 'w', encoding='utf-8') as f:
        f.write("SCHD vs Best Dividend Growth ETFs: Performance Comparison\n")
        f.write("Strategy: Fusion Model (Sentinel + Validator)\n")
        f.write("Evaluation Period: 2020-05-22 to Present (JEPI Incl.)\n")
        f.write("="*95 + "\n")
        f.write(f"{'ETF Ticker':<10} | {'Final Value ($)':<18} | {'CAGR (%)':<10} | {'MDD (%)':<10} | {'Sharpe':<8} | {'Trades':<6}\n")
        f.write("-" * 95 + "\n")
        
        for name, m in sorted_cores:
            line = f"{name:<10} | {m['Final']:>18,.0f} | {m['CAGR']:>10.2f} | {m['MDD']:>10.2f} | {m['Sharpe']:>8.2f} | {m['Trades']:>6}"
            print(line)
            f.write(line + "\n")
        
    # 시각화
    plt.figure(figsize=(14, 8))
    for core in CORES:
        norm_series = results[core]['Series'] / results[core]['Series'].iloc[0]
        plt.plot(norm_series, label=f"{core} (CAGR {results[core]['CAGR']:.1f}%)")
        
    plt.title(f"Portfolio Performance with different Core Assets (2020 - Present)\n[Strategy: Fusion Hybrid | Costs & Taxes Included]", fontsize=14)
    plt.ylabel("Normalized Wealth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('schd_alternatives_chart.png', dpi=120)
    print("\nFiles generated: schd_alternatives_report.txt, schd_alternatives_chart.png")

if __name__ == "__main__":
    main()
