# -*- coding: utf-8 -*-
"""
SCHD vs 범용 고샤프 ETF (QUAL, COWZ, MOAT, SPGP, VGT) 정밀 비교
- 대상: SCHD, QUAL (Quality), COWZ (Cash Flow), MOAT (Moat), SPGP (GARP), VGT (Tech), VTI (Market), FDVV (Prev Winner)
- 전략: 융합 모델 (Sentinel + Validator) 적용
- 기간: 2020-05-22 (JEPI 상장) ~ 현재
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
EVAL_START = "2020-05-22" 
END_DATE = datetime.now().strftime("%Y-%m-%d")

# 비교할 범용 코어 자산들
CORES = ['SCHD', 'QUAL', 'COWZ', 'MOAT', 'SPGP', 'VGT', 'VTI', 'FDVV']
# 보조 자산 (전략 고정)
FIXED = ['QQQ', 'JEPI', 'GLD', '^KS200']
MACRO = {'VIX': '^VIX'}

COST = 0.0011 # 0.11%
TAX_RATE = 0.22
USD_KRW = 1450
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

# --- 2. 시그널 엔진 ---

def get_fusion_signals(df_p):
    returns = np.log(df_p['SPY'] / df_p['SPY'].shift(1))
    ma15 = returns.rolling(15).mean()
    vol30 = returns.rolling(30).std()
    
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
        m1_danger = False
        if len(ma_hist) > 100:
            p25 = np.percentile(ma_hist, 25)
            p65 = np.percentile(vol_hist, 65)
            m1_danger = (m < p25) or (v > p65)
        
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
    
    shares = {
        core_ticker: (initial_cash * 0.38 * (1-COST)) / p0[core_ticker],
        'QQQ': (initial_cash * 0.38 * (1-COST)) / p0['QQQ'],
        'GLD': (initial_cash * 0.05 * (1-COST)) / p0['GLD'],
        '^KS200': (initial_cash * 0.19 * (1-COST)) / p0['^KS200'],
        'JEPI': 0.0
    }
    
    history = []
    current_mode = 0 
    yearly_profit = 0
    prev_year = p0.name.year
    trade_count = 0
    all_active_tickers = [core_ticker, 'QQQ', 'JEPI', 'GLD', '^KS200']
    
    for i in range(eval_idx, len(df_p)):
        date = df_p.index[i]
        p, d = df_p.iloc[i], df_d.iloc[i]
        sig = signals.iloc[i]
        
        if date.year != prev_year:
            if yearly_profit > TAX_FREE_LIMIT_USD:
                tax = (yearly_profit - TAX_FREE_LIMIT_USD) * TAX_RATE
                total_val = sum(shares[t] * p[t] for t in all_active_tickers)
                tax_ratio = tax / total_val
                for t in all_active_tickers:
                    shares[t] *= (1 - tax_ratio)
            yearly_profit = 0
            prev_year = date.year
            
        for t in all_active_tickers:
            if shares[t] > 0 and d[t] > 0:
                shares[t] += (shares[t] * d[t] * (1-0.154) * (1-COST)) / p[t]
        
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
        if len(history) > 1: yearly_profit += (history[-1] - history[-2])
            
    return pd.Series(history, index=df_p.index[eval_idx:]), trade_count

# --- 4. 메인 실행 ---

def main():
    df_p, df_d = fetch_data()
    fusion_signals = get_fusion_signals(df_p)
    
    results = {}
    print("\nStarting Broad ETF Comparison...")
    for core in CORES:
        print(f"Evaluating Core: {core}...")
        res_series, trades = backtest_etf(core, df_p, df_d, fusion_signals)
        rets = res_series.pct_change().dropna()
        cagr = (res_series.iloc[-1] / res_series.iloc[0]) ** (252/len(res_series)) - 1
        mdd = (res_series / res_series.cummax() - 1).min()
        sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
        results[core] = {'Series': res_series, 'Final': res_series.iloc[-1], 'CAGR': cagr * 100, 'MDD': mdd * 100, 'Sharpe': sharpe, 'Trades': trades}
        
    print("\n" + "="*100)
    print(f"{'ETF Ticker':<10} | {'Category':<15} | {'Final ($)':<15} | {'CAGR (%)':<10} | {'MDD (%)':<10} | {'Sharpe':<8}")
    print("-" * 100)
    
    categories = {
        'SCHD': 'Div Growth', 'QUAL': 'Quality', 'COWZ': 'Cash Flow', 'MOAT': 'Wide Moat',
        'SPGP': 'GARP', 'VGT': 'Tech Growth', 'VTI': 'Total Market', 'FDVV': 'Div/Tech'
    }
    
    sorted_cores = sorted(results.items(), key=lambda x: x[1]['Sharpe'], reverse=True)
    
    with open('diverse_etf_report.txt', 'w', encoding='utf-8') as f:
        f.write("Diverse ETF Performance Comparison (High Sharpe Focus)\n")
        f.write("Strategy: Fusion Model (Hybrid Defense)\n")
        f.write("Costs: Slippage, Dividends Tax, Capital Gains Tax Included\n")
        f.write("="*100 + "\n")
        f.write(f"{'ETF Ticker':<10} | {'Category':<15} | {'Final ($)':<15} | {'CAGR (%)':<10} | {'MDD (%)':<10} | {'Sharpe':<8}\n")
        f.write("-" * 100 + "\n")
        for name, m in sorted_cores:
            line = f"{name:<10} | {categories[name]:<15} | {m['Final']:>15,.0f} | {m['CAGR']:>10.2f} | {m['MDD']:>10.2f} | {m['Sharpe']:>8.2f}"
            print(line)
            f.write(line + "\n")
            
    plt.figure(figsize=(14, 8))
    for core in CORES:
        plt.plot(results[core]['Series'] / results[core]['Series'].iloc[0], label=f"{core} ({categories[core]})")
    plt.title("Wealth Evolution: Diverse Core Assets with Fusion Strategy", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('diverse_etf_chart.png', dpi=120)

if __name__ == "__main__":
    main()
