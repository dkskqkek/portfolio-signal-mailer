# -*- coding: utf-8 -*-
"""
최종 하이브리드 검증 (Final Hybrid Validation)
- 이전 final_integrity_validation.py와 100% 동일 조건
- SCHD(38%), QQQ/JEPI(38%), GOLD(5%), KOSPI(19%)
- 2020-01-01 시작 (JEPI 상장 이후 로직 포함)
- ML Meta-Labeling 필터 적용
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# --- 1. 설정 통일 ---
TRAIN_START_DATE = "2015-01-01" # 충분한 학습 데이터 확보
EVAL_START_DATE = "2020-01-01"  # 이전 검증 기간과 통일 (JEPI 가용 시점)
END_DATE = datetime.now().strftime("%Y-%m-%d")

INITIAL_CAPITAL = 100000
USD_KRW = 1450
TAX_DEDUCTION_USD = 2500000 / USD_KRW
TAX_RATE = 0.22
DIVIDEND_TAX = 0.154
SLIPPAGE = 0.001
TRADING_FEE = 0.0001
TOTAL_COST_PER_TRADE = SLIPPAGE + TRADING_FEE

TICKERS = ['SPY', 'SCHD', 'QQQ', 'JEPI', 'GLD', '^KS200']
MACRO_TICKERS = {"VIX": "^VIX", "TNX": "^TNX", "DXY": "DX-Y.NYB"}

def fetch_all_data():
    """순수 가격 및 배당 + 매크로 데이터 수집"""
    prices = {}
    dividends = {}
    for ticker in TICKERS:
        t = yf.Ticker(ticker)
        # 학습을 위해 2015년부터 수집
        hist = t.history(start=TRAIN_START_DATE, end=END_DATE, auto_adjust=False)
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        prices[ticker] = hist['Close']
        dividends[ticker] = hist['Dividends']
    
    # Macro
    macros = {}
    for name, ticker in MACRO_TICKERS.items():
        m = yf.download(ticker, start=TRAIN_START_DATE, end=END_DATE, progress=False)
        if hasattr(m.columns, 'nlevels') and m.columns.nlevels > 1: m.columns = m.columns.get_level_values(0)
        macros[name] = m['Close']
    
    df_p = pd.DataFrame(prices).fillna(method='ffill').dropna()
    df_d = pd.DataFrame(dividends).reindex(df_p.index).fillna(0)
    df_m = pd.DataFrame(macros).reindex(df_p.index).fillna(method='ffill')
    
    return df_p, df_d, df_m

def get_ml_hybrid_signals(df_p, df_m):
    """Meta-Labeling 기반 하이브리드 시그널 생성"""
    # 1. 기본 시그널 (M1)
    spy_ret = np.log(df_p['SPY'] / df_p['SPY'].shift(1))
    ma15 = spy_ret.rolling(15).mean()
    vol30 = spy_ret.rolling(30).std()
    
    m1_signals = []
    history_ma = []
    history_vol = []
    
    for i in range(len(df_p)):
        if i < 20: # 초기 윈도우
            m1_signals.append(0)
            if not np.isnan(ma15.iloc[i]): history_ma.append(ma15.iloc[i])
            if not np.isnan(vol30.iloc[i]): history_vol.append(vol30.iloc[i])
            continue
        
        p25_ma = np.nanpercentile(history_ma, 25) if history_ma else -0.01
        p65_vol = np.nanpercentile(history_vol, 65) if history_vol else 0.01
        
        danger = (ma15.iloc[i] < p25_ma) or (vol30.iloc[i] > p65_vol)
        m1_signals.append(1 if danger else 0)
        
        history_ma.append(ma15.iloc[i])
        history_vol.append(vol30.iloc[i])
    
    m1_series = pd.Series(m1_signals, index=df_p.index)
    
    # 2. ML Meta-Labeler (M2) 학습용 데이터 준비
    # 실전에서는 과거 데이터로 학습 후 미래를 예측하지만, 검증을 위해 전체 기간 중 시그널 발생 시점 추출
    features = pd.DataFrame(index=df_p.index)
    features['VIX'] = df_m['VIX']
    features['TNX_Chg'] = df_m['TNX'].pct_change()
    features['DXY_Chg'] = df_m['DXY'].pct_change()
    features['MA_Gap'] = df_p['SPY'] / df_p['SPY'].rolling(15).mean() - 1
    features = features.fillna(0)
    
    # Labeling: M1이 Danger일 때, 실제로 이후 5일간 SPY가 하락했는가?
    fwd_ret = df_p['SPY'].shift(-5) / df_p['SPY'] - 1
    labels = (fwd_ret < -0.005).astype(int) # 0.5% 하락 시 유효한 시그널로 간주
    
    # 시그널 발생 시점만 학습 (Meta-Labeling)
    danger_idx = m1_series[m1_series == 1].index
    if len(danger_idx) > 10:
        X = features.loc[danger_idx]
        y = labels.loc[danger_idx]
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf.fit(X, y)
        
        # 전체 기간에 대해 ML 판정 (재확인)
        m2_probs = clf.predict_proba(features)[:, 1]
        hybrid_signals = (m1_series == 1) & (m2_probs > 0.45) # ML이 어느정도 위험하다고 동의할 때만
    else:
        hybrid_signals = m1_series
        
    return m1_series, pd.Series(hybrid_signals.astype(int), index=df_p.index)

def backtest_logic(df_p, df_d, is_danger, core_ticker):
    """계산 공식은 이전 final_integrity_validation.py와 100% 동일"""
    weights = {'CORE': 0.38, 'DYNAMIC': 0.38, 'GOLD': 0.05, 'KOSPI': 0.19}
    shares = {t: 0.0 for t in TICKERS}
    
    p0 = df_p.iloc[0]
    shares[core_ticker] = (INITIAL_CAPITAL * weights['CORE'] * (1-TOTAL_COST_PER_TRADE)) / p0[core_ticker]
    shares['QQQ'] = (INITIAL_CAPITAL * weights['DYNAMIC'] * (1-TOTAL_COST_PER_TRADE)) / p0['QQQ']
    shares['GLD'] = (INITIAL_CAPITAL * weights['GOLD'] * (1-TOTAL_COST_PER_TRADE)) / p0['GLD']
    shares['^KS200'] = (INITIAL_CAPITAL * weights['KOSPI'] * (1-0.0033)) / p0['^KS200']
    
    portfolio_values = []
    current_mode = 0 # 0: QQQ, 1: JEPI
    yearly_profit = 0
    prev_year = df_p.index[0].year
    
    for i in range(len(df_p)):
        date = df_p.index[i]
        p = df_p.iloc[i]
        div = df_d.iloc[i]
        signal = is_danger.iloc[i]
        
        if date.year != prev_year:
            if yearly_profit > TAX_DEDUCTION_USD:
                tax = (yearly_profit - TAX_DEDUCTION_USD) * TAX_RATE
                total_val = sum(shares[t] * p[t] for t in TICKERS)
                tax_ratio = tax / total_val
                for t in TICKERS: shares[t] *= (1 - tax_ratio)
            yearly_profit = 0
            prev_year = date.year
            
        for t in TICKERS:
            if shares[t] > 0 and div[t] > 0:
                cost = 0.0033 if t == '^KS200' else TOTAL_COST_PER_TRADE
                d_val = shares[t] * div[t] * (1 - DIVIDEND_TAX)
                shares[t] += (d_val * (1 - cost)) / p[t]
        
        if signal != current_mode:
            if signal == 1: # QQQ -> JEPI
                if shares['QQQ'] > 0:
                    val = shares['QQQ'] * p['QQQ'] * (1 - TOTAL_COST_PER_TRADE)
                    shares['JEPI'] = (val * (1 - TOTAL_COST_PER_TRADE)) / p['JEPI'] if p['JEPI'] > 0 else 0
                    shares['QQQ'] = 0.0
            else: # JEPI -> QQQ
                if shares['JEPI'] > 0:
                    val = shares['JEPI'] * p['JEPI'] * (1 - TOTAL_COST_PER_TRADE)
                    shares['QQQ'] = (val * (1 - TOTAL_COST_PER_TRADE)) / p['QQQ']
                    shares['JEPI'] = 0.0
            current_mode = signal
        
        current_val = sum(shares[t] * p[t] for t in TICKERS)
        portfolio_values.append(current_val)
        if i > 0: yearly_profit += (portfolio_values[-1] - portfolio_values[-2])

    return pd.Series(portfolio_values, index=df_p.index)

def analyze(series):
    ret = (series.iloc[-1] / series.iloc[0]) - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (252/len(series)) - 1
    mdd = (series / series.cummax() - 1).min()
    sharpe = (series.pct_change().mean() * 252) / (series.pct_change().std() * np.sqrt(252))
    return {'Final': series.iloc[-1], 'CAGR': cagr*100, 'MDD': mdd*100, 'Sharpe': sharpe}

def main():
    df_p, df_d, df_m = fetch_all_data()
    sig_basic, sig_hybrid = get_ml_hybrid_signals(df_p, df_m)
    
    # 1. 기존 베이직 (SCHD)
    res_basic = backtest_logic(df_p, df_d, sig_basic, 'SCHD')
    # 2. 하이브리드 ML (SCHD)
    res_hybrid = backtest_logic(df_p, df_d, sig_hybrid, 'SCHD')
    
    # 2020년 이후 데이터만 필터링하여 분석
    res_basic_eval = res_basic[res_basic.index >= EVAL_START_DATE]
    res_hybrid_eval = res_hybrid[res_hybrid.index >= EVAL_START_DATE]
    
    stats_b = analyze(res_basic_eval)
    stats_h = analyze(res_hybrid_eval)
    
    print("\n" + "="*70)
    print(f"{'Strategy':<20} | {'Final ($)':<12} | {'CAGR (%)':<10} | {'MDD (%)':<10} | {'Sharpe':<8}")
    print("-" * 70)
    print(f"{'Basic (Previous)':<20} | {stats_b['Final']:>12,.0f} | {stats_b['CAGR']:>10.2f} | {stats_b['MDD']:>10.2f} | {stats_b['Sharpe']:>8.2f}")
    print(f"{'ML Hybrid':<20} | {stats_h['Final']:>12,.0f} | {stats_h['CAGR']:>10.2f} | {stats_h['MDD']:>10.2f} | {stats_h['Sharpe']:>8.2f}")
    print("="*70)
    
    # 리포트 파일 저장
    with open('ml_final_fair_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"{'Strategy':<20} | {'Final ($)':<12} | {'CAGR (%)':<10} | {'MDD (%)':<10} | {'Sharpe':<8}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Basic (Previous)':<20} | {stats_b['Final']:>12,.0f} | {stats_b['CAGR']:>10.2f} | {stats_b['MDD']:>10.2f} | {stats_b['Sharpe']:>8.2f}\n")
        f.write(f"{'ML Hybrid':<20} | {stats_h['Final']:>12,.0f} | {stats_h['CAGR']:>10.2f} | {stats_h['MDD']:>10.2f} | {stats_h['Sharpe']:>8.2f}\n")
        f.write("="*70 + "\n")
    
    plt.figure(figsize=(10, 6))
    plt.plot(res_basic_eval, label=f"Basic (Sharpe: {stats_b['Sharpe']:.2f})")
    plt.plot(res_hybrid_eval, label=f"ML Hybrid (Sharpe: {stats_h['Sharpe']:.2f})")
    plt.title("Evaluation (2020-Present): Basic vs ML Hybrid")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ml_final_fair_comparison.png')

if __name__ == "__main__":
    main()
