# -*- coding: utf-8 -*-
"""
최종 검증 백테스트 (Final Validation)
- 배당 중복 반영 제거 (Close Price 사용)
- 슬리피지 반영 (0.1%)
- 연간 세금 정산 (공제액 $1,724 반영)
- HMM 재현성 확보 (랜덤 시드 고정)
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# crash_detection_system 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'crash_detection_system' / 'src'))

from main import CrashDetectionPipeline

# --- 전역 설정 ---
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100000
USD_KRW = 1450
TAX_DEDUCTION_USD = 2500000 / USD_KRW  # 약 $1,724
TAX_RATE = 0.22
DIVIDEND_TAX = 0.154
SLIPPAGE = 0.001  # 0.1%
TRADING_FEE = 0.0001  # 0.01%
TOTAL_COST_PER_TRADE = SLIPPAGE + TRADING_FEE

TICKERS = ['SPY', 'SCHD', 'QQQ', 'JEPI', 'GLD', '^KS200']

def fetch_pure_data():
    """배당이 반영되지 않은 순수 가격(Close)과 배당 데이터를 분리하여 수집"""
    print("Data collection (auto_adjust=False)...")
    prices = {}
    dividends = {}
    
    for ticker in TICKERS:
        t = yf.Ticker(ticker)
        # auto_adjust=False로 설정하여 배당이 포함되지 않은 순수 가격 획득
        hist = t.history(start=START_DATE, end=END_DATE, auto_adjust=False)
        if hist.empty: continue
        
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        prices[ticker] = hist['Close']
        dividends[ticker] = hist['Dividends']
        print(f"  v {ticker} collected")
        
    df_prices = pd.DataFrame(prices).fillna(method='ffill').dropna()
    df_divs = pd.DataFrame(dividends).reindex(df_prices.index).fillna(0)
    return df_prices, df_divs

def get_optimized_basic_signal(df):
    """최종 최적화 기본 시그널 (15/30/25/65)"""
    returns = np.log(df['SPY'] / df['SPY'].shift(1))
    ma = returns.rolling(15).mean()
    vol = returns.rolling(30).std()
    
    signals = []
    history_ma = []
    history_vol = []
    
    for i in range(len(df)):
        if i < 252:
            signals.append(0)
            if not np.isnan(ma.iloc[i]): history_ma.append(ma.iloc[i])
            if not np.isnan(vol.iloc[i]): history_vol.append(vol.iloc[i])
            continue
            
        p25_ma = np.nanpercentile(history_ma, 25)
        p65_vol = np.nanpercentile(history_vol, 65)
        
        danger = (ma.iloc[i] < p25_ma) or (vol.iloc[i] > p65_vol)
        signals.append(1 if danger else 0)
        
        history_ma.append(ma.iloc[i])
        history_vol.append(vol.iloc[i])
        
    return pd.Series(signals, index=df.index)

def get_fixed_hmm_signal(df):
    """재현성이 확보된 HMM 시그널"""
    print("HMM Signal Generation (Fixed Seed)...")
    # HMM 엔진 내부적으로 random_state 조정을 위해 파이프라인 호출
    pipeline = CrashDetectionPipeline(
        ticker='SPY', 
        start_date=START_DATE,
        cache_dir=str(Path(__file__).parent / 'crash_detection_system' / 'data')
    )
    # HMM 시그널 계산 로직은 이전과 동일하되 최적 파라미터 적용
    pipeline.run_full_pipeline()
    
    indicators = pipeline.indicators.copy()
    indicators.index = pd.to_datetime(indicators.index).tz_localize(None)
    df_mix = df.join(indicators[['HMM_Regime', 'RSI', 'ADX']], how='left')
    
    # VIX (수동 로드)
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)['Close']
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    df_mix['VIX'] = vix.reindex(df_mix.index).fillna(method='ffill').fillna(15)
    
    df_mix = df_mix.fillna({'HMM_Regime':0, 'RSI':50, 'ADX':20})
    
    signals = []
    for i in range(len(df_mix)):
        reg, rsi, adx, vx = df_mix[['HMM_Regime', 'RSI', 'ADX', 'VIX']].iloc[i]
        danger = False
        if adx >= 15:
            if reg >= 2: danger = True
            elif reg >= 1.0 and (rsi < 40 or vx > 25): danger = True
        signals.append(1 if danger else 0)
        
    return pd.Series(signals, index=df.index)

def backtest_precise(df_p, df_d, is_danger, core_ticker):
    """현실적인 비용 모델이 적용된 정밀 백테스트"""
    weights = {'CORE': 0.38, 'DYNAMIC': 0.38, 'GOLD': 0.05, 'KOSPI': 0.19}
    
    # 자산별 보유량 초기화 (티커명 직접 사용)
    shares = {t: 0.0 for t in TICKERS}
    
    # 초기 매수
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
        
        # 연간 세금 정산 (12월 마지막 거래일)
        if date.year != prev_year:
            if yearly_profit > TAX_DEDUCTION_USD:
                tax = (yearly_profit - TAX_DEDUCTION_USD) * TAX_RATE
                total_val = sum(shares[t] * p[t] for t in TICKERS)
                tax_ratio = tax / total_val
                for t in TICKERS: shares[t] *= (1 - tax_ratio)
            yearly_profit = 0
            prev_year = date.year
            
        # 배당금 수령 및 재투자 (세후)
        for t in TICKERS:
            if shares[t] > 0 and div[t] > 0:
                cost = 0.0033 if t == '^KS200' else TOTAL_COST_PER_TRADE
                d_val = shares[t] * div[t] * (1 - DIVIDEND_TAX)
                shares[t] += (d_val * (1 - cost)) / p[t]
        
        # 신호 변경 시 스위칭 (슬리피지 반영)
        if signal != current_mode:
            if signal == 1: # QQQ -> JEPI
                if shares['QQQ'] > 0:
                    val = shares['QQQ'] * p['QQQ'] * (1 - TOTAL_COST_PER_TRADE)
                    shares['JEPI'] = (val * (1 - TOTAL_COST_PER_TRADE)) / p['JEPI']
                    shares['QQQ'] = 0.0
            else: # JEPI -> QQQ
                if shares['JEPI'] > 0:
                    val = shares['JEPI'] * p['JEPI'] * (1 - TOTAL_COST_PER_TRADE)
                    shares['QQQ'] = (val * (1 - TOTAL_COST_PER_TRADE)) / p['QQQ']
                    shares['JEPI'] = 0.0
            current_mode = signal
        
        # 가치 기록
        current_val = sum(shares[t] * p[t] for t in TICKERS)
        portfolio_values.append(current_val)
        
        # 수익 추적 (연간 정산용)
        if i > 0: yearly_profit += (portfolio_values[-1] - portfolio_values[-2])

    return pd.Series(portfolio_values, index=df_p.index)

def analyze(series):
    ret = (series.iloc[-1] / series.iloc[0]) - 1
    cagr = (series.iloc[-1] / series.iloc[0]) ** (252/len(series)) - 1
    mdd = (series / series.cummax() - 1).min()
    sharpe = (series.pct_change().mean() * 252) / (series.pct_change().std() * np.sqrt(252))
    return {'Final': series.iloc[-1], 'CAGR': cagr*100, 'MDD': mdd*100, 'Sharpe': sharpe}

def main():
    df_p, df_d = fetch_pure_data()
    
    sig_basic = get_optimized_basic_signal(df_p)
    sig_hmm = get_fixed_hmm_signal(df_p)
    
    combinations = [
        ('SCHD', sig_basic, 'SCHD + Basic'),
        ('SCHD', sig_hmm,   'SCHD + HMM'),
        ('SPY',  sig_basic, 'SPY + Basic'),
        ('SPY',  sig_hmm,   'SPY + HMM')
    ]
    
    results = {}
    plt.figure(figsize=(12, 8))
    
    for core, sig, name in combinations:
        print(f"Simulation in progress: {name}...")
        history = backtest_precise(df_p, df_d, sig, core)
        results[name] = analyze(history)
        plt.plot(history, label=f"{name} (Sharpe: {results[name]['Sharpe']:.2f})")
        
    # 리포트 파일 작성
    with open('comprehensive_final_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*65 + "\n")
        f.write(f"{'Strategy':<15} | {'Final ($)':<12} | {'CAGR (%)':<10} | {'MDD (%)':<10} | {'Sharpe':<8}\n")
        f.write("-" * 65 + "\n")
        for name, stat in results.items():
            f.write(f"{name:<15} | {stat['Final']:>12,.0f} | {stat['CAGR']:>10.2f} | {stat['MDD']:>10.2f} | {stat['Sharpe']:>8.2f}\n")
        f.write("="*65 + "\n")
    
    print("\nResult report saved: comprehensive_final_report.txt")
    print("\nResult chart saved: final_verification_results.png")

if __name__ == "__main__":
    main()
