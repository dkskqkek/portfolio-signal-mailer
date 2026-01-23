# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. Data Loading ---
def get_data():
    path = 'd:/gg/comprehensive_ma_data.csv'
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # Standardize column names
    df.columns = [c.replace('^', '').upper() for c in df.columns]
    return df

# --- 2. Logic Implementation ---
def calculate_signals(df):
    spy = df['SPY']
    qqq = df['QQQ']
    vix = df['VIX'] if 'VIX' in df.columns else df.iloc[:, 3] # Fallback if missing/misnamed
    
    # Fusion Logic
    ema200 = spy.ewm(span=200, adjust=False).mean()
    ema_dist = (spy - ema200) / ema200
    delta = spy.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain/loss.replace(0, np.nan))).fillna(100)
    
    def get_cdf(series, lookback=126, inv=False):
        z = (series - series.rolling(lookback).mean()) / (series.rolling(lookback).std() + 1e-6)
        score = pd.Series(norm.cdf(z.values), index=series.index) * 100
        return 100 - score if inv else score
        
    mf_score = (get_cdf(ema_dist, inv=True)*0.2 + get_cdf(rsi, inv=True)*0.4 + get_cdf(vix, inv=False)*0.4).rolling(3).mean()
    
    ret = np.log(spy / spy.shift(1))
    ma15 = ret.rolling(15).mean()
    vol30 = ret.rolling(30).std()
    ma15_rank = ma15.rolling(450).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    vol30_rank = vol30.rolling(450).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    m_danger = (ma15_rank < 0.25) | (vol30_rank > 0.65)
    fusion_signal = np.where(m_danger & (mf_score <= 40), 1, 0)
    
    # SMA 150 Hard Floor Logic (Hybrid)
    sma150 = qqq.rolling(150).mean()
    hybrid_signal = np.where(fusion_signal == 1, 1, np.where(qqq < sma150, 1, 0))
    
    signals = pd.DataFrame(index=df.index)
    signals['Original'] = fusion_signal
    signals['SMA150_Hybrid'] = hybrid_signal
    return signals

# --- 3. Professional Backtest with Costs ---
def run_real_backtest(df, signals, cost=0.002): # 0.2% cost per flip
    results = {}
    
    risky_ret = df['QQQ'].pct_change().fillna(0)
    safe_ret = df['BIL'].pct_change().fillna(0)
    
    start_date = '2010-01-01'
    idx = df.loc[start_date:].index
    
    for strategy in ['Original', 'SMA150_Hybrid', 'BuyHold']:
        cash = 100.0
        equity = [cash]
        state = 0 # 0: Risky, 1: Safe
        trades = []
        
        current_signals = signals[strategy] if strategy != 'BuyHold' else pd.Series(0, index=df.index)
        
        for i in range(1, len(idx)):
            date = idx[i]
            prev_date = idx[i-1]
            
            # Prediction from yesterday
            signal = current_signals.loc[prev_date]
            
            # Check for Switch
            if signal != state:
                cash *= (1 - cost) # Apply commission
                trades.append({'date': date, 'from': 'Risky' if state==0 else 'Safe', 'to': 'Safe' if state==0 else 'Risky'})
                state = signal
            
            # Apply Return
            ret = safe_ret.loc[date] if state == 1 else risky_ret.loc[date]
            cash *= (1 + ret)
            equity.append(cash)
            
        results[strategy] = {
            'equity': pd.Series(equity, index=idx),
            'trades': trades
        }
    return results

# --- 4. Main & Reporting ---
if __name__ == "__main__":
    df = get_data()
    signals = calculate_signals(df)
    results = run_real_backtest(df, signals)
    
    # 1. Performance Table
    stats = []
    for s in results:
        eq = results[s]['equity']
        cagr = (eq.iloc[-1]/eq.iloc[0])**(252/len(eq)) - 1
        mdd = (eq / eq.cummax() - 1).min()
        trade_count = len(results[s]['trades'])
        stats.append({'Strategy': s, 'Final Value': eq.iloc[-1], 'CAGR (%)': cagr*100, 'MDD (%)': mdd*100, 'Trades': trade_count})
    
    print("\n--- Real Environment Performance Comparison (Cost 0.2%) ---")
    print(pd.DataFrame(stats).set_index('Strategy').round(2))
    
    # 2. Trade Log Comparison (Last 5 trades)
    print("\n--- Recent Trade Comparison ---")
    for s in ['Original', 'SMA150_Hybrid']:
        print(f"\n[{s}] Last 5 trades:")
        for t in results[s]['trades'][-5:]:
            print(f"- {t['date'].strftime('%Y-%m-%d')}: {t['from']} -> {t['to']}")

    # 3. Chart
    plt.figure(figsize=(14, 8))
    for s in results:
        plt.plot(results[s]['equity'], label=s, linewidth=2 if s != 'BuyHold' else 1)
    plt.title("Real Environment Comparison: Original Fusion vs SMA 150 Hybrid", fontsize=15)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('d:/gg/real_env_comparison.png')
    print("\nChart saved to d:/gg/real_env_comparison.png")
