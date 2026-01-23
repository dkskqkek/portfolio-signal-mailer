# -*- coding: utf-8 -*-
"""
MA Strategy Comparison Verification Script
Compares:
1. Fusion (Original)
2. Hybrid (Fusion + MA200 Floor)
3. Faber (10-month SMA)
4. Golden Cross (50/200 SMA)

Asset Allocation:
- Normal: QQQ 100% (Simplified for signal test)
- Danger: BIL 100% (Cash Equivalent)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime

# --- 1. Data Fetching ---
def fetch_data(start_date="2008-01-01"):
    tickers = ['SPY', 'QQQ', 'BIL', '^VIX']
    print(f"Downloading {tickers}...")
    # Use yf.download with multi_level_index=False for simpler DataFrame
    data = yf.download(tickers, start=start_date, progress=False, auto_adjust=True, multi_level_index=False)
    
    # yfinance returns 'Close' directly if only one ticker, but with multiple it's a MultiIndex or flat
    # With multi_level_index=False, it should be flat tickers if only one price type is requested?
    # Actually, yfinance download usually returns (Price, Ticker) structure.
    
    if isinstance(data.columns, pd.MultiIndex):
        # The default is (Price, Ticker)
        df = data['Close']
    else:
        # If somehow it's flat tickers (only Close requested)
        df = data
        
    return df.fillna(method='ffill').dropna()

# --- 2. Logic Implementations ---

def calculate_ma_signals(df):
    spy = df['SPY']
    vix = df['^VIX']
    
    # 1. Base Indicators
    ema200 = spy.ewm(span=200, adjust=False).mean()
    ema_dist = (spy - ema200) / ema200
    
    delta = spy.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain/loss.replace(0, np.nan))).fillna(100)
    
    def get_cdf(series, lookback=126, inv=False):
        rolling_mean = series.rolling(lookback).mean()
        rolling_std = series.rolling(lookback).std() + 1e-6
        z = (series - rolling_mean) / rolling_std
        score = pd.Series(norm.cdf(z.values), index=series.index) * 100
        return 100 - score if inv else score

    mf_score = (get_cdf(ema_dist, inv=True)*0.2 + get_cdf(rsi, inv=True)*0.4 + get_cdf(vix, inv=False)*0.4).rolling(3).mean()
    
    ret = np.log(spy / spy.shift(1))
    ma15 = ret.rolling(15).mean()
    vol30 = ret.rolling(30).std()
    
    ma15_rank = ma15.rolling(450).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    vol30_rank = vol30.rolling(450).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    sma200 = spy.rolling(200).mean()
    sma50 = spy.rolling(50).mean()
    
    monthly_spy = spy.resample('M').last()
    ma10mo = monthly_spy.rolling(10).mean()
    ma10mo_daily = ma10mo.reindex(spy.index, method='ffill')

    signals = pd.DataFrame(index=df.index)
    signals['Fusion'] = 0
    signals['Hybrid'] = 0
    signals['Faber'] = 0
    signals['GoldenCross'] = 0
    
    # Vectorized signals as much as possible to avoid slow loop and errors
    m_danger = (ma15_rank < 0.25) | (vol30_rank > 0.65)
    signals['Fusion'] = np.where(m_danger & (mf_score <= 40), 1, 0)
    signals['Hybrid'] = np.where(signals['Fusion'] == 1, 1, np.where(spy < sma200, 1, 0))
    signals['Faber'] = np.where(spy < ma10mo_daily, 1, 0)
    signals['GoldenCross'] = np.where(sma50 < sma200, 1, 0)
            
    return signals

def backtest(df, signals):
    strategies = ['Fusion', 'Hybrid', 'Faber', 'GoldenCross', 'BuyHold']
    equity = pd.DataFrame(index=df.index, columns=strategies, dtype=float)
    
    # Find first index where all signals are NOT Nan (post 450 burn-in)
    start_idx = 450
    equity.iloc[start_idx] = 100.0
    
    risky_ret = df['QQQ'].pct_change().fillna(0)
    safe_ret = df['BIL'].pct_change().fillna(0)
    
    for t in range(start_idx + 1, len(df)):
        for s in ['Fusion', 'Hybrid', 'Faber', 'GoldenCross']:
             sig = signals[s].iloc[t-1]
             ret = safe_ret.iloc[t] if sig == 1 else risky_ret.iloc[t]
             equity[s].iloc[t] = equity[s].iloc[t-1] * (1 + ret)
        
        equity['BuyHold'].iloc[t] = equity['BuyHold'].iloc[t-1] * (1 + risky_ret.iloc[t])
    
    return equity.iloc[start_idx:]

# --- 3. Main ---
if __name__ == "__main__":
    df = fetch_data(start_date="2008-01-01")
    print(f"Data fetched: {len(df)} rows. Columns: {df.columns.tolist()}")
    
    signals = calculate_ma_signals(df)
    results = backtest(df, signals)
    
    # Evaluate from 2010 to ensure stability
    eval_df = results.loc['2010-01-01':]
    
    if len(eval_df) == 0:
        print("Error: No data in eval period.")
    else:
        stats = []
        for col in eval_df.columns:
            total_ret = (eval_df[col].iloc[-1] / eval_df[col].iloc[0]) - 1
            cagr = (eval_df[col].iloc[-1] / eval_df[col].iloc[0]) ** (252/len(eval_df)) - 1
            mdd = (eval_df[col] / eval_df[col].cummax() - 1).min()
            stats.append({'Strategy': col, 'Total Return': total_ret*100, 'CAGR': cagr*100, 'MDD': mdd*100})
            
        stats_df = pd.DataFrame(stats).set_index('Strategy')
        print("\nPerformance Summary (2010-Now):")
        print(stats_df.round(2))
        
        plt.figure(figsize=(12, 7))
        for col in eval_df.columns:
            plt.plot(eval_df[col] / eval_df[col].iloc[0], label=col)
        plt.title("MA Strategy Comparison (QQQ/BIL, 2010-Present)")
        plt.yscale('log')
        plt.ylabel("Normalized Equity (Log Scale)")
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        plt.savefig('d:/gg/ma_strategy_comparison.png')
        print("\nChart saved to d:/gg/ma_strategy_comparison.png")
