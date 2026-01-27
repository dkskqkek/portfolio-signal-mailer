import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def calculate_mf_score(spy_close, vix_close, lookback=126):
    ema200 = spy_close.ewm(span=200, adjust=False).mean()
    ema_dist = (spy_close - ema200) / ema200
    
    delta = spy_close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain/loss.replace(0, np.nan))).fillna(100)
    
    def get_z_score(series):
        m = series.rolling(lookback).mean()
        s = series.rolling(lookback).std()
        return (series - m) / (s + 1e-6)
    
    s_trend = 100 - norm.cdf(get_z_score(ema_dist)) * 100
    s_mom = 100 - norm.cdf(get_z_score(rsi)) * 100
    s_vol = norm.cdf(get_z_score(vix_close)) * 100
    
    return s_trend * 0.2 + s_mom * 0.4 + s_vol * 0.4

def run_backtest(df, strategy_name, cost=0.002):
    equity = [1.0]
    position = 1 # 1: QQQ, 0: BIL
    trades = 0
    
    signals = []
    
    for i in range(len(df)):
        if i < 200: # Warm up
            equity.append(equity[-1])
            signals.append(1)
            continue
            
        current_signal = 1
        
        # Strategy Logic
        if strategy_name == 'SMA 150':
            ma150 = df['QQQ_MA150'].iloc[i]
            if df['QQQ'].iloc[i] < ma150:
                current_signal = 0
        
        elif strategy_name == 'Original Fusion':
            if df['m1_danger'].iloc[i] and df['mf_score'].iloc[i] <= 40:
                current_signal = 0
        
        elif strategy_name == 'Optimized Hybrid':
            fusion_danger = df['m1_danger'].iloc[i] and df['mf_score'].iloc[i] <= 40
            ma150 = df['QQQ_MA150'].iloc[i]
            if df['QQQ'].iloc[i] < ma150 or fusion_danger:
                current_signal = 0
        
        # Trade Execution
        if current_signal != position:
            equity[-1] *= (1 - cost)
            position = current_signal
            trades += 1
            
        # Daily Return
        ret = df['QQQ_ret'].iloc[i] if position == 1 else df['BIL_ret'].iloc[i]
        equity.append(equity[-1] * (1 + ret))
        signals.append(current_signal)
        
    return pd.Series(equity[1:], index=df.index), trades, signals

# Load and Prepare Data
data_path = 'd:/gg/comprehensive_ma_data.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
df.columns = [c.replace('^', '').upper() for c in df.columns]

# Pre-calculate Indicators
df['QQQ_ret'] = df['QQQ'].pct_change()
df['JEPI_ret'] = df['JEPI'].pct_change().fillna(0)
df['QQQ_MA150'] = df['QQQ'].rolling(150).mean()

# Sentinel & Validator signals calculated day-by-day (mimicking real-time)
log_ret = np.log(df['SPY'] / df['SPY'].shift(1))
df['ma15'] = log_ret.rolling(15).mean()
df['vol30'] = log_ret.rolling(30).std()
df['mf_score'] = calculate_mf_score(df['SPY'], df['VIX'])

def get_signal_at_step(i, df, strategy_name, ma_long=150):
    # Mimic SignalDetector(days_back=450)
    start_idx = max(0, i - 450)
    window = df.iloc[start_idx:i+1]
    
    ma15_win = window['ma15'].dropna()
    vol30_win = window['vol30'].dropna()
    
    if len(ma15_win) < 100: return 1 # Warmup
    
    ma_thresh = np.percentile(ma15_win, 25)
    vol_thresh = np.percentile(vol30_win, 65)
    
    latest_ma = df['ma15'].iloc[i]
    latest_vol = df['vol30'].iloc[i]
    mf_score = df['mf_score'].iloc[i]
    
    # Re-apply exact production logic
    m1_danger = (latest_ma < ma_thresh) or (latest_vol > vol_thresh)
    
    if strategy_name == 'SMA 150':
        return 0 if df['QQQ'].iloc[i] < df['QQQ_MA150'].iloc[i] else 1
        
    elif strategy_name == 'Original Fusion':
        return 0 if (m1_danger and mf_score <= 40) else 1
        
    elif strategy_name == 'Optimized Hybrid':
        fusion_danger = (m1_danger and mf_score <= 40)
        floor_breached = df['QQQ'].iloc[i] < df['QQQ_MA150'].iloc[i]
        return 0 if (fusion_danger or floor_breached) else 1
    
    return 1

def run_faithful_backtest(df, strategy_name, cost=0.002):
    eq = [1.0]
    pos = 1
    trades = 0
    signals = []
    
    for i in range(len(df)):
        if i < 450: # Enough data for 450d window
            eq.append(eq[-1])
            signals.append(1)
            continue
            
        sig = sig = get_signal_at_step(i, df, strategy_name)
        
        if sig != pos:
            eq[-1] *= (1 - cost)
            pos = sig
            trades += 1
            
        ret = df['QQQ_ret'].iloc[i] if pos == 1 else df['JEPI_ret'].iloc[i]
        eq.append(eq[-1] * (1 + ret))
        signals.append(pos)
        
    return pd.Series(eq[1:], index=df.index), trades

# Run Comparison (From JEPI Launch Date)
test_df = df['2020-06-01':].copy() 
eq_bh, t_bh = run_faithful_backtest(test_df, 'B&H', cost=0)
eq_sma, t_sma = run_faithful_backtest(test_df, 'SMA 150')
eq_fusion, t_fusion = run_faithful_backtest(test_df, 'Original Fusion')
eq_hybrid, t_hybrid = run_faithful_backtest(test_df, 'Optimized Hybrid')

# Metrics and Plotting (rest of the script)

def get_metrics(eq, trades):
    cagr = (eq.iloc[-1]**(252/len(eq)) - 1) * 100
    mdd = (eq / eq.cummax() - 1).min() * 100
    sharpe = (eq.pct_change().mean() / eq.pct_change().std()) * np.sqrt(252)
    return cagr, mdd, sharpe, trades

m_bh = get_metrics(eq_bh, 0)
m_sma = get_metrics(eq_sma, t_sma)
m_fusion = get_metrics(eq_fusion, t_fusion)
m_hybrid = get_metrics(eq_hybrid, t_hybrid)

# Visualization
plt.figure(figsize=(14, 8))
plt.plot(eq_bh, label=f'B&H QQQ (CAGR: {m_bh[0]:.1f}%, MDD: {m_bh[1]:.1f}%)', color='gray', alpha=0.5)
plt.plot(eq_sma, label=f'SMA 150 (CAGR: {m_sma[0]:.1f}%, MDD: {m_sma[1]:.1f}%, Trades: {t_sma})', color='blue')
plt.plot(eq_fusion, label=f'Original Fusion (CAGR: {m_fusion[0]:.1f}%, MDD: {m_fusion[1]:.1f}%, Trades: {t_fusion})', color='green')
plt.plot(eq_hybrid, label=f'Optimized Hybrid (CAGR: {m_hybrid[0]:.1f}%, MDD: {m_hybrid[1]:.1f}%, Trades: {t_hybrid})', color='red', linewidth=2)

plt.title('Tri-Strategy Performance Comparison (Including 0.2% Trade Costs)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.savefig('d:/gg/tri_strategy_comparison.png')
print("Comparison chart saved to d:/gg/tri_strategy_comparison.png")

# Report Data
results = {
    'Strategy': ['B&H QQQ', 'SMA 150', 'Original Fusion', 'Optimized Hybrid'],
    'CAGR (%)': [m_bh[0], m_sma[0], m_fusion[0], m_hybrid[0]],
    'MDD (%)': [m_bh[1], m_sma[1], m_fusion[1], m_hybrid[1]],
    'Sharpe': [m_bh[2], m_sma[2], m_fusion[2], m_hybrid[2]],
    'Trades': [0, t_sma, t_fusion, t_hybrid]
}
pd.DataFrame(results).to_csv('d:/gg/tri_strategy_results.csv', index=False)
print("Results CSV saved to d:/gg/tri_strategy_results.csv")
