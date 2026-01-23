import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# --- PROTOCOL: Strict Standardized Backtest ---

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

def get_signal_at_step(i, df, strategy_name, vix_threshold=30):
    # PROTOCOL: 450d Rolling Window Only (No Look-ahead)
    start_idx = max(0, i - 450)
    window = df.iloc[start_idx:i] # Use data UP TO YESTERDAY
    
    if len(window) < 300: return 1 # Warm-up
    
    ma15_win = window['ma15'].dropna()
    vol30_win = window['vol30'].dropna()
    
    ma_thresh = np.percentile(ma15_win, 25)
    vol_thresh = np.percentile(vol30_win, 65)
    
    latest_ma = df['ma15'].iloc[i]
    latest_vol = df['vol30'].iloc[i]
    mf_score = df['mf_score'].iloc[i]
    current_px = df['QQQ'].iloc[i]
    ma150 = df['QQQ_MA150'].iloc[i]
    vix = df['VIX'].iloc[i]
    
    # Logic Definitions
    
    if strategy_name == 'SMA 150 (Pure)':
        return 0 if current_px < ma150 else 1
        
    elif strategy_name == 'SMA 150 + VIX Filter':
        # Defensive if Price < SMA150 OR VIX > Threshold (Panic)
        if current_px < ma150 or vix > vix_threshold:
            return 0
        return 1
        
    elif strategy_name == 'Optimized Hybrid':
        m1_danger = (latest_ma < ma_thresh) or (latest_vol > vol_thresh)
        fusion_danger = (m1_danger and mf_score <= 40)
        floor_breached = current_px < ma150
        return 0 if (fusion_danger or floor_breached) else 1
    
    return 1

def run_vix_experiment(df, strategy_name, vix_threshold=30, cost=0.002):
    # Standard Portfolio Weights
    w_schd = 0.38
    w_tactical = 0.38
    w_kospi = 0.19
    w_gold = 0.05
    
    comp_schd = 1.0 * w_schd
    comp_tactical = 1.0 * w_tactical
    comp_kospi = 1.0 * w_kospi
    comp_gold = 1.0 * w_gold
    
    pos_tactical = 1
    trades = 0
    total_eq = [1.0]
    
    for i in range(len(df)):
        if i < 450:
            total_eq.append(comp_schd + comp_tactical + comp_kospi + comp_gold)
            continue
            
        sig = get_signal_at_step(i, df, strategy_name, vix_threshold)
        
        if sig != pos_tactical:
            comp_tactical *= (1 - cost)
            pos_tactical = sig
            trades += 1
            
        comp_schd *= (1 + df['SCHD_ret'].iloc[i])
        comp_kospi *= (1 + df['KS200_ret'].iloc[i])
        comp_gold *= (1 + df['GLD_ret'].iloc[i])
        
        is_jepi_avail = df.index[i] >= pd.Timestamp('2020-06-01')
        defensive_ret = df['JEPI_ret'].iloc[i] if is_jepi_avail else df['BIL_ret'].iloc[i]
        
        tactical_ret = df['QQQ_ret'].iloc[i] if pos_tactical == 1 else defensive_ret
        comp_tactical *= (1 + tactical_ret)
        
        total_eq.append(comp_schd + comp_tactical + comp_kospi + comp_gold)
        
    return pd.Series(total_eq[1:], index=df.index), trades

# Data Prep
df = pd.read_csv('d:/gg/comprehensive_ma_data.csv', index_col=0, parse_dates=True)
df.columns = [c.replace('^', '').upper() for c in df.columns]

df['QQQ_ret'] = df['QQQ'].pct_change().fillna(0)
df['JEPI_ret'] = df['JEPI'].pct_change().fillna(0)
df['BIL_ret'] = df['BIL'].pct_change().fillna(0)
df['SCHD_ret'] = df['SCHD'].pct_change().fillna(0)
df['KS200_ret'] = df['KS200'].pct_change().fillna(0)
df['GLD_ret'] = df['GLD'].pct_change().fillna(0)
df['QQQ_MA150'] = df['QQQ'].rolling(150).mean()

log_ret_spy = np.log(df['SPY'] / df['SPY'].shift(1))
df['ma15'] = log_ret_spy.rolling(15).mean()
df['vol30'] = log_ret_spy.rolling(30).std()
df['mf_score'] = calculate_mf_score(df['SPY'], df['VIX'])

# Test Period
test_df = df['2020-06-01':].copy()

# Run Experiments
eq_sma, t_sma = run_vix_experiment(test_df, 'SMA 150 (Pure)')
eq_vix30, t_vix30 = run_vix_experiment(test_df, 'SMA 150 + VIX Filter', vix_threshold=30)
eq_vix35, t_vix35 = run_vix_experiment(test_df, 'SMA 150 + VIX Filter', vix_threshold=35)
eq_hybrid, t_hybrid = run_vix_experiment(test_df, 'Optimized Hybrid')

# Reporting
def get_metrics(eq, trades):
    ret = eq.pct_change().dropna()
    cagr = (eq.iloc[-1]**(252/len(eq)) - 1) * 100
    mdd = (eq / eq.cummax() - 1).min() * 100
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252)
    return cagr, mdd, sharpe, trades

m_sma = get_metrics(eq_sma, t_sma)
m_vix30 = get_metrics(eq_vix30, t_vix30)
m_vix35 = get_metrics(eq_vix35, t_vix35)
m_hybrid = get_metrics(eq_hybrid, t_hybrid)

results = pd.DataFrame({
    'Strategy': ['SMA 150 (Pure)', 'SMA 150 + VIX>30', 'SMA 150 + VIX>35', 'Optimized Hybrid'],
    'CAGR (%)': [m_sma[0], m_vix30[0], m_vix35[0], m_hybrid[0]],
    'MDD (%)': [m_sma[1], m_vix30[1], m_vix35[1], m_hybrid[1]],
    'Sharpe': [m_sma[2], m_vix30[2], m_vix35[2], m_hybrid[2]],
    'Trades': [t_sma, t_vix30, t_vix35, t_hybrid]
})
results.to_csv('d:/gg/vix_optimization_results.csv', index=False)
print(results)
