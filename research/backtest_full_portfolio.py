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

def get_signal_at_step(i, df, strategy_name, ma_long=150):
    start_idx = max(0, i - 450)
    window = df.iloc[start_idx:i+1]
    
    ma15_win = window['ma15'].dropna()
    vol30_win = window['vol30'].dropna()
    
    if len(ma15_win) < 100: return 1
    
    ma_thresh = np.percentile(ma15_win, 25)
    vol_thresh = np.percentile(vol30_win, 65)
    
    latest_ma = df['ma15'].iloc[i]
    latest_vol = df['vol30'].iloc[i]
    mf_score = df['mf_score'].iloc[i]
    
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

def run_portfolio_backtest(df, strategy_name, cost=0.002):
    # Initial Equity components
    eq_total = [1.0]
    
    # Portfolio Weights
    w_schd = 0.38
    w_tactical = 0.38
    w_kospi = 0.19
    w_gold = 0.05
    
    # Tracking components separately for accurate cost calculation
    comp_schd = 1.0 * w_schd
    comp_tactical = 1.0 * w_tactical
    comp_kospi = 1.0 * w_kospi
    comp_gold = 1.0 * w_gold
    
    pos_tactical = 1 # 1: QQQ, 0: JEPI
    trades = 0
    
    for i in range(len(df)):
        if i < 450:
            eq_total.append(comp_schd + comp_tactical + comp_kospi + comp_gold)
            continue
            
        # 1. Update Signals
        sig = get_signal_at_step(i, df, strategy_name)
        
        # 2. Check for trade in Tactical portion
        if sig != pos_tactical:
            comp_tactical *= (1 - cost) # Apply cost only to tactical portion
            pos_tactical = sig
            trades += 1
            
        # 3. Apply Daily Returns
        comp_schd *= (1 + df['SCHD_ret'].iloc[i])
        comp_kospi *= (1 + df['KS200_ret'].iloc[i])
        comp_gold *= (1 + df['GLD_ret'].iloc[i])
        
        tactical_ret = df['QQQ_ret'].iloc[i] if pos_tactical == 1 else df['JEPI_ret'].iloc[i]
        comp_tactical *= (1 + tactical_ret)
        
        eq_total.append(comp_schd + comp_tactical + comp_kospi + comp_gold)
        
    return pd.Series(eq_total[1:], index=df.index), trades

# Data Prep
data_path = 'd:/gg/comprehensive_ma_data.csv'
df = pd.read_csv(data_path, index_col=0, parse_dates=True)
df.columns = [c.replace('^', '').upper() for c in df.columns]

# Pre-calculate common indicators
df['QQQ_ret'] = df['QQQ'].pct_change().fillna(0)
df['JEPI_ret'] = df['JEPI'].pct_change().fillna(0)
df['SCHD_ret'] = df['SCHD'].pct_change().fillna(0)
df['KS200_ret'] = df['KS200'].pct_change().fillna(0)
df['GLD_ret'] = df['GLD'].pct_change().fillna(0)
df['QQQ_MA150'] = df['QQQ'].rolling(150).mean()

log_ret_spy = np.log(df['SPY'] / df['SPY'].shift(1))
df['ma15'] = log_ret_spy.rolling(15).mean()
df['vol30'] = log_ret_spy.rolling(30).std()
df['mf_score'] = calculate_mf_score(df['SPY'], df['VIX'])

# Slice from JEPI Inception
test_df = df['2020-06-01':].copy()

# Run Portfolios
eq_bh, _ = run_portfolio_backtest(test_df, 'B&H') # Always QQQ in tactical
eq_sma, t_sma = run_portfolio_backtest(test_df, 'SMA 150')
eq_fusion, t_fusion = run_portfolio_backtest(test_df, 'Original Fusion')
eq_hybrid, t_hybrid = run_portfolio_backtest(test_df, 'Optimized Hybrid')

# Metrics
def get_metrics(eq, trades):
    cagr = (eq.iloc[-1]**(252/len(eq)) - 1) * 100
    mdd = (eq / eq.cummax() - 1).min() * 100
    vol = eq.pct_change().std() * np.sqrt(252)
    sharpe = (eq.pct_change().mean() / eq.pct_change().std()) * np.sqrt(252)
    return cagr, mdd, vol, sharpe, trades

res_bh = get_metrics(eq_bh, 0)
res_sma = get_metrics(eq_sma, t_sma)
res_fusion = get_metrics(eq_fusion, t_fusion)
res_hybrid = get_metrics(eq_hybrid, t_hybrid)

# Save Results
results = pd.DataFrame({
    'Strategy': ['Portfolio (Static QQQ)', 'Portfolio (SMA 150)', 'Portfolio (Original Fusion)', 'Portfolio (Optimized Hybrid)'],
    'CAGR (%)': [res_bh[0], res_sma[0], res_fusion[0], res_hybrid[0]],
    'MDD (%)': [res_bh[1], res_sma[1], res_fusion[1], res_hybrid[1]],
    'Vol (%)': [res_bh[2]*100, res_sma[2]*100, res_fusion[2]*100, res_hybrid[2]*100],
    'Sharpe': [res_bh[3], res_sma[3], res_fusion[3], res_hybrid[3]],
    'Trades': [0, t_sma, t_fusion, t_hybrid]
})
results.to_csv('d:/gg/full_portfolio_results.csv', index=False)

# Chart
plt.figure(figsize=(12, 7))
plt.plot(eq_bh, label=f'Static QQQ (CAGR: {res_bh[0]:.1f}%, MDD: {res_bh[1]:.1f}%)', color='gray', alpha=0.6)
plt.plot(eq_sma, label=f'SMA 150 Tactical (CAGR: {res_sma[0]:.1f}%, MDD: {res_sma[1]:.1f}%)', color='blue')
plt.plot(eq_fusion, label=f'Fusion Tactical (CAGR: {res_fusion[0]:.1f}%, MDD: {res_fusion[1]:.1f}%)', color='green')
plt.plot(eq_hybrid, label=f'Hybrid Tactical (CAGR: {res_hybrid[0]:.1f}%, MDD: {res_hybrid[1]:.1f}%)', color='red', linewidth=2)

plt.title('Real Account Portfolio Backtest (2020-06 ~ Present)\nWeights: SCHD 38%, QQQ/JEPI 38%, KOSPI 19%, GLD 5%')
plt.xlabel('Date')
plt.ylabel('Normalized Equity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('d:/gg/full_portfolio_comparison.png')
print("Full portfolio backtest complete.")
