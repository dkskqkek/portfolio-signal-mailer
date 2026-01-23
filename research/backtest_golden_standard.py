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

def get_signal_at_step(i, df, strategy_name):
    if i == 0: return 1
    
    past_idx = i - 1
    past_px = df['QQQ'].iloc[past_idx]
    
    if strategy_name == 'SMA 150':
        past_ma = df['QQQ'].rolling(150).mean().iloc[past_idx]
        if np.isnan(past_ma): return 1
        return 0 if past_px < past_ma else 1
    elif strategy_name == 'SMA 185':
        past_ma = df['QQQ'].rolling(185).mean().iloc[past_idx]
        if np.isnan(past_ma): return 1
        return 0 if past_px < past_ma else 1
    elif strategy_name == 'Original Fusion':
        return 0 if (df['ma15_danger'].iloc[past_idx] and df['mf_score'].iloc[past_idx] <= 40) else 1
    elif strategy_name == 'Optimized Hybrid':
        past_ma150 = df['QQQ'].rolling(150).mean().iloc[past_idx]
        floor_breached = past_px < past_ma150
        fusion_danger = (df['ma15_danger'].iloc[past_idx] and df['mf_score'].iloc[past_idx] <= 40)
        return 0 if (fusion_danger or floor_breached) else 1
    return 1

def run_golden_backtest(df, strategy_name, cost=0.002):
    w_schd, w_tactical, w_kospi, w_gold = 0.38, 0.38, 0.19, 0.05
    comp_schd, comp_tactical, comp_kospi, comp_gold = w_schd, w_tactical, w_kospi, w_gold
    
    pos_tactical, trades, total_eq = 1, 0, []
    
    for i in range(len(df)):
        # Daily Return Update (Today's market movement)
        comp_schd *= (1 + df['SCHD_ret'].iloc[i])
        comp_kospi *= (1 + df['KS200_ret'].iloc[i])
        comp_gold *= (1 + df['GLD_ret'].iloc[i])
        
        is_jepi_avail = df.index[i] >= pd.Timestamp('2020-06-01')
        defensive_ret = df['JEPI_ret'].iloc[i] if is_jepi_avail else df['BIL_ret'].iloc[i]
        
        tactical_ret = df['QQQ_ret'].iloc[i] if pos_tactical == 1 else defensive_ret
        comp_tactical *= (1 + tactical_ret)
        
        curr_eq = comp_schd + comp_tactical + comp_kospi + comp_gold
        total_eq.append(curr_eq)

        # Finalize position for NEXT DAY based on today's close (Realistic Lag)
        sig = get_signal_at_step(i + 1 if i + 1 < len(df) else i, df, strategy_name)
        if sig != pos_tactical:
            comp_tactical *= (1 - cost) # Apply cost once when switching
            pos_tactical = sig
            trades += 1
            
    return pd.Series(total_eq, index=df.index), trades

# Data Prep
df = pd.read_csv('d:/gg/long_term_ma_data.csv', index_col=0, parse_dates=True)
df.columns = [c.replace('^', '').upper() for c in df.columns]

# Ensure SCHD (launch 2011) and others are filled properly if NaNs exist early on
for col in ['QQQ', 'JEPI', 'BIL', 'SCHD', 'KS200', 'GLD']:
    df[f'{col}_ret'] = df[col].pct_change().fillna(0)

log_ret_spy = np.log(df['SPY'] / df['SPY'].shift(1))
df['ma15'] = log_ret_spy.rolling(15).mean()
df['vol30'] = log_ret_spy.rolling(30).std()
df['mf_score'] = calculate_mf_score(df['SPY'], df['VIX'])

# Pre-calculate rolling thresholds for danger filter
ma_thresh = df['ma15'].rolling(450).apply(lambda x: np.percentile(x, 25))
vol_thresh = df['vol30'].rolling(450).apply(lambda x: np.percentile(x, 65))
df['ma15_danger'] = (df['ma15'] < ma_thresh) | (df['vol30'] > vol_thresh)

# Execute on FULL Data
results_raw = {}
metrics = {}
# OUT-OF-SAMPLE Evaluation Period
eval_start = '2011-10-01' # Starting when SCHD has enough data
eval_end = '2019-12-31'
for s in ['SMA 150', 'SMA 185', 'Original Fusion', 'Optimized Hybrid']:
    eq, t = run_golden_backtest(df, s)
    results_raw[s] = eq
    
    # Advanced Quant Analytics
    eval_eq = eq[eval_start:eval_end]
    ret = eval_eq.pct_change().dropna()
    
    cagr = (eval_eq.iloc[-1]/eval_eq.iloc[0])**(252/len(eval_eq)) - 1
    mdd = (eval_eq / eval_eq.cummax() - 1).min()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252)
    
    # Sortino: Only downside risk
    downside_ret = ret[ret < 0]
    sortino = (ret.mean() / downside_ret.std()) * np.sqrt(252) if len(downside_ret) > 0 else 0
    
    # Calmar: CAGR / ABS(MDD)
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    
    # Profit Factor: Gross Profit / Gross Loss
    total_profit = ret[ret > 0].sum()
    total_loss = abs(ret[ret < 0].sum())
    profit_factor = total_profit / total_loss if total_loss != 0 else 0
    
    metrics[s] = {
        'CAGR (%)': cagr*100,
        'MDD (%)': mdd*100,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'PF': profit_factor,
        'Trades': t
    }

# Output
final_res = pd.DataFrame(metrics).T
final_res.to_csv('d:/gg/advanced_quant_results.csv')
print("\n--- Advanced Quant Analytics Results ---")
print(final_res[['CAGR (%)', 'MDD (%)', 'Sharpe', 'Sortino', 'Calmar', 'PF']])

# Plot
plt.figure(figsize=(12,7))
for s, eq in results_raw.items():
    plt.plot(eq[eval_start:eval_end], label=s)
plt.yscale('log')
plt.legend()
plt.title('Out-of-Sample Comparison (2010-2018) - Robustness Test')
plt.savefig('d:/gg/oos_test_comparison.png')
