import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_3day_confirmation_backtest(df, cost=0.002):
    # Weights
    w_spy, w_gld = 0.35, 0.10
    w_qqq, w_vxus = 0.35, 0.20
    
    # Pre-calculate returns
    df_ret = df.pct_change().dropna()
    df = df.loc[df_ret.index]
    
    # Initialize
    capital = 1.0
    equity = [capital]
    
    # State: 1 for Safe (Bull), 0 for Danger (Bear)
    current_state = 1 
    signal_count = 0
    
    # SMA
    sma150 = df['QQQ'].rolling(150).mean()
    
    # Daily implementation
    for i in range(len(df)):
        if i == 0: continue
        
        # 1. Price Update (Based on current state and weights)
        ret_spy = df_ret['SPY'].iloc[i-1] # Note: df_ret index is 'i', which is today's return
        ret_gld = df_ret['GLD'].iloc[i-1]
        
        # Strategic 45%
        strat_part = (w_spy * ret_spy) + (w_gld * ret_gld)
        
        # Tactical 55%
        if current_state == 1:
            tactical_part = (w_qqq * df_ret['QQQ'].iloc[i-1]) + (w_vxus * df_ret['VXUS'].iloc[i-1])
        else:
            tactical_part = (w_qqq + w_vxus) * df_ret['BTAL'].iloc[i-1]
            
        daily_ret = strat_part + tactical_part
        capital *= (1 + daily_ret)
        equity.append(capital)
        
        # 2. Signal Check for state transition
        # We check today's close vs SMA to set signal for tomorrow
        price = df['QQQ'].iloc[i]
        ma = sma150.iloc[i]
        
        if pd.isna(ma): continue
        
        target_state = 1 if price > ma else 0
        
        if target_state != current_state:
            signal_count += 1
            if signal_count >= 3:
                # 3-day confirmation met -> Switch state for tomorrow
                current_state = target_state
                signal_count = 0
                capital *= (1 - cost) # Apply switch cost
        else:
            # Signal broken or matching
            signal_count = 0

    return pd.Series(equity, index=df.index)

# Load Data
df = pd.read_csv('d:/gg/btal_strategy_data.csv', index_col=0, parse_dates=True)
df = df.ffill().dropna()

# Baseline: 100% Buy & Hold of the target allocation (No switching)
def run_bh_baseline(df):
    weights = {'SPY': 0.35, 'QQQ': 0.35, 'VXUS': 0.20, 'GLD': 0.10}
    rets = df.pct_change().dropna()
    port_ret = (rets * pd.Series(weights)).sum(axis=1)
    return (1 + port_ret).cumprod()

# Sim Start: Oct 3, 2011 (BTAL launch approx)
sim_df = df['2011-10-03':].copy()

# Execution
res_3day = run_3day_confirmation_backtest(sim_df)
res_bh = run_bh_baseline(sim_df)

# Stats
def get_stats(eq):
    ret = eq.pct_change().dropna()
    cagr = (eq.iloc[-1]/eq.iloc[0])**(252/len(eq)) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252)
    down_ret = ret[ret < 0]
    sortino = (ret.mean() / down_ret.std()) * np.sqrt(252) if len(down_ret) > 0 else 0
    return {
        'CAGR (%)': cagr * 100,
        'MDD (%)': mdd * 100,
        'Sharpe': sharpe,
        'Sortino': sortino
    }

stats_3day = get_stats(res_3day)
stats_bh = get_stats(res_bh)

print("--- 3-Day Confirmed Tactical Strategy Results ---")
print(pd.DataFrame({'3rd Day Strategy': stats_3day, 'Buy & Hold': stats_bh}))

# Plot
plt.figure(figsize=(12,7))
plt.plot(res_3day, label='3-Day Confirmed + BTAL Defense', linewidth=2)
plt.plot(res_bh, label='Buy & Hold (Standard Portfolio)', linestyle='--', alpha=0.7)
plt.title('3-Day Confirmation Filter vs B&H (2011-2026)')
plt.ylabel('Normalized Equity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('d:/gg/btal_3day_comparison.png')

# Save results
comparison = pd.DataFrame({'Strategy': stats_3day, 'B&H': stats_bh}).T
comparison.to_csv('d:/gg/btal_backtest_results.csv')
