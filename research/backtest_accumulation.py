import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pyxirr import xirr
from datetime import date

def run_precision_accumulation(df, initial_cap=10000, monthly_add=500, cost=0.002, proxy='JEPI'):
    # Weights
    w_schd, w_tactical, w_kospi, w_gold = 0.38, 0.38, 0.19, 0.05
    
    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)
    
    cash_flows = []
    # Initial Investment
    cash_flows.append((df.index[0].date(), -float(initial_cap)))
    
    # Initialize components
    comp_schd = float(initial_cap) * w_schd
    comp_tactical = float(initial_cap) * w_tactical
    comp_kospi = float(initial_cap) * w_kospi
    comp_gold = float(initial_cap) * w_gold
    
    pos_tactical = 1 # 1: QQQ, 0: Proxy
    total_wealth = []
    
    last_month = -1
    
    for i in range(len(df)):
        # 1. Monthly Contribution
        current_dt = df.index[i].date()
        if current_dt.month != last_month:
            comp_schd += float(monthly_add) * w_schd
            comp_tactical += float(monthly_add) * w_tactical
            comp_kospi += float(monthly_add) * w_kospi
            comp_gold += float(monthly_add) * w_gold
            cash_flows.append((current_dt, -float(monthly_add)))
            last_month = current_dt.month
            
        # 2. Daily Price Update
        comp_schd *= (1.0 + float(df['SCHD_ret'].iloc[i]))
        comp_kospi *= (1.0 + float(df['KS200_ret'].iloc[i]))
        comp_gold *= (1.0 + float(df['GLD_ret'].iloc[i]))
        
        tactical_ret = float(df['QQQ_ret'].iloc[i]) if pos_tactical == 1 else float(df[f'{proxy}_ret'].iloc[i])
        comp_tactical *= (1.0 + tactical_ret)
        
        curr_val = comp_schd + comp_tactical + comp_kospi + comp_gold
        total_wealth.append(curr_val)
        
        # 3. SMA 150 Signal (Always relative to QQQ for consistency)
        past_ma = df['QQQ'].rolling(150).mean().iloc[i]
        if not np.isnan(past_ma):
            sig = 0 if float(df['QQQ'].iloc[i]) < float(past_ma) else 1
            if sig != pos_tactical:
                comp_tactical *= (1.0 - cost)
                pos_tactical = sig

    # Add Final Valuation
    cash_flows.append((df.index[-1].date(), float(total_wealth[-1])))
    
    # XIRR
    xirr_val = xirr(cash_flows)
    
    return pd.Series(total_wealth, index=df.index), xirr_val

def get_metrics(wealth, xirr_val):
    ret = wealth.pct_change().dropna()
    mdd = (wealth / wealth.cummax() - 1).min()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0
    down_ret = ret[ret < 0]
    sortino = (ret.mean() / down_ret.std()) * np.sqrt(252) if len(down_ret) > 0 and down_ret.std() != 0 else 0
    return {
        'Final Wealth': wealth.iloc[-1],
        'XIRR (%)': xirr_val * 100,
        'MDD (%)': mdd * 100,
        'Sharpe': sharpe,
        'Sortino': sortino
    }

# Load Data
df = pd.read_csv('d:/gg/long_term_ma_data.csv', index_col=0, parse_dates=True)
df.columns = [str(c).replace('^', '').strip().upper() for c in df.columns]

# Ensure required columns
required_assets = ['QQQ', 'SCHD', 'KS200', 'GLD', 'JEPI', 'BIL', 'XLU']
for col in required_assets:
    df[f'{col}_ret'] = df[col].ffill().pct_change().fillna(0)

# Start Date: When JEPI data becomes available
start_date = df['JEPI'].dropna().index.min()
sim_df = df[start_date:].copy()

# Run Simulations
wealth_jepi, xirr_jepi = run_precision_accumulation(sim_df, proxy='JEPI')
wealth_bil, xirr_bil = run_precision_accumulation(sim_df, proxy='BIL')
wealth_xlu, xirr_xlu = run_precision_accumulation(sim_df, proxy='XLU')

# Collect Metrics
results = {
    'Defensive: JEPI': get_metrics(wealth_jepi, xirr_jepi),
    'Defensive: BIL': get_metrics(wealth_bil, xirr_bil),
    'Defensive: XLU': get_metrics(wealth_xlu, xirr_xlu)
}

final_df = pd.DataFrame(results).T
print("\n--- 3-Way Accumulation Comparison (JEPI Period) ---")
print(f"Period: {start_date.date()} to {sim_df.index[-1].date()}")
print(final_df)

final_df.to_csv('d:/gg/accumulation_3way_results.csv')

# Plot
plt.figure(figsize=(12,7))
plt.plot(wealth_jepi, label=f'JEPI Version | XIRR: {xirr_jepi*100:.1f}%', linewidth=2)
plt.plot(wealth_bil, label=f'BIL Version | XIRR: {xirr_bil*100:.1f}%', linestyle='--')
plt.plot(wealth_xlu, label=f'XLU Version | XIRR: {xirr_xlu*100:.1f}%', linestyle=':')
plt.title(f'Accumulation Growth Comparison (Since JEPI Launch: {start_date.date()})')
plt.ylabel('Asset Value ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('d:/gg/accumulation_3way_plot.png')
