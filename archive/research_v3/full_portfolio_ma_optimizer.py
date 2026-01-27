import pandas as pd
import numpy as np
import os

# --- Standard Portfolio Configuration ---
W_SCHD = 0.38
W_TACTICAL = 0.38
W_KOSPI = 0.19
W_GOLD = 0.05
COST = 0.002

def run_portfolio_backtest(df, ma_period):
    # Daily logic simulation
    comp_schd = 1.0 * W_SCHD
    comp_tactical = 1.0 * W_TACTICAL
    comp_kospi = 1.0 * W_KOSPI
    comp_gold = 1.0 * W_GOLD
    
    pos_tactical = 1
    trades = 0
    total_eq = []
    
    # Pre-calculate MA to avoid repeated rolling in the loop
    qqq_ma = df['QQQ'].rolling(ma_period).mean()
    
    for i in range(len(df)):
        # Daily Returns (Data already checked for NaN)
        schd_ret = df['SCHD_ret'].iloc[i]
        ks_ret = df['KS200_ret'].iloc[i]
        gold_ret = df['GLD_ret'].iloc[i]
        qqq_ret = df['QQQ_ret'].iloc[i]
        
        is_jepi_avail = df.index[i] >= pd.Timestamp('2020-06-01')
        defensive_ret = df['JEPI_ret'].iloc[i] if is_jepi_avail else df['BIL_ret'].iloc[i]
        
        # Signal Check (Look at yesterday's MA status)
        if i > 0:
            past_px = df['QQQ'].iloc[i-1]
            past_ma = qqq_ma.iloc[i-1]
            
            if not np.isnan(past_ma):
                new_pos = 0 if past_px < past_ma else 1
                if new_pos != pos_tactical:
                    comp_tactical *= (1 - COST)
                    pos_tactical = new_pos
                    trades += 1
        
        # Update Assets
        comp_schd *= (1 + schd_ret)
        comp_kospi *= (1 + ks_ret)
        comp_gold *= (1 + gold_ret)
        
        tactical_ret = qqq_ret if pos_tactical == 1 else defensive_ret
        comp_tactical *= (1 + tactical_ret)
        
        total_eq.append(comp_schd + comp_tactical + comp_kospi + comp_gold)
        
    return pd.Series(total_eq, index=df.index), trades

# Data Prep
df_raw = pd.read_csv('d:/gg/comprehensive_ma_data.csv', index_col=0, parse_dates=True)
df_raw.columns = [c.replace('^', '').upper() for c in df_raw.columns]

# Calculate Returns
for asset in ['QQQ', 'JEPI', 'BIL', 'SCHD', 'KS200', 'GLD']:
    df_raw[f'{asset}_ret'] = df_raw[asset].pct_change().fillna(0)

# Optimization Loop
periods = range(50, 251, 5)
results = []

# Test Window (Standard: Post-JEPI for highest fidelity)
test_df = df_raw['2020-06-01':].copy()

print(f"Starting Full Portfolio Grid Search (50-250)...")
for p in periods:
    eq, t = run_portfolio_backtest(test_df, p)
    
    # Metrics
    ret = eq.pct_change().dropna()
    cagr = (eq.iloc[-1]**(252/len(eq)) - 1) * 100
    mdd = (eq / eq.cummax() - 1).min() * 100
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0
    
    results.append({
        'period': p,
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'trades': t
    })

res_df = pd.DataFrame(results)
res_df.to_csv('d:/gg/full_portfolio_ma_optimization.csv', index=False)

# Analyze results
best_sharpe = res_df.loc[res_df['sharpe'].idxmax()]
best_mdd = res_df.loc[res_df['mdd'].idxmax()]

print("\n--- Optimization Complete ---")
print(f"Best Sharpe: Period {best_sharpe['period']} (Sharpe {best_sharpe['sharpe']:.2f}, CAGR {best_sharpe['cagr']:.1f}%, MDD {best_sharpe['mdd']:.1f}%)")
print(f"Best MDD: Period {best_mdd['period']} (MDD {best_mdd['mdd']:.1f}%)")

# Comparison with current SMA 150
current = res_df[res_df['period'] == 150].iloc[0]
print(f"Current SMA 150: Sharpe {current['sharpe']:.2f}, CAGR {current['cagr']:.1f}%, MDD {current['mdd']:.1f}%")
