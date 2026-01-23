import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Data Load
df = pd.read_csv('d:/gg/swan_comparison_data_final.csv', index_col=0, parse_dates=True)
df = df.ffill().dropna()

print(f"Data Loaded: {len(df)} rows")

def run_backtest_simple(df, weights, tactical_assets, defensive_asset, signal_asset='QQQ', cost=0.002):
    ret_df = df.pct_change().fillna(0)
    sma = df[signal_asset].rolling(150).mean()
    
    pos_series = (df[signal_asset] > sma).astype(int).shift(1).fillna(1)
    
    tactical_weight = sum(weights[a] for a in tactical_assets)
    strategic_weight = 1.0 - tactical_weight
    
    strat_ret = 0
    if strategic_weight > 0:
        for asset, w in weights.items():
            if asset not in tactical_assets:
                strat_ret += (w / strategic_weight) * ret_df[asset]
                
    tac_bull_ret = 0
    for asset in tactical_assets:
        tac_bull_ret += (weights[asset] / tactical_weight) * ret_df[asset]
    tac_bear_ret = ret_df[defensive_asset]
    
    tac_ret = (pos_series * tac_bull_ret) + ((1 - pos_series) * tac_bear_ret)
    total_ret = (strategic_weight * strat_ret) + (tactical_weight * tac_ret)
    total_ret -= (pos_series.diff().abs().fillna(0) * cost)
    
    start_idx = sma.dropna().index[0]
    equity = (1 + total_ret.loc[start_idx:]).cumprod()
    return equity

# Config
# New Strategy: SPY(35), GLD(10), QQQ(35), KS200(20) | Defensive: SWAN
w_new = {'SPY': 0.35, 'GLD': 0.10, 'QQQ': 0.35, 'KS200': 0.20}
t_new = ['QQQ', 'KS200']
w_old = {'SCHD': 0.38, 'KS200': 0.19, 'GLD': 0.05, 'QQQ': 0.38}
t_old = ['QQQ']

# Sim Segment
sim_df = df['2018-01-01':].copy()

# Exec
res_swan = run_backtest_simple(sim_df, w_new, t_new, 'SWAN')
res_bil_final = run_backtest_simple(sim_df, w_old, t_old, 'BIL')

# Benchmarks
res_spy = (1 + sim_df['SPY'].pct_change().loc[res_swan.index]).fillna(0).cumprod()
res_bh = (1 + (sim_df[list(w_new.keys())].pct_change().loc[res_swan.index] * pd.Series(w_new)).sum(axis=1)).fillna(0).cumprod()

# Analytics
def get_stats(eq):
    ret = eq.pct_change().dropna()
    cagr = (eq.iloc[-1]/eq.iloc[0])**(252/len(eq)) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252)
    return {'CAGR (%)': cagr*100, 'MDD (%)': mdd*100, 'Sharpe': sharpe}

summary = pd.DataFrame({
    'Strategy (SWAN Def)': get_stats(res_swan),
    'Orig Port (BIL Def)': get_stats(res_bil_final),
    'SPY Market': get_stats(res_spy),
    'B&H (New Port)': get_stats(res_bh)
}).T

print("\n--- Final Competition Results ---")
print(summary)
summary.to_csv('d:/gg/swan_comparison_battle.csv')

# Plot
plt.figure(figsize=(12,7))
plt.plot(res_swan, label='Strategy (SWAN Def)')
plt.plot(res_bil_final, label='Original Port (BIL Def)', linestyle='--')
plt.plot(res_spy, label='SPY Market', alpha=0.5)
plt.title('SWAN vs BIL vs SPY Battle (2018-2026)')
plt.legend()
plt.savefig('d:/gg/swan_battle_final.png')
