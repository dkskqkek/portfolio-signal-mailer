import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_tactical_optimization(df, tactical_ratio, signal_asset='QQQ', cost=0.002):
    # Base Assets: SPY, GLD, QQQ, KS200, SWAN
    ret_df = df.pct_change().fillna(0)
    sma = df[signal_asset].rolling(150).mean()
    pos = (df[signal_asset] > sma).astype(int).shift(1).fillna(1)
    
    # Weights internal ratio (35:10:35:20)
    # W_total = W_core (SPY, GLD) + W_tactical (QQQ, KS200)
    # Internal ratio of tactical: QQQ 35/55, KS200 20/55
    # Internal ratio of core: SPY 35/45, GLD 10/45
    
    w_t_total = tactical_ratio
    w_c_total = 1.0 - w_t_total
    
    # 1. Strategic Core (Always On)
    # Using internal 35:10 ratio within the core portion
    strat_ret = 0
    if w_c_total > 0:
        w_spy = (35/45) * w_c_total
        w_gld = (10/45) * w_c_total
        strat_ret = (w_spy * ret_df['SPY']) + (w_gld * ret_df['GLD'])
        
    # 2. Tactical Sleeve (Switching)
    # Bull mode: QQQ (35/55), KS200 (20/55)
    # Bear mode: SWAN (100% of sleeve)
    tac_bull_ret = (35/55 * ret_df['QQQ']) + (20/55 * ret_df['KS200'])
    tac_bear_ret = ret_df['SWAN']
    
    tac_ret = w_t_total * ((pos * tac_bull_ret) + ((1 - pos) * tac_bear_ret))
    
    # Total
    total_ret = strat_ret + tac_ret
    total_ret -= (pos.diff().abs().fillna(0) * cost * w_t_total)
    
    start_idx = sma.dropna().index[0]
    equity = (1 + total_ret.loc[start_idx:]).cumprod()
    return equity

# Load Data
df = pd.read_csv('d:/gg/swan_comparison_data_final.csv', index_col=0, parse_dates=True)
df = df.ffill().dropna()

# Optimization Loop
results = []
ratios = np.arange(0, 1.05, 0.05) # 0% to 100%

for r in ratios:
    eq = run_tactical_optimization(df, r)
    ret = eq.pct_change().dropna()
    cagr = (eq.iloc[-1]/eq.iloc[0])**(252/len(eq)) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0
    results.append({
        'Tactical Ratio': r * 100,
        'CAGR (%)': cagr * 100,
        'MDD (%)': mdd * 100,
        'Sharpe': sharpe
    })

res_df = pd.DataFrame(results)
print(res_df)
res_df.to_csv('d:/gg/tactical_optimization_results.csv')

# Visual Analysis
fig, ax1 = plt.subplots(figsize=(12,7))
ax1.set_xlabel('Tactical Switching Ratio (%)')
ax1.set_ylabel('Sharpe Ratio', color='tab:blue')
ax1.plot(res_df['Tactical Ratio'], res_df['Sharpe'], color='tab:blue', marker='o', label='Sharpe Ratio')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('MDD (%)', color='tab:red')
ax2.plot(res_df['Tactical Ratio'], res_df['MDD (%)'], color='tab:red', linestyle='--', marker='x', label='MDD (%)')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Optimal Switching Ratio Search (Sharpe vs MDD)')
plt.grid(True, alpha=0.3)
plt.savefig('d:/gg/tactical_optimization_plot.png')
