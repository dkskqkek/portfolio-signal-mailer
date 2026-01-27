import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_proxy_backtest(df, defensive_asset, ma_period=150, cost=0.002):
    # Standard Portfolio Weights
    w_schd, w_tactical, w_kospi, w_gold = 0.38, 0.38, 0.19, 0.05
    comp_schd, comp_tactical, comp_kospi, comp_gold = w_schd, w_tactical, w_kospi, w_gold
    
    pos_tactical, trades, total_eq = 1, 0, []
    
    for i in range(len(df)):
        # Daily Return Update
        comp_schd *= (1 + df['SCHD_ret'].iloc[i])
        comp_kospi *= (1 + df['KS200_ret'].iloc[i])
        comp_gold *= (1 + df['GLD_ret'].iloc[i])
        
        # Defensive Switch
        defensive_ret = df[f'{defensive_asset}_ret'].iloc[i]
        tactical_ret = df['QQQ_ret'].iloc[i] if pos_tactical == 1 else defensive_ret
        comp_tactical *= (1 + tactical_ret)
        
        curr_eq = comp_schd + comp_tactical + comp_kospi + comp_gold
        total_eq.append(curr_eq)

        # Signal for Tomorrow (Lagged)
        current_idx = i
        past_px = df['QQQ'].iloc[current_idx]
        past_ma = df['QQQ'].rolling(ma_period).mean().iloc[current_idx]
        
        if not np.isnan(past_ma):
            sig = 0 if past_px < past_ma else 1
            if sig != pos_tactical:
                comp_tactical *= (1 - cost)
                pos_tactical = sig
                trades += 1
            
    return pd.Series(total_eq, index=df.index), trades

# Data Preparation
df = pd.read_csv('d:/gg/long_term_ma_data.csv', index_col=0, parse_dates=True)
# Standardize column names
df.columns = [str(c).replace('^', '').strip().upper() for c in df.columns]

# Necessary Columns
required = ['QQQ', 'SCHD', 'KS200', 'GLD', 'MBB', 'XLU', 'BIL']

# Calculate Returns (Safe dropna/fillna)
for col in required:
    df[f'{col}_ret'] = df[col].ffill().pct_change().fillna(0)

# Evaluation Window: Start from SCHD full data availability (Launched Oct 2011)
eval_start = '2011-10-21'
test_df = df[eval_start:].copy()

# Run Experiments
results = {}
metrics = []

for asset in ['MBB', 'XLU', 'BIL']:
    name = f'SMA 150 ({asset})'
    eq, t = run_proxy_backtest(test_df, asset)
    results[name] = eq
    
    # Metrics
    ret = eq.pct_change().dropna()
    cagr = (eq.iloc[-1]/eq.iloc[0])**(252/len(eq)) - 1
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0
    
    metrics.append({
        'Strategy': name,
        'CAGR (%)': cagr * 100,
        'MDD (%)': mdd * 100,
        'Sharpe': sharpe,
        'Trades': t
    })

# Add Buy & Hold QQQ for comparison
def run_bh(df):
    w_schd, w_tactical, w_kospi, w_gold = 0.38, 0.38, 0.19, 0.05
    eq = (1 + df['SCHD_ret']).cumprod() * w_schd + \
         (1 + df['QQQ_ret']).cumprod() * w_tactical + \
         (1 + df['KS200_ret']).cumprod() * w_kospi + \
         (1 + df['GLD_ret']).cumprod() * w_gold
    return eq

bh_eq = run_bh(test_df)
ret_bh = bh_eq.pct_change().dropna()
metrics.append({
    'Strategy': 'Buy & Hold (QQQ)',
    'CAGR (%)': ((bh_eq.iloc[-1]/bh_eq.iloc[0])**(252/len(bh_eq)) - 1) * 100,
    'MDD (%)': (bh_eq / bh_eq.cummax() - 1).min() * 100,
    'Sharpe': (ret_bh.mean() / ret_bh.std()) * np.sqrt(252),
    'Trades': 0
})

res_df = pd.DataFrame(metrics)
res_df.to_csv('d:/gg/long_term_proxy_results.csv', index=False)
print("--- Long-Term Proxy Backtest Results (Since 2011) ---")
print(res_df)

# Plotting
plt.figure(figsize=(12,7))
for name, eq in results.items():
    plt.plot(eq, label=name)
plt.plot(bh_eq, label='Buy & Hold', linestyle='--', alpha=0.7)
plt.yscale('log')
plt.legend()
plt.title('Long-Term Backtest: Alternative Defensive Assets (MBS vs XLU)')
plt.savefig('d:/gg/long_term_proxy_comparison.png')
print("Comparison chart saved as long_term_proxy_comparison.png")
