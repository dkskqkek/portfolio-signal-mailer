import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
start_date = "2018-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')

pure_1x_pool = [
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", 
    "PFIX", "DBMF", "TAIL", "IVOL", "KMLM", "CTA", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV"
]
core_assets = ["QQQ", "QLD", "SPY", "GLD", "^KS200"]
all_tickers = list(set(pure_1x_pool + core_assets))

print("Fetching data for Top-N sensitivity study...")
raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, group_by='ticker')

data_dict = {}
for ticker in all_tickers:
    try:
        t_data = raw_data[ticker]
        col = 'Adj Close' if 'Adj Close' in t_data.columns else 'Close'
        data_dict[ticker] = t_data[col]
    except: pass

data = pd.DataFrame(data_dict).ffill().dropna(subset=["QQQ", "QLD", "SPY", "GLD", "^KS200"])
returns = data.pct_change().fillna(0)

# 2. 신호 처리 (Dual SMA 110/250)
ma110 = data['QQQ'].rolling(110).mean()
ma250 = data['QQQ'].rolling(250).mean()
status = pd.Series(index=data.index)
curr_status = "NORMAL"

for i in range(len(data)):
    p = data['QQQ'].iloc[i]
    m1 = ma110.iloc[i]
    m2 = ma250.iloc[i]
    if pd.isna(m2): continue
    if p > m1 and p > m2: curr_status = "NORMAL"
    elif p < m1 and p < m2: curr_status = "DANGER"
    status.iloc[i] = curr_status

# 3. 백테스트 엔진
def run_top_n_backtest(n_top):
    mom_window = 168
    mom_data = data[pure_1x_pool].pct_change(mom_window)
    w_tact, w_ks, w_spy, w_gld = 0.45, 0.20, 0.20, 0.15
    
    strat_rets = []
    
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        fixed_ret = returns["SPY"].iloc[i] * w_spy + \
                    returns["GLD"].iloc[i] * w_gld + \
                    returns["^KS200"].iloc[i] * w_ks
        
        if prev_status == "NORMAL":
            tact_ret = returns["QLD"].iloc[i] * w_tact
        else:
            row_mom = mom_data.iloc[i-1].dropna().sort_values(ascending=False)
            if row_mom.empty:
                target_assets = ["BIL"]
            else:
                valid_top = row_mom[row_mom > 0].head(n_top)
                if valid_top.empty:
                    target_assets = ["BIL"]
                else:
                    target_assets = list(valid_top.index)
            
            tact_ret = returns[target_assets].iloc[i].mean() * w_tact
            
        strat_rets.append(fixed_ret + tact_ret)
        
    rets = pd.Series(strat_rets, index=data.index[1:])
    cum_rets = (1 + rets).cumprod()
    cagr = (cum_rets.iloc[-1] ** (252 / len(rets))) - 1
    mdd = (cum_rets / cum_rets.cummax() - 1).min()
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / vol
    return cagr, mdd, sharpe

# 4. 루프 최적화
results = []
test_range = [1, 2, 3, 5, 8, 10, 15]

for n in test_range:
    print(f"Testing Top-{n} ...")
    cagr, mdd, sharpe = run_top_n_backtest(n)
    results.append({
        'N': n,
        'CAGR': cagr,
        'MDD': mdd,
        'Sharpe': sharpe
    })

res_df = pd.DataFrame(results)
print("\n--- Defensive Assets Count Sensitivity Analysis ---")
print(res_df.set_index('N'))

# 시각화
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Number of Defensive Assets (Top-N)')
ax1.set_ylabel('CAGR / Sharpe', color=color)
ax1.plot(res_df['N'], res_df['CAGR'], marker='o', color='red', label='CAGR')
ax1.plot(res_df['N'], res_df['Sharpe'] * 0.1, marker='s', color='orange', label='Sharpe (x10 scaled)') # Scale for visibility
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('MDD', color=color)
ax2.plot(res_df['N'], res_df['MDD'], marker='x', color='blue', label='MDD')
ax2.tick_params(axis='y', labelcolor=color)
ax2.invert_yaxis() # MDD is usually negative, invert to show depth
ax2.legend(loc='upper right')

plt.title('Optimization of Defensive Asset Count (Top-N Analysis)')
plt.grid(True, alpha=0.3)
plt.savefig('top_n_optimization.png')

print("\nResults saved to top_n_optimization.png")
res_df.to_csv("top_n_optimization_results.csv", index=False)
