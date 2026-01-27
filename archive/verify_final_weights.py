import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 확정 설정
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2018-01-01"

# [최종 확정 포트폴리오]
# Tactical (45%): 25-asset Super Ensemble
# KOSPI (20%): Fixed
# SPY (20%): Fixed
# GLD (15%): Fixed (User requested increment from 5%)

pool_25 = [
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", 
    "PFIX", "DBMF", "TAIL", "IVOL", "KMLM", "CTA", "SH", "TBF", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV"
]
unique_pool = list(set(pool_25))
assets = ["SPY", "QQQ", "GLD", "^KS200"]
tickers = list(set(assets + unique_pool))

print(f"Verifying final setup since {start_date}...")
raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')

data_dict = {}
for ticker in tickers:
    try:
        if ticker in raw_data.columns.get_level_values(0):
            data_dict[ticker] = raw_data[ticker]['Adj Close'] if 'Adj Close' in raw_data[ticker].columns else raw_data[ticker]['Close']
    except: pass

data = pd.DataFrame(data_dict).ffill().dropna(subset=['SPY', 'QQQ', 'GLD', '^KS200'])

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

returns = data.pct_change().fillna(0)

# 3. 확정 비중 시뮬레이션
def run_final_strategy():
    valid_pool = [t for t in unique_pool if t in data.columns]
    strat_rets = []
    # 확정 비중: Tactical 45%, KS 20%, SPY 20%, GLD 15%
    w_tact, w_ks, w_spy, w_gld = 0.45, 0.20, 0.20, 0.15
    mom_window = 168
    mom_data = data[valid_pool].pct_change(mom_window)
    
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        if pd.isna(prev_status): prev_status = "NORMAL"
        
        fixed_ret = returns["SPY"].iloc[i] * w_spy + \
                    returns["GLD"].iloc[i] * w_gld + \
                    returns["^KS200"].iloc[i] * w_ks
        
        if prev_status == "NORMAL":
            tact_ret = returns["QQQ"].iloc[i] * w_tact
        else:
            row_mom = mom_data.iloc[i-1].dropna()
            if row_mom.empty: actual_def = "BIL"
            else:
                best_def = row_mom.idxmax()
                actual_def = best_def if row_mom[best_def] > 0 else "BIL"
            tact_ret = returns[actual_def].iloc[i] * w_tact
        strat_rets.append(fixed_ret + tact_ret)
    return pd.Series(strat_rets, index=data.index[1:])

rets_final = run_final_strategy()
cum_final = (1 + rets_final).cumprod()

cagr = (cum_final.iloc[-1] ** (252 / len(rets_final))) - 1
mdd = (cum_final / cum_final.cummax() - 1).min()
sharpe = (cagr - 0.04) / (rets_final.std() * np.sqrt(252))

print(f"\n[FINAL PERFORMANCE VERIFICATION]")
print(f"Weight: Tactical 45% / KOSPI 20% / SPY 20% / GLD 15%")
print(f"CAGR: {cagr:.2%}")
print(f"MDD: {mdd:.2%}")
print(f"Sharpe: {sharpe:.3f}")

plt.figure(figsize=(12, 6))
plt.plot(cum_final, label=f'Final Strategy (Sharpe: {sharpe:.3f})', color='green', linewidth=2)
plt.title('Final Optimized Portfolio Performance (2018-Present)')
plt.ylabel('Cumulative Return')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('final_portfolio_performance.png')

print("\nFinal results saved to final_portfolio_performance.png")
