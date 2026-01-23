import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2018-01-01"

# 슈퍼 앙상블 자산군 (최신 25종)
pool_25 = [
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", 
    "PFIX", "DBMF", "TAIL", "IVOL", "KMLM", "CTA", "SH", "TBF", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV"
]
unique_pool = list(set(pool_25))
assets = ["SPY", "QQQ", "GLD", "^KS200"]
tickers = list(set(assets + unique_pool))

print(f"Downloading data for {len(tickers)} assets...")
raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')

data_dict = {}
for ticker in tickers:
    try:
        if ticker in raw_data.columns.get_level_values(0):
            ticker_data = raw_data[ticker]
            col = 'Adj Close' if 'Adj Close' in ticker_data.columns else 'Close'
            data_dict[ticker] = ticker_data[col]
    except Exception as e:
        print(f"Warning: Could not extract data for {ticker}: {e}")

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
    if pd.isna(m2):
        status.iloc[i] = "NORMAL"
        continue
    if p > m1 and p > m2: curr_status = "NORMAL"
    elif p < m1 and p < m2: curr_status = "DANGER"
    status.iloc[i] = curr_status

returns = data.pct_change().fillna(0)

# 3. 민감도 테스트 루프
def evaluate_weights(w_spy, w_gld):
    # 고정 비중: Tactical(QQQ portion) 35%, KOSPI 20%
    w_tact = 0.35
    w_ks = 0.20
    
    mom_window = 168
    valid_pool = [t for t in unique_pool if t in data.columns]
    mom_data = data[valid_pool].pct_change(mom_window)
    
    strat_rets = []
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        fixed_ret = returns["SPY"].iloc[i] * w_spy + \
                    returns["GLD"].iloc[i] * w_gld + \
                    returns["^KS200"].iloc[i] * w_ks
        
        if prev_status == "NORMAL":
            tact_ret = returns["QQQ"].iloc[i] * w_tact
        else:
            row_mom = mom_data.iloc[i-1].dropna()
            if row_mom.empty:
                actual_def = "BIL"
            else:
                best_def = row_mom.idxmax()
                actual_def = best_def if row_mom[best_def] > 0 else "BIL"
            tact_ret = returns[actual_def].iloc[i] * w_tact
            
        strat_rets.append(fixed_ret + tact_ret)
        
    rets = pd.Series(strat_rets)
    cagr = ( (1 + rets).prod() ** (252 / len(rets)) ) - 1
    mdd = ( (1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1 ).min()
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / vol
    return cagr, mdd, sharpe

results = []
# 전체 합 100% 중 고정부 55% (Tact 35 + KS 20) 제외한 45%를 SPY와 GLD에 배분
for gld_pct in range(0, 50, 5):
    spy_pct = 45 - gld_pct
    c, m, s = evaluate_weights(spy_pct/100, gld_pct/100)
    results.append({
        'GLD_Weight': gld_pct,
        'SPY_Weight': spy_pct,
        'CAGR': c,
        'MDD': m,
        'Sharpe': s
    })

df = pd.DataFrame(results)
print(df)

# 시각화
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('GLD Weight (%) (Remaining is SPY)')
ax1.set_ylabel('CAGR (%)', color='tab:blue')
ax1.plot(df['GLD_Weight'], df['CAGR'] * 100, marker='o', color='tab:blue', label='CAGR')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('MDD (%)', color='tab:red')
ax2.plot(df['GLD_Weight'], df['MDD'] * 100, marker='s', color='tab:red', label='MDD')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('CAGR/MDD Sensitivity: GLD vs SPY (Fixed: Tactical 35%, KOSPI 20%)')
fig.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig('gld_spy_sensitivity.png')

df.to_csv("gld_spy_sensitivity_results.csv")
print("\nSensitivity results saved to gld_spy_sensitivity_results.csv")
