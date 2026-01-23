import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# 1. 데이터 수집 (코로나 이전: 2020-01-31까지)
# 최대한 과거로 가기 위해 USFR(2014-02) 런칭 이후인 2014년부터 시작
end_date = "2020-01-31"
start_date = "2014-03-01"

assets = ["SPY", "QQQ", "GLD", "^KS200"]
defensive_pool = ["BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY"]
tickers = list(set(assets + defensive_pool + ["BIL"]))

print(f"Downloading pre-COVID data from {start_date} to {end_date}...")
raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns.get_level_values(0) else raw_data['Close']
data = data.ffill().dropna()

# 2. 신호 처리 (Dual SMA 110/250 on QQQ)
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

# 3. 최적화 엔진
def evaluate_portfolio(w_spy, w_qqq, w_gld, w_ks):
    strat_rets = []
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        
        fixed_ret = returns["SPY"].iloc[i] * w_spy + \
                    returns["GLD"].iloc[i] * w_gld + \
                    returns["^KS200"].iloc[i] * w_ks
        
        if prev_status == "NORMAL":
            tact_ret = returns["QQQ"].iloc[i] * w_qqq
        else:
            mom_rets = data[defensive_pool].pct_change(126).iloc[i-1]
            best_def = mom_rets.idxmax()
            actual_def = best_def if mom_rets[best_def] > 0 else "BIL"
            tact_ret = returns[actual_def].iloc[i] * w_qqq
            
        strat_rets.append(fixed_ret + tact_ret)
        
    rets = pd.Series(strat_rets)
    cagr = ( (1 + rets).prod() ** (252 / len(rets)) ) - 1
    mdd = ( (1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1 ).min()
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / vol if vol > 0 else 0
    return cagr, mdd, sharpe

# 4. 그리드 서치 실행
results = []
# QQQ: 10~80%, SPY: 0~80%, GLD: 5~20%, KOSPI: 5~30%
for w_qqq in range(10, 85, 5):
    for w_spy in range(0, 85, 5):
        for w_gld in range(5, 25, 5):
            for w_ks in range(5, 35, 5):
                if w_spy + w_qqq + w_gld + w_ks == 100:
                    c, m, s = evaluate_portfolio(w_spy/100, w_qqq/100, w_gld/100, w_ks/100)
                    results.append({
                        'SPY': w_spy, 'QQQ': w_qqq, 'GLD': w_gld, 'KOSPI': w_ks,
                        'CAGR': c, 'MDD': m, 'Sharpe': s
                    })

df = pd.DataFrame(results)
if df.empty:
    print("Error: No valid combinations found.")
    exit(1)
    
best_sharpe = df.loc[df['Sharpe'].idxmax()]

print("\nBest Sharpe Portfolio (Pre-COVID, Long-term):")
print(best_sharpe)

df.to_csv("pre_covid_optimization_results.csv")
print("\nResults saved to pre_covid_optimization_results.csv")
