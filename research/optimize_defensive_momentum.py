import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2018-01-01"

assets = ["SPY", "QQQ", "GLD", "^KS200"]
defensive_pool = ["BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY"]
tickers = list(set(assets + defensive_pool + ["BIL"]))

print("Downloading data...")
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

# 3. 최적화 엔진: 모멘텀 윈도우별 테스트
def evaluate_momentum_window(window_days):
    strat_rets = []
    # 최적 비중 적용 (SPY 10, KOSPI 25, GLD 20, QQQ/Ens 45)
    w_spy, w_ks, w_gld, w_tact = 0.10, 0.25, 0.20, 0.45
    
    # 모멘텀 계산
    mom_data = data[defensive_pool].pct_change(window_days)
    
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        
        fixed_ret = returns["SPY"].iloc[i] * w_spy + \
                    returns["GLD"].iloc[i] * w_gld + \
                    returns["^KS200"].iloc[i] * w_ks
        
        if prev_status == "NORMAL":
            tact_ret = returns["QQQ"].iloc[i] * w_tact
        else:
            # ensemble selection based on momentum window
            if i <= window_days:
                actual_def = "BIL"
            else:
                row_mom = mom_data.iloc[i-1]
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

# 4. 루프 최적화 (1개월 ~ 12개월)
windows = [21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252]
results = []

print("Running momentum window optimization...")
for w in windows:
    c, m, s = evaluate_momentum_window(w)
    results.append({'Window': f"{w}d", 'CAGR': c, 'MDD': m, 'Sharpe': s})

df = pd.DataFrame(results)
print("\nOptimization Results:")
print(df)

best_row = df.loc[df['Sharpe'].idxmax()]
print(f"\nBest Momentum Window: {best_row['Window']}")

df.to_csv("defensive_momentum_optimization.csv")
print("\nResults saved to defensive_momentum_optimization.csv")
