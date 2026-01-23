import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2018-01-01"

# [순수 1배 방어군] (No Inverse, No Leverage)
pure_1x_pool = [
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", 
    "PFIX", "DBMF", "TAIL", "IVOL", "KMLM", "CTA", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV"
]

# [2배/포트폴리오 레버리지 방어군]
leveraged_2x_pool = pure_1x_pool + ["NTSX", "UBT", "UST"]

core_assets = ["SPY", "QQQ", "QLD", "GLD", "^KS200"]
all_tickers = list(set(leveraged_2x_pool + core_assets))

print(f"Downloading data for {len(all_tickers)} assets...")
raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, group_by='ticker')

data_dict = {}
for ticker in all_tickers:
    try:
        if ticker in raw_data.columns.get_level_values(0):
            t_data = raw_data[ticker]
            col = 'Adj Close' if 'Adj Close' in t_data.columns else 'Close'
            data_dict[ticker] = t_data[col]
    except: pass

data = pd.DataFrame(data_dict).ffill().dropna(subset=["SPY", "QQQ", "QLD", "GLD", "^KS200"])
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
def run_comparison(tact_ticker, pool):
    valid_pool = [t for t in pool if t in data.columns]
    mom_window = 168
    mom_data = data[valid_pool].pct_change(mom_window)
    
    # 비중: Tactical 45%, KS 20%, SPY 20%, GLD 15%
    w_tact, w_ks, w_spy, w_gld = 0.45, 0.20, 0.20, 0.15
    
    strat_rets = []
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        
        fixed_ret = returns["SPY"].iloc[i] * w_spy + \
                    returns["GLD"].iloc[i] * w_gld + \
                    returns["^KS200"].iloc[i] * w_ks
        
        if prev_status == "NORMAL":
            tact_ret = returns[tact_ticker].iloc[i] * w_tact
        else:
            row_mom = mom_data.iloc[i-1].dropna()
            if row_mom.empty: actual_def = "BIL"
            else:
                best_def = row_mom.idxmax()
                actual_def = best_def if row_mom[best_def] > 0 else "BIL"
            tact_ret = returns[actual_def].iloc[i] * w_tact
            
        strat_rets.append(fixed_ret + tact_ret)
        
    rets = pd.Series(strat_rets, index=data.index[1:])
    cum_rets = (1 + rets).cumprod()
    cagr = (cum_rets.iloc[-1] ** (252 / len(rets))) - 1
    mdd = (cum_rets / cum_rets.cummax() - 1).min()
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / vol
    return cum_rets, cagr, mdd, sharpe

# 4. 분석 실행
print("Simulating QLD + Pure 1x Defensive Pool...")
cum_pure, cagr_pure, mdd_pure, sharpe_pure = run_comparison("QLD", pure_1x_pool)

print("Simulating QLD + 2x Leveraged Defensive Pool...")
cum_lev, cagr_lev, mdd_lev, sharpe_lev = run_comparison("QLD", leveraged_2x_pool)

print("\n--- QLD Defensive Strategy Comparison (Since 2018) ---")
print(f"{'Defensive Pool':<25} | {'CAGR':<10} | {'MDD':<10} | {'Sharpe':<10}")
print("-" * 65)
print(f"{'Pure 1x (23 assets)':<25} | {cagr_pure:>10.2%} | {mdd_pure:>10.2%} | {sharpe_pure:>10.3f}")
print(f"{'2x/Portfolio (26 assets)':<25} | {cagr_lev:>10.2%} | {mdd_lev:>10.2%} | {sharpe_lev:>10.3f}")

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(cum_pure, label=f'QLD + Pure 1x (Sharpe: {sharpe_pure:.3f})', color='blue', alpha=0.8)
plt.plot(cum_lev, label=f'QLD + 2x Capped (Sharpe: {sharpe_lev:.3f})', color='red', linewidth=2)
plt.title('QLD Strategic Core: Impact of Defensive Leverage Policy')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('qld_defensive_comparison.png')

print("\nResults saved to qld_defensive_comparison.png")
