import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2021-01-01" # 신규 자산(PFIX 등) 상장 시점을 고려하여 2021년부터 집중 분석

# 기존 풀 + 신규 후보군
current_pool = ["BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY"]
new_candidates = ["PFIX", "TAIL", "DBMF", "IVOL"]
all_defensive = list(set(current_pool + new_candidates))

assets = ["SPY", "QQQ", "GLD", "^KS200", "BIL"]
tickers = list(set(assets + all_defensive))

print(f"Downloading data for {len(tickers)} assets...")
raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns.get_level_values(0) else raw_data['Close']
data = data.ffill().dropna()

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

# 3. 백테스트 엔진
def run_backtest(pool):
    strat_rets = []
    # 최적 비원 (SPY 10, KOSPI 25, GLD 20, Tactical 45)
    w_spy, w_ks, w_gld, w_tact = 0.10, 0.25, 0.20, 0.45
    
    # 최근 최적화된 168일 모멘텀 윈도우 사용
    mom_window = 168
    mom_data = data[pool].pct_change(mom_window)
    
    selected_assets = []
    
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        
        fixed_ret = returns["SPY"].iloc[i] * w_spy + \
                    returns["GLD"].iloc[i] * w_gld + \
                    returns["^KS200"].iloc[i] * w_ks
        
        if prev_status == "NORMAL":
            tact_ret = returns["QQQ"].iloc[i] * w_tact
            selected_assets.append("QQQ")
        else:
            if i <= mom_window:
                actual_def = "BIL"
            else:
                row_mom = mom_data.iloc[i-1]
                # Filter out NaNs if any asset hasn't launched yet
                row_mom = row_mom.dropna()
                if row_mom.empty:
                    actual_def = "BIL"
                else:
                    best_def = row_mom.idxmax()
                    actual_def = best_def if row_mom[best_def] > 0 else "BIL"
            tact_ret = returns[actual_def].iloc[i] * w_tact
            selected_assets.append(actual_def)
            
        strat_rets.append(fixed_ret + tact_ret)
        
    rets = pd.Series(strat_rets, index=data.index[1:])
    cum_rets = (1 + rets).cumprod()
    cagr = (cum_rets.iloc[-1] ** (252 / len(rets))) - 1
    mdd = (cum_rets / cum_rets.cummax() - 1).min()
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / vol
    
    return cum_rets, cagr, mdd, sharpe, selected_assets

# 4. 비교 실행
print("Running Backtest: Current Pool...")
cum_curr, cagr_curr, mdd_curr, sharpe_curr, assets_curr = run_backtest(current_pool)

print("Running Backtest: Expanded Pool...")
cum_exp, cagr_exp, mdd_exp, sharpe_exp, assets_exp = run_backtest(all_defensive)

# 결과 요약
print("\n--- Comparative Analysis (2021-Present) ---")
print(f"{'Metric':<15} | {'Current Pool':<15} | {'Expanded Pool':<15}")
print("-" * 50)
print(f"{'CAGR':<15} | {cagr_curr:>15.2%} | {cagr_exp:>15.2%}")
print(f"{'MDD':<15} | {mdd_curr:>15.2%} | {mdd_exp:>15.2%}")
print(f"{'Sharpe':<15} | {sharpe_curr:>15.3f} | {sharpe_exp:>15.3f}")

# 자산 선택 빈도 분석 (DANGER 구간에서만)
def analyze_usage(assets_list):
    usage = pd.Series(assets_list).value_counts()
    return usage[usage.index != "QQQ"]

print("\nAsset Usage in DANGER zones (Expanded Pool):")
print(analyze_usage(assets_exp))

# 시각화
plt.figure(figsize=(12, 7))
plt.plot(cum_curr, label=f'Current Pool (Sharpe: {sharpe_curr:.3f})', color='gray', alpha=0.7)
plt.plot(cum_exp, label=f'Expanded Pool (Sharpe: {sharpe_exp:.3f})', color='blue', linewidth=2)
plt.title('Defensive Ensemble Expansion: Performance Comparison (2021-Present)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('defensive_expansion_comparison.png')

# CSV 저장
results = pd.DataFrame({
    'Current_Pool_CumRet': cum_curr,
    'Expanded_Pool_CumRet': cum_exp
})
results.to_csv("defensive_expansion_results.csv")
print("\nResults saved to defensive_expansion_results.csv and defensive_expansion_comparison.png")
