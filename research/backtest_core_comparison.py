import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2018-01-01"

# 자산 리스트
core_options = ["SCHD", "SPY"]
fixed_assets = ["GLD", "^KS200"]
tactical_asset = "QQQ"
defensive_pool = ["BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY"]

tickers = list(set(core_options + fixed_assets + [tactical_asset] + defensive_pool + ["BIL"]))

print(f"Downloading data for {len(tickers)} assets...")
raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
if 'Adj Close' in raw_data.columns.get_level_values(0):
    data = raw_data['Adj Close']
else:
    data = raw_data['Close']
data = data.ffill().dropna()

# 2. 백테스트 엔진 함수
def run_backtest(data, core_ticker="SCHD"):
    # 가중치 설정 (USER 요청 반영)
    # Core (Fixed): core_ticker(30%), GLD(5%), KS200(20%) -> Total 55%
    # Tactical (Switching): QQQ(45%)
    
    # Signal: Dual SMA 110/250 on QQQ
    ma110 = data[tactical_asset].rolling(110).mean()
    ma250 = data[tactical_asset].rolling(250).mean()
    
    status = pd.Series(index=data.index)
    curr_status = "NORMAL"
    
    for i in range(len(data)):
        price = data[tactical_asset].iloc[i]
        m1 = ma110.iloc[i]
        m2 = ma250.iloc[i]
        
        if pd.isna(m2):
            status.iloc[i] = "NORMAL"
            continue
            
        if price > m1 and price > m2:
            curr_status = "NORMAL"
        elif price < m1 and price < m2:
            curr_status = "DANGER"
        
        status.iloc[i] = curr_status
        
    returns = data.pct_change().fillna(0)
    strat_returns = []
    
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        
        # Fixed Portion (55%)
        fixed_ret = returns[core_ticker].iloc[i] * 0.30 + \
                    returns["GLD"].iloc[i] * 0.05 + \
                    returns["^KS200"].iloc[i] * 0.20
        
        # Tactical Portion (45%)
        if prev_status == "NORMAL":
            tact_ret = returns[tactical_asset].iloc[i] * 0.45
        else:
            # Dynamic Ensemble
            mom_rets = data[defensive_pool].pct_change(126).iloc[i-1]
            best_def = mom_rets.idxmax()
            actual_def = best_def if mom_rets[best_def] > 0 else "BIL"
            tact_ret = returns[actual_def].iloc[i] * 0.45
            
        # Transaction Cost (0.2%)
        cost = 0
        if i > 1 and status.iloc[i-1] != status.iloc[i-2]:
            cost = 0.002
            
        strat_returns.append(fixed_ret + tact_ret - cost)
        
    return pd.Series(strat_returns, index=data.index[1:])

# 3. 실행 및 비교
print("Running SCHD-Core Backtest...")
schd_rets = run_backtest(data, core_ticker="SCHD")
print("Running SPY-Core Backtest...")
spy_rets = run_backtest(data, core_ticker="SPY")

# 4. 성과 지표 계산
def get_metrics(rets):
    cum_rets = (1 + rets).cumprod()
    cagr = (cum_rets.iloc[-1] ** (252 / len(rets))) - 1
    mdd = (cum_rets / cum_rets.cummax() - 1).min()
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / vol
    return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Volatility": vol}

schd_metrics = get_metrics(schd_rets)
spy_metrics = get_metrics(spy_rets)

results = pd.DataFrame({
    "SCHD Core Portfolio": schd_metrics,
    "SPY Core Portfolio": spy_metrics
}).T

print("\nPerformance Comparison (2018-present):")
print(results)

# 5. 차트 생성
plt.figure(figsize=(12, 7))
(1 + schd_rets).cumprod().plot(label=f"SCHD Core (CAGR: {schd_metrics['CAGR']*100:.1f}%)")
(1 + spy_rets).cumprod().plot(label=f"SPY Core (CAGR: {spy_metrics['CAGR']*100:.1f}%)")
plt.title("Backtest Case: SCHD vs SPY as Core (30%) with 45% Tactical QQQ")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True)
plt.savefig("core_comparison_plot.png")
results.to_csv("core_comparison_results.csv")
print("\nResults saved to core_comparison_results.csv and core_comparison_plot.png")
