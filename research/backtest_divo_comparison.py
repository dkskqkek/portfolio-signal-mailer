import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2018-01-01"

# 자산 리스트
core_assets = ["SCHD", "GLD"]
tactical_normal = ["QQQ", "^KS200"]
defensive_pool = ["BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY"]
alt_defensive = ["DIVO"]
signal_ticker = "QQQ"

tickers = list(set([signal_ticker] + core_assets + tactical_normal + defensive_pool + alt_defensive + ["BIL"]))

print(f"Downloading data for {len(tickers)} assets...")
raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)

if 'Adj Close' in raw_data.columns.get_level_values(0):
    data = raw_data['Adj Close']
else:
    data = raw_data['Close']

data = data.ffill().dropna()

# 데이터 부족 확인
if data.empty:
    print("Error: No data downloaded. Check tickers or date range.")
    exit(1)

print(f"Data ready. Date range: {data.index[0]} to {data.index[-1]}")

# 2. 백테스트 엔진 함수
def run_backtest(data, defensive_mode="ensemble"):
    # 가중치 설정
    # Core: SCHD(35%), GLD(10%)
    # Tactical: 55%
    
    # Signal: Dual SMA 110/250 on QQQ
    ma110 = data[signal_ticker].rolling(110).mean()
    ma250 = data[signal_ticker].rolling(250).mean()
    
    status = pd.Series(index=data.index)
    curr_status = "NORMAL"
    
    for i in range(len(data)):
        price = data[signal_ticker].iloc[i]
        m1 = ma110.iloc[i]
        m2 = ma250.iloc[i]
        
        if pd.isna(m2):
            status.iloc[i] = "NORMAL"
            continue
            
        if price > m1 and price > m2:
            curr_status = "NORMAL"
        elif price < m1 and price < m2:
            curr_status = "DANGER"
        # Else 유지 (Hysteresis)
        
        status.iloc[i] = curr_status
        
    # 일간 수익률
    returns = data.pct_change().fillna(0)
    
    # 전략 수익률 계산
    strat_returns = []
    
    for i in range(1, len(data)):
        date = data.index[i]
        prev_status = status.iloc[i-1]
        
        # Core 수익률 (45%)
        core_ret = returns["SCHD"].iloc[i] * 0.35 + returns["GLD"].iloc[i] * 0.10
        
        # Tactical 수익률 (55%)
        if prev_status == "NORMAL":
            tactical_ret = returns["QQQ"].iloc[i] * 0.35 + returns["^KS200"].iloc[i] * 0.20
        else:
            if defensive_mode == "ensemble":
                # 6개월Relative Momentum (126일)
                mom_rets = data[defensive_pool].pct_change(126).iloc[i-1]
                best_def = mom_rets.idxmax()
                # Absolute Momentum Check
                actual_def = best_def if mom_rets[best_def] > 0 else "BIL"
                tactical_ret = returns[actual_def].iloc[i] * 0.55
            else: # DIVO Only
                tactical_ret = returns["DIVO"].iloc[i] * 0.55
                
        # 거래 비용 (0.2%) - 상태 변화 시만 적용 (단순화)
        cost = 0
        if i > 1 and status.iloc[i-1] != status.iloc[i-2]:
            cost = 0.002
            
        strat_returns.append(core_ret + tactical_ret - cost)
        
    return pd.Series(strat_returns, index=data.index[1:])

# 3. 실행 및 비교
print("Running Ensemble Backtest...")
ensemble_rets = run_backtest(data, defensive_mode="ensemble")
print("Running DIVO Backtest...")
divo_rets = run_backtest(data, defensive_mode="divo")

# 벤치마크 (Buy & Hold QQQ)
qqq_rets = data[signal_ticker].pct_change().iloc[1:]

# 4. 성과 지표 계산
def get_metrics(rets):
    cum_rets = (1 + rets).cumprod()
    total_ret = cum_rets.iloc[-1] - 1
    cagr = (cum_rets.iloc[-1] ** (252 / len(rets))) - 1
    mdd = (cum_rets / cum_rets.cummax() - 1).min()
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / vol
    return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe, "Volatility": vol}

ensemble_metrics = get_metrics(ensemble_rets)
divo_metrics = get_metrics(divo_rets)
qqq_metrics = get_metrics(qqq_rets)

results = pd.DataFrame({
    "Ensemble Strategy": ensemble_metrics,
    "DIVO Strategy": divo_metrics,
    "Buy & Hold QQQ": qqq_metrics
}).T

print("\nPerformance Comparison (2018-present):")
print(results)

# 5. 차트 생성
plt.figure(figsize=(12, 7))
(1 + ensemble_rets).cumprod().plot(label=f"Ensemble (CAGR: {ensemble_metrics['CAGR']*100:.1f}%)")
(1 + divo_rets).cumprod().plot(label=f"DIVO Mode (CAGR: {divo_metrics['CAGR']*100:.1f}%)")
(1 + qqq_rets).cumprod().plot(label="QQQ Buy & Hold", alpha=0.5, linestyle="--")
plt.title("Backtest: Dynamic Ensemble vs DIVO Defense (Dual SMA 110/250)")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid(True)
plt.savefig("divo_comparison_plot.png")
results.to_csv("divo_comparison_results.csv")
print("\nResults saved to divo_comparison_results.csv and divo_comparison_plot.png")
