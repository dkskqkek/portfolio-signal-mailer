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

print("Fetching data for Top-1 vs Top-3 comparison...")
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

# 3. 백테스트 엔진 (Turnover 측정 포함)
def run_top_n_backtest(n_top):
    mom_window = 168
    mom_data = data[pure_1x_pool].pct_change(mom_window)
    
    w_tact, w_ks, w_spy, w_gld = 0.45, 0.20, 0.20, 0.15
    
    strat_rets = []
    asset_history = []
    trade_count = 0
    current_assets = set()
    
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        
        # 고정 자산 수익률
        fixed_ret = returns["SPY"].iloc[i] * w_spy + \
                    returns["GLD"].iloc[i] * w_gld + \
                    returns["^KS200"].iloc[i] * w_ks
        
        if prev_status == "NORMAL":
            target_assets = {"QLD"}
            tact_ret = returns["QLD"].iloc[i] * w_tact
        else:
            row_mom = mom_data.iloc[i-1].dropna().sort_values(ascending=False)
            if row_mom.empty:
                target_assets = {"BIL"}
            else:
                # Top N 중 절대 모멘텀 > 0 인 것만 추출, 없으면 BIL
                valid_top = row_mom[row_mom > 0].head(n_top)
                if valid_top.empty:
                    target_assets = {"BIL"}
                else:
                    target_assets = set(valid_top.index)
            
            # Top N 균등 배분 수익률
            tact_ret = returns[list(target_assets)].iloc[i].mean() * w_tact
            
        # 교체 매매 체크 (자산 구성이 달라지면 카운트)
        if target_assets != current_assets:
            trade_count += 1
            current_assets = target_assets
            
        strat_rets.append(fixed_ret + tact_ret)
        
    rets = pd.Series(strat_rets, index=data.index[1:])
    cum_rets = (1 + rets).cumprod()
    cagr = (cum_rets.iloc[-1] ** (252 / len(rets))) - 1
    mdd = (cum_rets / cum_rets.cummax() - 1).min()
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / vol
    
    return cum_rets, cagr, mdd, sharpe, trade_count

# 4. 분석 실행
print("Simulating Top-1 (Concentrated) ...")
cum1, cagr1, mdd1, sharpe1, trades1 = run_top_n_backtest(1)

print("Simulating Top-3 (Diversified) ...")
cum3, cagr3, mdd3, sharpe3, trades3 = run_top_n_backtest(3)

# 결과 출력
print("\n--- Concentration vs Diversification Comparison ---")
print(f"{'Strategy':<15} | {'CAGR':<10} | {'MDD':<10} | {'Sharpe':<10} | {'Switches':<10}")
print("-" * 70)
print(f"{'Top-1 Core':<15} | {cagr1:>10.2%} | {mdd1:>10.2%} | {sharpe1:>10.3f} | {trades1:>10}")
print(f"{'Top-3 Ensemble':<15} | {cagr3:>10.2%} | {mdd3:>10.2%} | {sharpe3:>10.3f} | {trades3:>10}")

reduction = (trades1 - trades3) / trades1
print(f"\nTurnover Reduction: {reduction:.2%}")

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(cum1, label=f'Top-1 Concentrated (Trades: {trades1})', color='blue', alpha=0.6)
plt.plot(cum3, label=f'Top-3 Diversified (Trades: {trades3})', color='red', linewidth=2)
plt.title('Impact of Defensive Diversification: Top-1 vs Top-3 assets')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('turnover_comparison.png')

print("\nResults saved to turnover_comparison.png")
