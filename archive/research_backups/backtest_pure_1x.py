import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2018-01-01"

# [기존 슈퍼 앙상블 25종에서 인버스/레버리지 필터링]
# SH: Inverse S&P 500 (제외)
# TBF: Inverse 20+yr Treasury (제외)
# 나머지 자산들은 1배수 Long 또는 Trend-following 타입임
pure_def_pool = [
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", 
    "PFIX", "DBMF", "TAIL", "IVOL", "KMLM", "CTA", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV"
]

# 전략 자산 (사용자 규칙 위반인 QLD 제외, 1배수인 VGT/QQQ 중 우세한 VGT 선택 고려)
tactical_options = ["QQQ", "VGT"]

core_assets = ["SPY", "GLD", "^KS200"]
all_tickers = list(set(pure_def_pool + tactical_options + core_assets))

print(f"Downloading data for {len(all_tickers)} pure 1x assets...")
raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, group_by='ticker')

data_dict = {}
for ticker in all_tickers:
    try:
        if ticker in raw_data.columns.get_level_values(0):
            t_data = raw_data[ticker]
            col = 'Adj Close' if 'Adj Close' in t_data.columns else 'Close'
            data_dict[ticker] = t_data[col]
    except: pass

data = pd.DataFrame(data_dict).ffill().dropna(subset=["SPY", "QQQ", "VGT", "GLD", "^KS200"])
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
def run_pure_backtest(tact_ticker):
    valid_pool = [t for t in pure_def_pool if t in data.columns]
    mom_window = 168
    mom_data = data[valid_pool].pct_change(mom_window)
    
    # 확정 비중: Tactical 45%, KS 20%, SPY 20%, GLD 15%
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
print("Simulating Pure 1x QQQ version...")
cum_qqq, cagr_qqq, mdd_qqq, sharpe_qqq = run_pure_backtest("QQQ")

print("Simulating Pure 1x VGT version...")
cum_vgt, cagr_vgt, mdd_vgt, sharpe_vgt = run_pure_backtest("VGT")

print("\n--- Pure 1x Strategy Comparison (No Inverse, No Leverage) ---")
print(f"{'Asset':<10} | {'CAGR':<10} | {'MDD':<10} | {'Sharpe':<10}")
print("-" * 50)
print(f"{'QQQ (1x)':<10} | {cagr_qqq:>10.2%} | {mdd_qqq:>10.2%} | {sharpe_qqq:>10.3f}")
print(f"{'VGT (1x)':<10} | {cagr_vgt:>10.2%} | {mdd_vgt:>10.2%} | {sharpe_vgt:>10.3f}")

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(cum_qqq, label=f'QQQ Mode (Sharpe: {sharpe_qqq:.3f})', color='gray', alpha=0.8)
plt.plot(cum_vgt, label=f'VGT Mode (Sharpe: {sharpe_vgt:.3f})', color='blue', linewidth=2)
plt.title('Pure 1x Long/Neutral Portfolio Performance (Since 2018)')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pure_1x_strategy_comparison.png')

print("\nResults saved to pure_1x_strategy_comparison.png")
