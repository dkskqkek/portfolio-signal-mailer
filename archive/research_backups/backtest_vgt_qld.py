import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2018-01-01"

# 슈퍼 앙상블 자산군 (최신 25종)
def_pool = [
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", 
    "PFIX", "DBMF", "TAIL", "IVOL", "KMLM", "CTA", "SH", "TBF", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV"
]
core_assets = ["SPY", "QQQ", "VGT", "QLD", "GLD", "^KS200"]
all_tickers = list(set(def_pool + core_assets))

print(f"Downloading data for {len(all_tickers)} assets...")
raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, group_by='ticker')

data_dict = {}
for ticker in all_tickers:
    try:
        if ticker in raw_data.columns.get_level_values(0):
            t_data = raw_data[ticker]
            col = 'Adj Close' if 'Adj Close' in t_data.columns else 'Close'
            data_dict[ticker] = t_data[col]
    except:
        pass

data = pd.DataFrame(data_dict).ffill().dropna(subset=["SPY", "QQQ", "VGT", "QLD", "GLD", "^KS200"])

# 2. 신호 처리 (Dual SMA 110/250)
# 신호는 여전히 QQQ 기준으로 판단 (시장 지표)
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

# 3. 백테스트 엔진
def run_backtest(tact_ticker):
    valid_pool = [t for t in def_pool if t in data.columns]
    mom_window = 168
    mom_data = data[valid_pool].pct_change(mom_window)
    
    # 비중: Tactical 45%, KS 20%, SPY 20%, GLD 15%
    w_tact, w_ks, w_spy, w_gld = 0.45, 0.20, 0.20, 0.15
    
    strat_rets = []
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        if pd.isna(prev_status): prev_status = "NORMAL"
        
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
print("Simulating QQQ version...")
cum_qqq, cagr_qqq, mdd_qqq, sharpe_qqq = run_backtest("QQQ")

print("Simulating VGT version...")
cum_vgt, cagr_vgt, mdd_vgt, sharpe_vgt = run_backtest("VGT")

print("Simulating QLD version...")
cum_qld, cagr_qld, mdd_qld, sharpe_qld = run_backtest("QLD")

# 결과 출력
print("\n--- Tactical Asset Comparison (2018-Present) ---")
print(f"{'Asset':<10} | {'CAGR':<10} | {'MDD':<10} | {'Sharpe':<10}")
print("-" * 50)
print(f"{'QQQ':<10} | {cagr_qqq:>10.2%} | {mdd_qqq:>10.2%} | {sharpe_qqq:>10.3f}")
print(f"{'VGT':<10} | {cagr_vgt:>10.2%} | {mdd_vgt:>10.2%} | {sharpe_vgt:>10.3f}")
print(f"{'QLD (2x)':<10} | {cagr_qld:>10.2%} | {mdd_qld:>10.2%} | {sharpe_qld:>10.3f}")

# 시각화
plt.figure(figsize=(14, 8))
plt.plot(cum_qqq, label=f'QQQ (Sharpe: {sharpe_qqq:.3f})', color='gray', alpha=0.7)
plt.plot(cum_vgt, label=f'VGT (Sharpe: {sharpe_vgt:.3f})', color='blue', linewidth=2)
plt.plot(cum_qld, label=f'QLD (Sharpe: {sharpe_qld:.3f})', color='red', linewidth=2)
plt.title('Tactical Asset Comparison: QQQ vs VGT vs QLD (2018-Present)')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('vgt_qld_comparison.png')

# Save results
comparison_data = pd.DataFrame({
    'QQQ': cum_qqq,
    'VGT': cum_vgt,
    'QLD': cum_qld
})
comparison_data.to_csv("vgt_qld_comparison_results.csv")
print("\nResults saved to vgt_qld_comparison_results.csv and vgt_qld_comparison.png")
