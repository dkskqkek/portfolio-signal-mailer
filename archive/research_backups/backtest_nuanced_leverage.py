import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2018-01-01"

# [규제 준수형 방어 자산군]
# - 3배 레버리지 제외 (TMF, SQQQ 등)
# - 인버스 제외 (SH, TBF 등)
# - 2배 이하 및 포트폴리오 레버리지 허용
nuanced_def_pool = [
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", 
    "PFIX", "DBMF", "TAIL", "IVOL", "KMLM", "CTA", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV",
    "NTSX", "UBT", "UST" # 2배 이하 레버리지 및 포트폴리오 레버리지 추가
]

core_assets = ["SPY", "QQQ", "QLD", "GLD", "^KS200"]
all_tickers = list(set(nuanced_def_pool + core_assets))

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
def run_nuanced_backtest(tact_ticker, pool):
    valid_pool = [t for t in pool if t in data.columns]
    mom_window = 168
    mom_data = data[valid_pool].pct_change(mom_window)
    
    # 확정 비중: Tactical 45%, KS 20%, SPY 20%, GLD 15%
    w_tact, w_ks, w_spy, w_gld = 0.45, 0.20, 0.20, 0.15
    
    strat_rets = []
    selected_defs = []
    
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
            selected_defs.append(actual_def)
            
        strat_rets.append(fixed_ret + tact_ret)
        
    rets = pd.Series(strat_rets, index=data.index[1:])
    cum_rets = (1 + rets).cumprod()
    cagr = (cum_rets.iloc[-1] ** (252 / len(rets))) - 1
    mdd = (cum_rets / cum_rets.cummax() - 1).min()
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / vol
    return cum_rets, cagr, mdd, sharpe, selected_defs

# 4. 비교 실행
print("Simulating QQQ + Refined 2x-capped Defense...")
# 기존 25종 풀 (인버스 SH, TBF는 제거됨)
legacy_pool = [t for t in nuanced_def_pool if t not in ["NTSX", "UBT", "UST"]]
cum_qqq, cagr_qqq, mdd_qqq, sharpe_qqq, _ = run_nuanced_backtest("QQQ", legacy_pool)

print("Simulating QLD (2x) + Refined 2x-capped Defense...")
# QLD 공격 + NTSX/UBT/UST 포함 전술 방어
cum_qld, cagr_qld, mdd_qld, sharpe_qld, defs_qld = run_nuanced_backtest("QLD", nuanced_def_pool)

print("\n--- Nuanced Leverage Strategy Comparison (Since 2018) ---")
print(f"{'Strategy':<35} | {'CAGR':<10} | {'MDD':<10} | {'Sharpe':<10}")
print("-" * 75)
print(f"{'QQQ + Legacy Pure 1x':<35} | {cagr_qqq:>10.2%} | {mdd_qqq:>10.2%} | {sharpe_qqq:>10.3f}")
print(f"{'QLD + 2x-Capped Defense (NTSX+)':<35} | {cagr_qld:>10.2%} | {mdd_qld:>10.2%} | {sharpe_qld:>10.3f}")

# 레버리지 방어 자산 빈도 확인
usage = pd.Series(defs_qld)[pd.Series(defs_qld).isin(["NTSX", "UBT", "UST"])].value_counts()
print("\n[Selection Frequency of Portfolio/2x Leverage Defense]")
print(usage if not usage.empty else "No 2x leverage defensive assets selected.")

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(cum_qqq, label=f'QQQ + Pure 1x (Sharpe: {sharpe_qqq:.3f})', color='gray', alpha=0.8)
plt.plot(cum_qld, label=f'QLD + 2x-Capped (Sharpe: {sharpe_qld:.3f})', color='red', linewidth=2)
plt.title('Nuanced Leverage Strategy: QLD + 2x-capped Defensive Ensemble')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('nuanced_leverage_comparison.png')

print("\nResults saved to nuanced_leverage_comparison.png")
