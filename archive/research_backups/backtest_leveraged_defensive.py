import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2018-01-01"

# [기존 슈퍼 앙상블 25종]
base_def_pool = [
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", 
    "PFIX", "DBMF", "TAIL", "IVOL", "KMLM", "CTA", "SH", "TBF", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV"
]

# [레버리지 방어 후보군 추가]
# TMF: 20년+ 국채 3배
# SQQQ: 나스닥 100 인버스 3배
# SPXU: S&P 500 인버스 3배
# TYD: 7-10년 국채 3배
# UDN: 달러 약세 2배 (UUP 인버스 대용)
leveraged_def_pool = base_def_pool + ["TMF", "SQQQ", "SPXU", "TYD", "UDN"]

core_assets = ["SPY", "QQQ", "GLD", "^KS200"]
all_tickers = list(set(leveraged_def_pool + core_assets))

print(f"Downloading data for {len(all_tickers)} assets...")
# Multi-index download
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

data = pd.DataFrame(data_dict).ffill().dropna(subset=["SPY", "QQQ", "GLD", "^KS200"])

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

returns = data.pct_change().fillna(0)

# 3. 백테스트 엔진
def run_backtest(pool, name):
    valid_pool = [t for t in pool if t in data.columns]
    mom_window = 168 # 8개월 모멘텀
    mom_data = data[valid_pool].pct_change(mom_window)
    
    # 확정 비중: Tactical 45%, KS 20%, SPY 20%, GLD 15%
    w_tact, w_ks, w_spy, w_gld = 0.45, 0.20, 0.20, 0.15
    
    strat_rets = []
    selected_defs = []
    
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        
        fixed_ret = returns["SPY"].iloc[i] * w_spy + \
                    returns["GLD"].iloc[i] * w_gld + \
                    returns["^KS200"].iloc[i] * w_ks
        
        if prev_status == "NORMAL":
            tact_ret = returns["QQQ"].iloc[i] * w_tact
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
print("Simulating Base Super Ensemble...")
cum_base, cagr_base, mdd_base, sharpe_base, defs_base = run_backtest(base_def_pool, "Base")

print("Simulating Leveraged Super Ensemble...")
cum_lev, cagr_lev, mdd_lev, sharpe_lev, defs_lev = run_backtest(leveraged_def_pool, "Leveraged")

# 결과 출력
print("\n--- Defensive Asset Evolution Comparison ---")
print(f"{'Strategy':<20} | {'CAGR':<10} | {'MDD':<10} | {'Sharpe':<10}")
print("-" * 60)
print(f"{'Base (25 assets)':<20} | {cagr_base:>10.2%} | {mdd_base:>10.2%} | {sharpe_base:>10.3f}")
print(f"{'Lev (30 assets)':<20} | {cagr_lev:>10.2%} | {mdd_lev:>10.2%} | {sharpe_lev:>10.3f}")

# 레버리지 자산 선택 빈도 확인
lev_assets = ["TMF", "SQQQ", "SPXU", "TYD", "UDN"]
selected_series = pd.Series(defs_lev)
usage = selected_series[selected_series.isin(lev_assets)].value_counts()
print("\n[Leveraged Defensive Asset Selection frequency]")
print(usage if not usage.empty else "No leveraged assets selected during danger periods.")

# 시각화
plt.figure(figsize=(14, 8))
plt.plot(cum_base, label=f'Base Ensemble (Sharpe: {sharpe_base:.3f})', color='blue', alpha=0.6)
plt.plot(cum_lev, label=f'Leveraged Ensemble (Sharpe: {sharpe_lev:.3f})', color='red', linewidth=2)
plt.title('Impact of Leveraged Assets in Defensive Ensemble (2018-Present)')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('leveraged_defensive_comparison.png')

print("\nResults saved to leveraged_defensive_comparison.png")
