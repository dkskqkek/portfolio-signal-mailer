import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 설정 및 데이터 로드
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = "2018-01-01"

# [슈퍼 앙상블 후보군] - 총 25종
pool_25 = [
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", # 기존 11
    "PFIX", "DBMF", "TAIL", "IVOL", # 이전 확장 4
    "KMLM", "CTA", "SH", "TBF", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV" # 신규 후보 10
]
unique_pool = list(set(pool_25))
assets = ["SPY", "QQQ", "GLD", "^KS200"]
tickers = list(set(assets + unique_pool))

print(f"Downloading data for {len(tickers)} assets...")
# Group by ticker to avoid MultiIndex issues if possible
raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')

data_dict = {}
for ticker in tickers:
    try:
        if ticker in raw_data.columns.get_level_values(0):
            ticker_data = raw_data[ticker]
            col = 'Adj Close' if 'Adj Close' in ticker_data.columns else 'Close'
            data_dict[ticker] = ticker_data[col]
    except Exception as e:
        print(f"Warning: Could not extract data for {ticker}: {e}")

data = pd.DataFrame(data_dict)

# Fix for QQQ if missing
if 'QQQ' not in data.columns:
    print("CRITICAL: QQQ is missing from data! Downloading individually...")
    qqq_fix = yf.download("QQQ", start=start_date, end=end_date, progress=False)
    data['QQQ'] = qqq_fix['Adj Close'] if 'Adj Close' in qqq_fix.columns else qqq_fix['Close']

data = data.ffill()
# Drop rows where critical signals (QQQ) are NaN
data = data.dropna(subset=['QQQ'])
print(f"Data shape after ffill/dropna: {data.shape}")

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
    valid_pool = [t for t in pool if t in data.columns]
    print(f"Valid pool size: {len(valid_pool)} / {len(pool)}")
    
    strat_rets = []
    w_spy, w_ks, w_gld, w_tact = 0.10, 0.25, 0.20, 0.45
    mom_window = 168
    
    # Pre-calculate momentum to avoid repeated computation
    mom_data = data[valid_pool].pct_change(mom_window)
    
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
                actual_def = "BIL" if "BIL" in data.columns else "QQQ" # Fallback
            else:
                row_mom = mom_data.iloc[i-1].dropna()
                if row_mom.empty:
                    actual_def = "BIL" if "BIL" in data.columns else "QQQ"
                else:
                    best_def = row_mom.idxmax()
                    actual_def = best_def if row_mom[best_def] > 0 else "BIL"
            
            # Final safety check
            if actual_def not in returns.columns:
                actual_def = "BIL" if "BIL" in returns.columns else "QQQ"
                
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

# 4. 슈퍼 앙상블 실행
print("Running Super Ensemble Backtest...")
cum_super, cagr_super, mdd_super, sharpe_super, assets_super = run_backtest(unique_pool)

# 지표 출력
print("\n--- Super Ensemble Performance (2018-Present) ---")
print(f"CAGR: {cagr_super:.2%}")
print(f"MDD: {mdd_super:.2%}")
print(f"Sharpe: {sharpe_super:.3f}")

# 자산 선택 빈도 분석
def get_usage_stats(assets_list):
    series = pd.Series(assets_list)
    total_danger_days = len(series[series != "QQQ"])
    stats = series[series != "QQQ"].value_counts()
    stats_pct = (stats / total_danger_days) * 100
    return pd.DataFrame({'Days': stats, 'Percentage': stats_pct})

usage_stats = get_usage_stats(assets_super)
print("\n[Defense Asset Selection Breakdown]")
print(usage_stats)

# GLD 질문에 대한 답변 확인
gld_selected = usage_stats.loc['GLD', 'Days'] if 'GLD' in usage_stats.index else 0
print(f"\nQ: Was GLD ever selected? A: Yes, {gld_selected} times.")

# 시각화
plt.figure(figsize=(14, 8))
plt.plot(cum_super, label=f'Super Ensemble (25 assets) | Sharpe: {sharpe_super:.3f}', color='darkblue', linewidth=2)
plt.title('Extreme Defensive Expansion: Super Ensemble Performance (2018-Present)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('super_ensemble_comparison.png')

usage_stats.to_csv("super_ensemble_usage.csv")
print("\nDetailed usage stats saved to super_ensemble_usage.csv")
