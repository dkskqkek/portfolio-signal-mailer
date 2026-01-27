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
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", 
    "PFIX", "DBMF", "TAIL", "IVOL", "KMLM", "CTA", "SH", "TBF", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV"
]
unique_pool = list(set(pool_25))
assets = ["SPY", "QQQ", "GLD", "^KS200"]
tickers = list(set(assets + unique_pool))

print(f"Downloading data for {len(tickers)} assets...")
# Group by ticker to handle data more reliably
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
data = data.ffill().dropna(subset=['SPY', 'QQQ']) # Essential core assets must be present

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

# 3. 전략 실행 함수
def run_super_ensemble():
    valid_pool = [t for t in unique_pool if t in data.columns]
    strat_rets = []
    # 최적 비원 (SPY 10, KOSPI 25, GLD 20, Tactical 45)
    w_spy, w_ks, w_gld, w_tact = 0.10, 0.25, 0.20, 0.45
    mom_window = 168
    mom_data = data[valid_pool].pct_change(mom_window)
    
    for i in range(1, len(data)):
        prev_status = status.iloc[i-1]
        fixed_ret = returns["SPY"].iloc[i] * w_spy + \
                    returns["GLD"].iloc[i] * w_gld + \
                    returns["^KS200"].iloc[i] * w_ks
        
        if prev_status == "NORMAL":
            tact_ret = returns["QQQ"].iloc[i] * w_tact
        else:
            if i <= mom_window:
                actual_def = "BIL" if "BIL" in data.columns else "QQQ"
            else:
                row_mom = mom_data.iloc[i-1].dropna()
                if row_mom.empty:
                    actual_def = "BIL"
                else:
                    best_def = row_mom.idxmax()
                    actual_def = best_def if row_mom[best_def] > 0 else "BIL"
            
            if actual_def not in returns.columns:
                actual_def = "BIL" if "BIL" in returns.columns else "QQQ"
            tact_ret = returns[actual_def].iloc[i] * w_tact
            
        strat_rets.append(fixed_ret + tact_ret)
    return pd.Series(strat_rets, index=data.index[1:])

def run_spy_bh():
    return returns["SPY"].iloc[1:]

# 4. 비교 분석
print("Simulating Strategies...")
rets_super = run_super_ensemble()
rets_spy = run_spy_bh()

cum_super = (1 + rets_super).cumprod()
cum_spy = (1 + rets_spy).cumprod()

def get_metrics(rets):
    cum = (1 + rets).cumprod()
    cagr = (cum.iloc[-1] ** (252 / len(rets))) - 1
    mdd = (cum / cum.cummax() - 1).min()
    vol = rets.std() * np.sqrt(252)
    sharpe = (cagr - 0.04) / vol
    return cagr, mdd, sharpe

m_super = get_metrics(rets_super)
m_spy = get_metrics(rets_spy)

print("\n--- Strategy Comparison (2018-Present) ---")
print(f"{'Metric':<15} | {'Super Ensemble':<15} | {'SPY B&H':<15}")
print("-" * 50)
print(f"{'CAGR':<15} | {m_super[0]:>15.2%} | {m_spy[0]:>15.2%}")
print(f"{'MDD':<15} | {m_super[1]:>15.2%} | {m_spy[1]:>15.2%}")
print(f"{'Sharpe':<15} | {m_super[2]:>15.3f} | {m_spy[2]:>15.3f}")

# 5. 시각화
plt.figure(figsize=(14, 8))
plt.plot(cum_spy, label=f'SPY Buy & Hold (Sharpe: {m_spy[2]:.3f})', color='gray', linestyle='--', alpha=0.7)
plt.plot(cum_super, label=f'Super Ensemble Strategy (Sharpe: {m_super[2]:.3f})', color='blue', linewidth=2)
plt.title('Super Ensemble Strategy vs SPY Buy & Hold (Since 2018)')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('super_vs_spy_comparison.png')

# Save results
comparison_data = pd.DataFrame({
    'Super_Ensemble': cum_super,
    'SPY_BH': cum_spy
})
comparison_data.to_csv("super_vs_spy_results.csv")
print("\nFinal comparison saved to super_vs_spy_results.csv and super_vs_spy_comparison.png")
