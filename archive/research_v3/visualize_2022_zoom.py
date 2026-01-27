import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 1. 데이터 로드 (2022년 집중)
start_date = "2020-01-01" # 충분한 워밍업 (MA 250일 확보)
end_date = "2023-06-01"

pure_1x_pool = [
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", 
    "PFIX", "DBMF", "TAIL", "IVOL", "KMLM", "CTA", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV"
]
core_assets = ["QQQ", "QLD"]
all_tickers = list(set(pure_1x_pool + core_assets))

print("Fetching data for 2022 zoom visualization...")
raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, group_by='ticker')

data_dict = {}
for ticker in all_tickers:
    try:
        t_data = raw_data[ticker]
        col = 'Adj Close' if 'Adj Close' in t_data.columns else 'Close'
        data_dict[ticker] = t_data[col]
    except: pass

df = pd.DataFrame(data_dict).ffill().dropna(subset=["QQQ", "QLD"])

# 신호 (Dual SMA 110/250)
ma110 = df['QQQ'].rolling(110).mean()
ma250 = df['QQQ'].rolling(250).mean()
signals = []
curr_status = "NORMAL"

for i in range(len(df)):
    p = df['QQQ'].iloc[i]
    m1 = ma110.iloc[i]
    m2 = ma250.iloc[i]
    if pd.isna(m2):
        signals.append("NORMAL")
        continue
    if p > m1 and p > m2: curr_status = "NORMAL"
    elif p < m1 and p < m2: curr_status = "DANGER"
    signals.append(curr_status)

df['Signal'] = signals

# 2. 자산 교체 이력 추출
mom_window = 168
mom_data = df[pure_1x_pool].pct_change(mom_window)

history = []
current_asset = ""
active_range = df.loc["2022-01-01":"2023-04-01"]

for i in range(len(df)):
    date = df.index[i]
    if date < pd.to_datetime("2022-01-01") or date > pd.to_datetime("2023-04-01"):
        continue
        
    sig = df['Signal'].iloc[i-1]
    
    target_asset = ""
    if sig == "NORMAL":
        target_asset = "QLD"
    else:
        row_mom = mom_data.iloc[i-1].dropna()
        if row_mom.empty: target_asset = "BIL"
        else:
            best = row_mom.idxmax()
            target_asset = best if row_mom[best] > 0 else "BIL"
            
    if target_asset != current_asset:
        history.append({
            'Date': date,
            'Asset': target_asset,
            'Type': 'NORMAL' if sig == 'NORMAL' else 'DEFENSIVE'
        })
        current_asset = target_asset

history_df = pd.DataFrame(history)

# 3. 도식화 (Zoomed Timeline)
fig, ax = plt.subplots(figsize=(18, 10))

# 주가 차트 (배경)
ax2 = ax.twinx()
ax2.plot(active_range.index, active_range['QLD'], color='gray', alpha=0.2, label='QLD Price')
ax2.set_ylabel('QLD Price Index', color='gray')

# 타임라인 막대 생성
y_labels = history_df['Asset'].unique()
y_map = {asset: i for i, asset in enumerate(y_labels)}

colors = plt.cm.tab20(np.linspace(0, 1, len(y_labels)))

for i in range(len(history_df)):
    start = history_df['Date'].iloc[i]
    end = history_df['Date'].iloc[i+1] if i+1 < len(history_df) else active_range.index[-1]
    asset = history_df['Asset'].iloc[i]
    
    # 막대 그리기
    ax.barh(y_map[asset], (end - start).days, left=start, height=0.6, 
            color='lightgreen' if asset == 'QLD' else 'salmon', alpha=0.8, edgecolor='black')
    
    # 텍스트 라벨
    ax.text(start + (end - start)/2, y_map[asset], asset, 
            ha='center', va='center', fontweight='bold', fontsize=10)

ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels)
ax.set_title('2022-2023 Crisis Response: Granular Defensive Rotations\nDetailed view of how the Ensemble switched assets during the bear market', fontsize=16)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('zoom_2022_history.png')
print(f"Zoomed timeline saved to zoom_2022_history.png")
print(f"Total rotations in this period: {len(history_df)}")
