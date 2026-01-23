import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 1. 데이터 로드 및 신호 생성
start_date = "2018-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')

pure_1x_pool = [
    "BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY", 
    "PFIX", "DBMF", "TAIL", "IVOL", "KMLM", "CTA", "PDBC", "SCHP", "TLT", "IEF", "BIL", "VXV"
]
core_assets = ["QQQ", "QLD"]
all_tickers = list(set(pure_1x_pool + core_assets))

print("Fetching data for timeline visualization...")
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
current_asset = "QLD"
last_signal = "NORMAL"

for i in range(1, len(df)):
    date = df.index[i]
    sig = df['Signal'].iloc[i-1] # 전일 신호 기준 오늘 자산 결정
    
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
            'From': current_asset,
            'To': target_asset,
            'Signal': sig
        })
        current_asset = target_asset

history_df = pd.DataFrame(history)
history_df.to_csv("transaction_history.csv", index=False)

# 3. 도식화 (Timeline)
plt.figure(figsize=(16, 10))

# 배경색으로 신호 표시
for i in range(len(df)-1):
    color = 'lightgreen' if df['Signal'].iloc[i] == 'NORMAL' else 'salmon'
    plt.axvspan(df.index[i], df.index[i+1], color=color, alpha=0.2)

# 주가 차트 (QLD 로그 스케일)
plt.plot(df['QLD'] / df['QLD'].iloc[0], color='black', alpha=0.3, label='QLD (Normalized)')

# 거래 시점 표시
for idx, row in history_df.iterrows():
    marker = '^' if row['To'] == 'QLD' else 'v'
    color = 'blue' if row['To'] == 'QLD' else 'red'
    plt.scatter(row['Date'], (df['QLD'] / df['QLD'].iloc[0]).loc[row['Date']], 
                marker=marker, color=color, s=100, zorder=5)
    
    # 텍스트 라벨 (겹침 방지를 위해 고도 조절)
    y_pos = (df['QLD'] / df['QLD'].iloc[0]).loc[row['Date']]
    plt.annotate(f"{row['To']}", (row['Date'], y_pos), 
                 xytext=(0, 10 if row['To'] == 'QLD' else -20), 
                 textcoords='offset points', fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle="->", color=color))

plt.title('Portfolio Tactical Switching History (2018-2026)\nGreen: QLD Core | Red: Defensive Ensemble Mode', fontsize=15)
plt.xlabel('Date')
plt.ylabel('Asset Price (Normalized)')
plt.yscale('log')
plt.grid(True, alpha=0.2)
plt.legend(['Signal: NORMAL (QLD)', 'Signal: DANGER (Ensemble)'], loc='upper left')

plt.tight_layout()
plt.savefig('transaction_timeline.png')
print(f"\nTimeline saved to transaction_timeline.png")
print(f"Total swaps: {len(history)}")
print("\nRecent 10 Transactions:")
print(history_df.tail(10))
