import pandas as pd
import yfinance as yf

# 1. Download US Data
tickers = ['SPY', 'QQQ', 'VXUS', 'GLD', 'SWAN', 'SCHD', 'BIL']
df_us = yf.download(tickers, start='2018-01-01', progress=False)['Close']

# 2. Load Local KOSPI Data
df_kr = pd.read_csv('d:/gg/long_term_ma_data.csv', index_col=0, parse_dates=True)

# 3. Merge
# Align indices
df_us.index = pd.to_datetime(df_us.index)
df_kr.index = pd.to_datetime(df_kr.index)

# Join and ffill
result = df_us.join(df_kr[['KS200']], how='outer').sort_index()
result = result.ffill().dropna()

# 4. Save
result.to_csv('d:/gg/swan_comparison_data_final.csv')
print(f"Final Merged Rows: {len(result)}")
print(f"Columns: {result.columns.tolist()}")
