import yfinance as yf
import pandas as pd
from datetime import datetime

assets = {
    'SPY': 'SPY',
    'QQQ': 'QQQ',
    'SCHD': 'SCHD',  # Launch 2011, will handle NaNs
    'KS200': '^KS200',
    'GLD': 'GLD',
    'VIX': '^VIX',
    'BIL': 'BIL',
    'MBB': 'MBB',   # Mortgage ETF
    'XLU': 'XLU',   # Utilities ETF
    'JEPI': 'JEPI'  # Launch 2020
}

start = "2008-01-01"
end = datetime.now().strftime("%Y-%m-%d")

print(f"Downloading data from {start} to {end}...")

all_data = pd.DataFrame()
for name, ticker in assets.items():
    print(f"Fetching {name} ({ticker})...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if not df.empty:
        # Access 'Close' column. Handle multi-index if necessary.
        if isinstance(df.columns, pd.MultiIndex):
            all_data[name] = df['Close'][ticker]
        else:
            all_data[name] = df['Close']
    else:
        print(f"Warning: {name} data is empty!")

all_data = all_data.ffill()

output_path = 'd:/gg/long_term_ma_data.csv'
all_data.to_csv(output_path)
print(f"Long-term data saved to {output_path}")
print(f"Data range: {all_data.index.min()} to {all_data.index.max()}")
