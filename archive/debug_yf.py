import yfinance as yf
import pandas as pd

tickers = ["QQQ", "SPY", "GOOGL"]
print("Downloading...")
data = yf.download(tickers, period="5d", group_by="ticker", auto_adjust=False)
print("\nType:", type(data))
print("\nColumns:", data.columns)
print("\nHead:\n", data.head())

# Check extraction
try:
    sub = data["QQQ"]
    print("\nQQQ Sub columns:", sub.columns)
    print("Adj Close in sub?", "Adj Close" in sub.columns)
except Exception as e:
    print("\nError accessing QQQ:", e)
