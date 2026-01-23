# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import os

def collect_comprehensive_data(tickers=['SPY', 'QQQ', 'TLT', 'GLD', 'BIL', 'JEPI', 'SCHD', '^KS200', '^VIX'], start="2000-01-01"):
    print(f"Collecting data for {tickers} from {start}...")
    
    # Download data
    data = yf.download(tickers, start=start, progress=True, auto_adjust=True)
    
    # Extract Close prices
    if isinstance(data.columns, pd.MultiIndex):
        close_data = data['Close']
    else:
        close_data = data
        
    # Standardize column names
    close_data.columns = [c.upper() for c in close_data.columns]
    
    # Fill missing values (forward fill then drop early NaNs)
    close_data = close_data.fillna(method='ffill').dropna()
    
    # Save to CSV for optimization tasks
    output_path = 'd:/gg/comprehensive_ma_data.csv'
    close_data.to_csv(output_path, encoding='utf-8-sig')
    
    print(f"Successfully collected {len(close_data)} rows.")
    print(f"Data saved to {output_path}")
    
    return close_data

if __name__ == "__main__":
    collect_comprehensive_data()
