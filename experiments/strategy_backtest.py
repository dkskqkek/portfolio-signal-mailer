# -*- coding: utf-8 -*-
"""
Strategy Backtest: QQQ <-> XLP Switching
Rule:
 - Signal Source: SPY
 - Danger Condition:
    1. 20-day MA (Log Return) < 25th Percentile
    2. 20-day Volatility (Log Return) > 75th Percentile
 - Action:
    - Normal: QQQ 34%, SCHD 34%, GLD 15%, KOSPI 17% (simulated with EWY if KS200 unavailable)
    - Danger: XLP 34%, SCHD 34%, GLD 15%, KOSPI 17%
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuration
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100000

# Tickers
# EWY used as KOSPI proxy if ^KS200 fails
TICKERS = {
    'SIGNAL': 'SPY',
    'RISK_ON': 'QQQ',
    'RISK_OFF': 'XLP',
    'DIVIDEND': 'SCHD',
    'GOLD': 'GLD',
    'KOSPI': '^KS200' 
}

def fetch_data():
    print("Fetching data...")
    data = {}
    for key, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if 'Close' in df.columns:
                 # Should handle MultiIndex columns if yfinance returns them
                if isinstance(df.columns, pd.MultiIndex):
                    data[key] = df['Close'][ticker] # Extract specific ticker
                else:
                    data[key] = df['Close']
            else:
                 data[key] = df['Close'] # Fallback
            print(f"Loaded {ticker}: {len(data[key])} rows")
        except Exception as e:
            print(f"Failed to load {ticker}: {e}")
    
    # Align dates
    df_aligned = pd.DataFrame(data).dropna()
    print(f"Aligned Data: {len(df_aligned)} rows from {df_aligned.index[0].date()} to {df_aligned.index[-1].date()}")
    return df_aligned

def calculate_signals(df):
    # Calculate SPY Log Returns
    spy = df['SIGNAL']
    log_ret = np.log(spy / spy.shift(1))
    
    # Rolling Metrics
    window = 20
    roll_mean = log_ret.rolling(window=window).mean()
    roll_vol = log_ret.rolling(window=window).std()
    
    # Calculate Percentiles (expanding window to avoid lookahead bias, or fixed lookback?)
    # The requirement implied "25 percentile" of WHAT? 
    # Usually it implies the distribution of the rolling metric itself.
    # To be realistic, we should use a rolling percentile or expanding window.
    # For simplicity and speed in this check, we'll use an expanding window with a minimum period.
    
    min_periods = 252 # 1 year initialization
    
    # Expanding Percentiles
    # This is computationally expensive in pandas with standard rolling.apply
    # We will approximate by recalculating thresholds daily or using a long rolling window (e.g. 2 years)
    
    # Let's use a 1-year rolling window for "percentile context"
    context_window = 252
    
    # Vectorized approach for efficiency:
    # However, rolling_rank/percentile is slow. 
    # Let's try simple Expanding window from start (after initial warm-up)
    
    signal = pd.Series(0, index=df.index) # 0: Normal, 1: Danger
    
    # We need to iterate to simulate 'live' decision
    # But for vectorization, we can try:
    # DANGER if (Current MA < 25th percentile of Past MA) OR (Current Vol > 75th Percentile of Past Vol)
    
    # Pre-calculate expanding quantiles? No, direct support is limited.
    # Let's use a loop for the "Backtest" part to be accurate.
    
    signals = []
    
    # Pre-calculate rolling stats
    ma_series = roll_mean
    vol_series = roll_vol
    
    history_ma = []
    history_vol = []
    
    print("Calculating signals...")
    for i in range(len(df)):
        if i < min_periods:
            signals.append(0) # Default Normal during warmup
            if not np.isnan(ma_series.iloc[i]): history_ma.append(ma_series.iloc[i])
            if not np.isnan(vol_series.iloc[i]): history_vol.append(vol_series.iloc[i])
            continue
            
        current_ma = ma_series.iloc[i]
        current_vol = vol_series.iloc[i]
        
        # Stats from history
        p25_ma = np.nanpercentile(history_ma, 25)
        p75_vol = np.nanpercentile(history_vol, 75)
        
        is_danger = False
        if current_ma < p25_ma:
            is_danger = True
        elif current_vol > p75_vol:
            is_danger = True
            
        signals.append(1 if is_danger else 0)
        
        # Add to history for next day
        history_ma.append(current_ma)
        history_vol.append(current_vol)
        
    df['is_danger'] = signals
    return df

def backtest(df):
    # Weights
    # Normal: QQQ 34%, SCHD 34%, GLD 15%, KOSPI 17%
    # Danger: XLP 34%, SCHD 34%, GLD 15%, KOSPI 17%
    
    w_fixed = {'DIVIDEND': 0.34, 'KOSPI': 0.17, 'GOLD': 0.15}
    w_dynamic_target = 0.34
    
    capital = INITIAL_CAPITAL
    shares = {k: 0 for k in TICKERS.keys()}
    
    # Portfolio Value Series
    p_values = []
    
    # Buy & Hold Benchmark (Fixed Normal Portfolio)
    bnh_capital = INITIAL_CAPITAL
    bnh_shares = {k: 0 for k in TICKERS.keys()}
    bnh_values = []
    
    # Initial Allocation (Normal)
    first_prices = df.iloc[0]
    
    # Setup Strategy Portfolio
    for k, w in w_fixed.items():
        shares[k] = (capital * w) / first_prices[k]
    shares['RISK_ON'] = (capital * w_dynamic_target) / first_prices['RISK_ON']
    shares['RISK_OFF'] = 0
    
    # Setup Benchmark Portfolio (Always Risk On)
    for k, w in w_fixed.items():
        bnh_shares[k] = (bnh_capital * w) / first_prices[k]
    bnh_shares['RISK_ON'] = (bnh_capital * w_dynamic_target) / first_prices['RISK_ON']
    bnh_shares['RISK_OFF'] = 0
    
    current_mode = 0 # 0: Normal
    
    print("Running simulation...")
    
    for i in range(len(df)):
        date = df.index[i]
        prices = df.iloc[i]
        signal = df['is_danger'].iloc[i]
        
        # --- Strategy Rebalancing (Checks signal) ---
        # Logic: If signal changes, swap assets.
        # Simplification: No transaction costs, instant swap at Close.
        
        # Calculate Current Value
        curr_val =  sum(shares[k] * prices[k] for k in shares if k in prices)
        p_values.append(curr_val)
        
        if signal != current_mode:
            # Switch!
            target_on_val = 0
            target_off_val = 0
            
            # The 34% portion
            dynamic_equity = (shares['RISK_ON'] * prices['RISK_ON']) + (shares['RISK_OFF'] * prices['RISK_OFF'])
            
            if signal == 1: # Normal -> Danger
                # Sell QQQ, Buy XLP
                shares['RISK_OFF'] = dynamic_equity / prices['RISK_OFF']
                shares['RISK_ON'] = 0
                # print(f"{date.date()}: 🚨 DANGER! Switched to Defensive")
            else: # Danger -> Normal
                # Sell XLP, Buy QQQ
                shares['RISK_ON'] = dynamic_equity / prices['RISK_ON']
                shares['RISK_OFF'] = 0
                # print(f"{date.date()}: ✅ NORMAL! Switched to Aggressive")
            
            current_mode = signal
        
        # --- Benchmark ---
        bnh_val = sum(bnh_shares[k] * prices[k] for k in bnh_shares if k in prices)
        bnh_values.append(bnh_val)
        
    df['Strategy'] = p_values
    df['Benchmark'] = bnh_values
    return df

def analyze_performance(df):
    results = {}
    
    for col in ['Strategy', 'Benchmark']:
        series = df[col]
        total_ret = (series.iloc[-1] / series.iloc[0]) - 1
        
        # CAGR
        days = (series.index[-1] - series.index[0]).days
        cagr = (series.iloc[-1] / series.iloc[0]) ** (365/days) - 1
        
        # MDD
        peak = series.cummax()
        dd = (series - peak) / peak
        mdd = dd.min()
        
        # Sharpe
        daily_ret = series.pct_change()
        sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252))
        
        results[col] = {
            'Total Return': f"{total_ret*100:.2f}%",
            'CAGR': f"{cagr*100:.2f}%",
            'MDD': f"{mdd*100:.2f}%",
            'Sharpe': f"{sharpe:.2f}",
            'Final Value': f"${series.iloc[-1]:,.0f}"
        }
    
    return results

def main():
    print("=== Strategy Backtest Start ===")
    df = fetch_data()
    df = calculate_signals(df)
    df = backtest(df)
    
    # Calculate stats
    stats = analyze_performance(df)
    
    # Write report to file
    with open('strategy_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== Strategy Backtest Results ===\n")
        f.write("="*40 + "\n")
        f.write(f"{'Metric':<15} {'Strategy':<15} {'Benchmark':<15}\n")
        f.write("-" * 45 + "\n")
        for metric in ['Final Value', 'Total Return', 'CAGR', 'MDD', 'Sharpe']:
            s_val = stats['Strategy'][metric]
            b_val = stats['Benchmark'][metric]
            f.write(f"{metric:<15} {s_val:<15} {b_val:<15}\n")
    
    # Also print to console for verification
    print("\n" + "="*40)
    print("📊 Backtest Results (vs Buy & Hold)")
    print("="*40)
    
    print(f"{'Metric':<15} {'Strategy':<15} {'Benchmark':<15}")
    print("-" * 45)
    for metric in ['Final Value', 'Total Return', 'CAGR', 'MDD', 'Sharpe']:
        s_val = stats['Strategy'][metric]
        b_val = stats['Benchmark'][metric]
        print(f"{metric:<15} {s_val:<15} {b_val:<15}")
        
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Strategy'], label='Strategy (Dynamic)')
    plt.plot(df.index, df['Benchmark'], label='Buy & Hold (Static)', alpha=0.7)
    
    # Highlight Danger Zones
    danger_indices = df[df['is_danger'] == 1].index
    if len(danger_indices) > 0:
        # Approximate filling area is tricky with gaps, but simple Span works
        # Let's just create a secondary axis or fill
        pass
        
    plt.title("Portfolio Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('strategy_result.png')
    print("\nChart saved as 'strategy_result.png'")
    
if __name__ == "__main__":
    main()
