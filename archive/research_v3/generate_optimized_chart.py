# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ma_optimization_engine import backtest_ma_logic, MAEngine

def generate_final_chart():
    df = pd.read_csv('d:/gg/comprehensive_ma_data.csv', index_col=0, parse_dates=True)
    
    # Best found for QQQ: SMA 150
    # Let's also compare with a faster one like HMA 100 for variety
    
    ticker = 'QQQ'
    res_sma = backtest_ma_logic(df, ticker, ma_type='SMA', period=150)
    res_hma = backtest_ma_logic(df, ticker, ma_type='HMA', period=100)
    
    # Redo backtest manually for plotting
    close = df[ticker]
    safe = df['BIL']
    
    def get_equity(ma_type, period):
        if ma_type == 'SMA': ma = close.rolling(period).mean()
        else:
             # Calculate HMA
             weights = np.arange(1, period + 1)
             wma_full = close.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
             half = int(period/2)
             weights_half = np.arange(1, half + 1)
             wma_half = close.rolling(half).apply(lambda x: np.dot(x, weights_half) / weights_half.sum(), raw=True)
             diff = 2 * wma_half - wma_full
             sqrt_len = int(np.sqrt(period))
             weights_sqrt = np.arange(1, sqrt_len + 1)
             ma = diff.rolling(sqrt_len).apply(lambda x: np.dot(x, weights_sqrt) / weights_sqrt.sum(), raw=True)
             
        sig = (close < ma).astype(int).shift(1).fillna(0)
        ret = np.where(sig == 1, safe.pct_change().fillna(0), close.pct_change().fillna(0))
        return pd.Series((1 + ret).cumprod() * 100, index=close.index).iloc[250:]

    eq_sma = get_equity('SMA', 150)
    eq_hma = get_equity('HMA', 100)
    eq_bh = (1 + close.pct_change().fillna(0)).cumprod().iloc[250:] * 100
    
    plt.figure(figsize=(15, 8))
    plt.plot(eq_bh / eq_bh.iloc[0], label='Buy & Hold', color='gray', alpha=0.5)
    plt.plot(eq_sma / eq_sma.iloc[0], label='Optimized SMA 150', color='#1f77b4', linewidth=2)
    plt.plot(eq_hma / eq_hma.iloc[0], label='Optimized HMA 100', color='#ff7f0e', linewidth=1.5)
    
    plt.title(f"Optimized MA Strategy Performance: {ticker} (250-day Burn-in)", fontsize=16)
    plt.yscale('log')
    plt.ylabel("Normalized Equity (Log Scale)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    output_path = 'd:/gg/optimized_ma_performance.png'
    plt.savefig(output_path, dpi=150)
    print(f"Final chart saved to {output_path}")

if __name__ == "__main__":
    generate_final_chart()
