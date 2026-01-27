# -*- coding: utf-8 -*-
import pandas as pd
from ma_optimization_engine import backtest_ma_logic

def run_optimization():
    # Load data
    df = pd.read_csv('d:/gg/comprehensive_ma_data.csv', index_col=0, parse_dates=True)
    
    tickers = ['QQQ']
    ma_types = ['SMA', 'EMA', 'WMA'] # Focus on the requested three types
    periods = range(50, 251, 5) # Finer grain (5 day steps)
    
    all_results = []
    
    print("Starting Grid Search Optimization...")
    for ticker in tickers:
        print(f"Optimizing for {ticker}...")
        for mt in ma_types:
            for p in periods:
                res = backtest_ma_logic(df, ticker, ma_type=mt, period=p)
                res['ticker'] = ticker
                all_results.append(res)
                
    results_df = pd.DataFrame(all_results)
    
    # Find best for each ticker based on Sharpe Ratio
    best_results = []
    for ticker in tickers:
        best = results_df[results_df['ticker'] == ticker].sort_values('sharpe', ascending=False).head(5)
        best_results.append(best)
        
    final_best = pd.concat(best_results)
    print("\n--- Optimization Results (Top 5 per Ticker) ---")
    print(final_best[['ticker', 'ma_type', 'period', 'cagr', 'mdd', 'sharpe']].to_string(index=False))
    
    final_best.to_csv('d:/gg/ma_optimization_results.csv', index=False)
    print("\nBest results saved to d:/gg/ma_optimization_results.csv")

if __name__ == "__main__":
    run_optimization()
