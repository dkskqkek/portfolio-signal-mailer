# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run_ensemble_backtest():
    # 1. 설정
    defensive_pool = ["BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY"]
    core_assets = ["SPY", "QQQ", "^KS200"]
    fixed_defensive = ["BIL", "SWAN"]
    
    all_tickers = list(set(defensive_pool + core_assets + fixed_defensive))
    start_date = "2019-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    cost = 0.002
    
    print(f"Downloading data for {len(all_tickers)} tickers...")
    try:
        raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
        if 'Adj Close' in raw_data.columns.get_level_values(0):
            data = raw_data['Adj Close']
        else:
            data = raw_data['Close']
    except Exception as e:
        print(f"Failed to download data: {e}")
        return
    
    data = data.fillna(method='ffill')
    
    # 3. 시그널 및 모멘텀 계산
    sma_period = 150
    qqq_sma = data['QQQ'].rolling(window=sma_period).mean()
    is_danger = data['QQQ'] < qqq_sma
    
    momentum_period = 126
    mom_returns = data[defensive_pool].pct_change(momentum_period)
    
    daily_returns = data.pct_change().fillna(0)
    
    # (0) 벤치마크 (SPY B&H)
    spy_cum = (1 + daily_returns['SPY']).cumprod()

    # (1) 동적 앙상블 전략
    def run_strategy(mode="ensemble", fixed_asset=None):
        vals = [1.0]
        curr_tactical = None
        asset_history = {}
        
        for i in range(1, len(data)):
            danger = is_danger.iloc[i-1]
            
            if not danger:
                target = "NORMAL"
                ret = (daily_returns['SPY'].iloc[i] * 0.35 + 
                       daily_returns['GLD'].iloc[i] * 0.10 + 
                       daily_returns['QQQ'].iloc[i] * 0.35 + 
                       daily_returns['^KS200'].iloc[i] * 0.20)
            else:
                if mode == "fixed":
                    target = fixed_asset
                else:
                    available = [t for t in defensive_pool if not np.isnan(mom_returns[t].iloc[i-1])]
                    target = mom_returns[available].iloc[i-1].idxmax() if available else "BIL"
                
                asset_history[target] = asset_history.get(target, 0) + 1
                ret = (daily_returns['SPY'].iloc[i] * 0.35 + 
                       daily_returns['GLD'].iloc[i] * 0.10 + 
                       daily_returns[target].iloc[i] * 0.55)
            
            c = cost if target != curr_tactical else 0
            curr_tactical = target
            vals.append(vals[-1] * (1 + ret - (c if i > sma_period else 0)))
        
        return pd.Series(vals, index=data.index), asset_history

    ensemble_cum, ensemble_history = run_strategy(mode="ensemble")
    bil_cum, _ = run_strategy(mode="fixed", fixed_asset="BIL")
    swan_cum, _ = run_strategy(mode="fixed", fixed_asset="SWAN")
    
    print("\n[Defense Asset Selection Frequency (Ensemble)]")
    for asset, count in sorted(ensemble_history.items(), key=lambda x: x[1], reverse=True):
        print(f"{asset}: {count} days")

    # 7. 지표 계산
    def get_metrics(cum_series):
        rets = cum_series.pct_change().dropna()
        cagr = (cum_series.iloc[-1] ** (252 / len(cum_series)) - 1) * 100
        mdd = (cum_series / cum_series.cummax() - 1).min() * 100
        vol = rets.std() * np.sqrt(252) * 100
        sharpe = (cagr - 4.0) / vol if vol > 0 else 0
        return cagr, mdd, sharpe

    results = [
        ["SPY (B&H)", *get_metrics(spy_cum)],
        ["Fixed Defense (BIL)", *get_metrics(bil_cum)],
        ["Fixed Defense (SWAN)", *get_metrics(swan_cum)],
        ["Dynamic Ensemble", *get_metrics(ensemble_cum)]
    ]
    
    df_res = pd.DataFrame(results, columns=["Strategy", "CAGR (%)", "MDD (%)", "Sharpe"])
    print("\n[Strategy Comparison Results]")
    print(df_res.to_markdown())
    df_res.to_csv("ensemble_strategy_results.csv", index=False)
    
    # 8. 시각화
    plt.figure(figsize=(12, 7))
    plt.plot(spy_cum, label="SPY (B&H)", alpha=0.5)
    plt.plot(bil_cum, label="Fixed Defense (BIL)", alpha=0.7)
    plt.plot(swan_cum, label="Fixed Defense (SWAN)", alpha=0.7)
    plt.plot(ensemble_cum, label="Dynamic Ensemble", linewidth=2, color='red')
    plt.yscale('log')
    plt.title("Dynamic Defensive Ensemble Comparison (Since 2019)", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig("ensemble_strategy_plot.png")
    print("\n✓ Plot saved as ensemble_strategy_plot.png")

if __name__ == "__main__":
    run_ensemble_backtest()
