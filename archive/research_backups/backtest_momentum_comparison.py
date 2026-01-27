# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def compare_momentum_logic():
    # 1. 설정
    defensive_pool = ["BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY"]
    core_assets = ["SPY", "QQQ", "^KS200"]
    fixed_assets = ["BIL"]
    all_tickers = list(set(defensive_pool + core_assets + fixed_assets))
    
    start_date = "2018-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    cost = 0.002
    
    print(f"Downloading data for momentum comparison...")
    try:
        raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
        data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns.get_level_values(0) else raw_data['Close']
    except Exception as e:
        print(f"Download failed: {e}")
        return
    
    data = data.fillna(method='ffill')
    
    # 2. 신호 및 모멘텀 계산
    sma_period = 150
    qqq_sma = data['QQQ'].rolling(window=sma_period).mean()
    is_danger = data['QQQ'] < qqq_sma
    
    momentum_period = 126
    mom_returns = data[defensive_pool].pct_change(momentum_period)
    daily_returns = data.pct_change().fillna(0)
    
    # 3. 전략 시뮬레이션 함수
    def run_sim(mode="relative"):
        vals = [1.0]
        curr_asset = None
        for i in range(1, len(data)):
            danger = is_danger.iloc[i-1]
            if not danger:
                target = "NORMAL"
                ret = (daily_returns['SPY'].iloc[i] * 0.35 + daily_returns['GLD'].iloc[i] * 0.10 + 
                       daily_returns['QQQ'].iloc[i] * 0.35 + daily_returns['^KS200'].iloc[i] * 0.20)
            else:
                available = [t for t in defensive_pool if not np.isnan(mom_returns[t].iloc[i-1])]
                if not available:
                    target = "BIL"
                else:
                    best_asset = mom_returns[available].iloc[i-1].idxmax()
                    best_ret = mom_returns[best_asset].iloc[i-1]
                    
                    if mode == "dual":
                        # 절대 모멘텀 필터: 1위조차 마이너스면 현금(BIL)
                        target = best_asset if best_ret > 0 else "BIL"
                    else:
                        target = best_asset
                
                ret = (daily_returns['SPY'].iloc[i] * 0.35 + daily_returns['GLD'].iloc[i] * 0.10 + 
                       daily_returns[target].iloc[i] * 0.55)
            
            c = cost if target != curr_asset else 0
            curr_asset = target
            vals.append(vals[-1] * (1 + ret - (c if i > sma_period else 0)))
        return pd.Series(vals, index=data.index)

    relative_cum = run_sim(mode="relative")
    dual_cum = run_sim(mode="dual")
    spy_cum = (1 + daily_returns['SPY']).cumprod()
    
    # 4. 성과 지표 산출
    def get_metrics(series):
        rets = series.pct_change().dropna()
        cagr = (series.iloc[-1] ** (252 / len(series)) - 1) * 100
        mdd = (series / series.cummax() - 1).min() * 100
        sharpe = (cagr - 4.0) / (rets.std() * np.sqrt(252) * 100)
        return cagr, mdd, sharpe

    results = [
        ["SPY (B&H)", *get_metrics(spy_cum)],
        ["Relative Momentum", *get_metrics(relative_cum)],
        ["Dual Momentum", *get_metrics(dual_cum)]
    ]
    
    df_res = pd.DataFrame(results, columns=["Strategy", "CAGR (%)", "MDD (%)", "Sharpe"])
    print("\n[Momentum Methodology Comparison]")
    print(df_res.to_markdown())
    
    # 5. 시각화
    plt.figure(figsize=(12, 7))
    plt.plot(relative_cum, label="Relative Momentum (Top 1)", alpha=0.8)
    plt.plot(dual_cum, label="Dual Momentum (Top 1 + Filter)", color='red', linewidth=2)
    plt.plot(spy_cum, label="SPY (B&H)", color='gray', alpha=0.4)
    plt.yscale('log')
    plt.title("Relative vs Dual Momentum in Defensive Ensemble (2018-2026)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("momentum_comparison_plot.png")
    print("\n✓ Comparison results saved and plotted.")

if __name__ == "__main__":
    compare_momentum_logic()
