# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def run_8year_evolution():
    # 1. 설정
    defensive_pool = ["BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY"]
    core_assets = ["SPY", "QQQ", "^KS200"]
    all_tickers = list(set(defensive_pool + core_assets))
    
    start_date = "2018-01-01" # 8년 전 (2018-2026)
    end_date = datetime.now().strftime("%Y-%m-%d")
    cost = 0.002
    
    print(f"Downloading data for 8-year evolution (from {start_date})...")
    raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
    data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns.get_level_values(0) else raw_data['Close']
    data = data.fillna(method='ffill')
    
    # 2. 지표 계산
    sma_period = 150
    qqq_sma = data['QQQ'].rolling(window=sma_period).mean()
    is_danger = data['QQQ'] < qqq_sma
    
    momentum_period = 126
    mom_returns = data[defensive_pool].pct_change(momentum_period)
    daily_returns = data.pct_change().fillna(0)
    
    # 3. 백테스트 및 히스토리 추적
    vals = [1.0]
    curr_tactical = "INIT"
    history = []
    
    for i in range(1, len(data)):
        date = data.index[i]
        danger = is_danger.iloc[i-1]
        
        if not danger:
            target = "NORMAL (QQQ/KOSPI)"
            ret = (daily_returns['SPY'].iloc[i] * 0.35 + 
                   daily_returns['GLD'].iloc[i] * 0.10 + 
                   daily_returns['QQQ'].iloc[i] * 0.35 + 
                   daily_returns['^KS200'].iloc[i] * 0.20)
        else:
            available = [t for t in defensive_pool if not np.isnan(mom_returns[t].iloc[i-1])]
            target = mom_returns[available].iloc[i-1].idxmax() if available else "BIL"
            ret = (daily_returns['SPY'].iloc[i] * 0.35 + 
                   daily_returns['GLD'].iloc[i] * 0.10 + 
                   daily_returns[target].iloc[i] * 0.55)
            
        # 이벤트 감지 (자산 변경 시)
        if target != curr_tactical:
            history.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Event": "Switch to " + target,
                "Status": "DANGER" if danger else "NORMAL",
                "QQQ_Price": data['QQQ'].iloc[i],
                "Portfolio_Value": vals[-1]
            })
            curr_tactical = target
            c = cost
        else:
            c = 0
            
        vals.append(vals[-1] * (1 + ret - (c if i > sma_period else 0)))
        
    portfolio_cum = pd.Series(vals, index=data.index)
    spy_cum = (1 + daily_returns['SPY']).cumprod()
    
    # 4. 결과 저장
    df_history = pd.DataFrame(history)
    df_history.to_csv("portfolio_8year_timeline.csv", index=False)
    
    # 지표 산출
    def get_metrics(cum):
        rets = cum.pct_change().dropna()
        cagr = (cum.iloc[-1] ** (252 / len(cum)) - 1) * 100
        mdd = (cum / cum.cummax() - 1).min() * 100
        return cagr, mdd

    p_cagr, p_mdd = get_metrics(portfolio_cum)
    s_cagr, s_mdd = get_metrics(spy_cum)
    
    print("\n[8-Year Portfolio Evolution (2018-2026)]")
    print(f"Strategy CAGR: {p_cagr:.2f}%, MDD: {p_mdd:.2f}%")
    print(f"SPY B&H  CAGR: {s_cagr:.2f}%, MDD: {s_mdd:.2f}%")
    
    # 5. 시각화 (타임라인 포함)
    plt.figure(figsize=(14, 8))
    plt.plot(spy_cum, label="SPY (Benchmark)", color='gray', alpha=0.5)
    plt.plot(portfolio_cum, label="Dynamic Ensemble Portfolio", color='red', linewidth=2)
    
    # 주요 이벤트 마킹 (일부 발췌)
    for h in history[::max(1, len(history)//10)]: # 너무 많으면 복잡하므로 10개 정도만
        plt.annotate(h['Event'].replace("Switch to ", ""), 
                     xy=(pd.to_datetime(h['Date']), h['Portfolio_Value']),
                     xytext=(0, 20), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='black'),
                     fontsize=8, rotation=45)

    plt.yscale('log')
    plt.title(f"8-Year Portfolio Evolution Trace (2018-2026)\nCAGR: {p_cagr:.2f}% vs SPY {s_cagr:.2f}%", fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("portfolio_8year_evolution.png")
    print("\n✓ Timeline saved to portfolio_8year_timeline.csv")
    print("✓ Plot saved to portfolio_8year_evolution.png")

if __name__ == "__main__":
    run_8year_evolution()
