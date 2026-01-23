# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def optimize_dual_ma():
    # 1. 데이터 준비
    def_pool = ["BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY"]
    core_assets = ["SPY", "QQQ", "^KS200"]
    all_tickers = list(set(def_pool + core_assets + ["BIL"]))
    
    start_date = "2018-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    cost = 0.002
    
    print("Downloading data for Dual SMA Optimization...")
    raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
    data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns.get_level_values(0) else raw_data['Close']
    data = data.fillna(method='ffill')
    dr = data.pct_change().fillna(0)
    mom_returns = data[def_pool].pct_change(126) # 6개월 모멘텀
    
    # 2. 최적화 그리드 설정
    ma1_range = range(50, 201, 10)
    ma2_range = range(100, 251, 10)
    
    results = []
    
    # 3. 시뮬레이션 루프
    print(f"Starting grid search for {len(ma1_range) * len(ma2_range)} combinations...")
    
    for ma1_p in ma1_range:
        for ma2_p in ma2_range:
            if ma1_p >= ma2_p: continue
            
            sma1 = data['QQQ'].rolling(ma1_p).mean()
            sma2 = data['QQQ'].rolling(ma2_p).mean()
            
            vals = [1.0]
            curr_status = "NORMAL"
            curr_asset = None
            switches = 0
            
            for i in range(1, len(data)):
                price = data['QQQ'].iloc[i-1]
                m1 = sma1.iloc[i-1]
                m2 = sma2.iloc[i-1]
                
                if np.isnan(m2):
                    vals.append(vals[-1])
                    continue
                
                # Logic: Both above -> NORMAL, Both below -> DANGER, else stay
                if price > m1 and price > m2:
                    status = "NORMAL"
                elif price < m1 and price < m2:
                    status = "DANGER"
                else:
                    status = curr_status
                
                if status == "NORMAL":
                    target = "NORMAL"
                    r = dr['SPY'].iloc[i]*0.35 + dr['GLD'].iloc[i]*0.1 + dr['QQQ'].iloc[i]*0.35 + dr['^KS200'].iloc[i]*0.2
                else:
                    # Defensive Ensemble (Best 1 with Absolute Filter)
                    al = [t for t in def_pool if not np.isnan(mom_returns[t].iloc[i-1])]
                    b = mom_returns[al].iloc[i-1].idxmax() if al else "BIL"
                    target = b if mom_returns[b].iloc[i-1] > 0 else "BIL"
                    r = dr['SPY'].iloc[i]*0.35 + dr['GLD'].iloc[i]*0.1 + dr[target].iloc[i]*0.55
                
                c = cost if target != curr_asset else 0
                if target != curr_asset: 
                    switches += 1
                    curr_asset = target
                
                curr_status = status
                vals.append(vals[-1] * (1 + r - c))
            
            # Metrics
            s = pd.Series(vals)
            rets = s.pct_change().dropna()
            cagr = (s.iloc[-1]**(252/len(s))-1)*100
            mdd = (s/s.cummax()-1).min()*100
            shrp = (cagr - 4.0) / (rets.std() * np.sqrt(252) * 100)
            
            results.append({
                "MA1": ma1_p,
                "MA2": ma2_p,
                "CAGR (%)": cagr,
                "MDD (%)": mdd,
                "Sharpe": shrp,
                "Switches": switches
            })
            
    # 4. 결과 정리
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Sharpe", ascending=False)
    
    print("\n[Optimization Results - Top 10 by Sharpe]")
    print(df_results.head(10).to_markdown())
    
    df_results.to_csv("dual_ma_optimization_raw.csv", index=False)
    print("\n✓ Full results saved to dual_ma_optimization_raw.csv")

if __name__ == "__main__":
    optimize_dual_ma()
