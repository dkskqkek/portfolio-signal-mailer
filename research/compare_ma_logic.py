# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def compare_ma_logic():
    # 1. 설정
    defensive_pool = ["BTAL", "XLP", "XLU", "GLD", "FXY", "UUP", "MNA", "QAI", "DBC", "USFR", "GSY"]
    core_assets = ["SPY", "QQQ", "^KS200"]
    fixed_assets = ["BIL"]
    all_tickers = list(set(defensive_pool + core_assets + fixed_assets))
    
    start_date = "2018-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    cost = 0.002
    momentum_period = 126
    
    print(f"Downloading data for MA logic comparison...")
    try:
        raw_data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
        data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns.get_level_values(0) else raw_data['Close']
    except Exception as e:
        print(f"Download failed: {e}")
        return
    
    data = data.fillna(method='ffill')
    
    # 2. 이평선 계산
    sma150 = data['QQQ'].rolling(150).mean()
    sma200 = data['QQQ'].rolling(200).mean()
    
    mom_returns = data[defensive_pool].pct_change(momentum_period)
    daily_returns = data.pct_change().fillna(0)
    
    # 3. 전략 시뮬레이션 함수
    def run_sim(mode="single"):
        vals = [1.0]
        curr_status = "NORMAL" # 초기값
        curr_tactical_asset = None
        switch_count = 0
        
        for i in range(1, len(data)):
            # 신호 판단 (전일 종가 기준)
            price = data['QQQ'].iloc[i-1]
            ma150 = sma150.iloc[i-1]
            ma200 = sma200.iloc[i-1]
            
            if np.isnan(ma200): # 워밍업 기간
                vals.append(vals[-1])
                continue
                
            # 모드 결정
            if mode == "single":
                new_status = "NORMAL" if price > ma150 else "DANGER"
            else: # dual sma 150/200 logic
                if price > ma150 and price > ma200:
                    new_status = "NORMAL"
                elif price < ma150 and price < ma200:
                    new_status = "DANGER"
                else:
                    new_status = curr_status # 상태 유지 (Hysteresis)
            
            # 자산 선택
            if new_status == "NORMAL":
                target = "NORMAL"
                ret = (daily_returns['SPY'].iloc[i] * 0.35 + daily_returns['GLD'].iloc[i] * 0.10 + 
                       daily_returns['QQQ'].iloc[i] * 0.35 + daily_returns['^KS200'].iloc[i] * 0.20)
            else:
                # 듀얼 모멘텀 앙상블 로직 적용 (Best 1 if ret > 0 else BIL)
                available = [t for t in defensive_pool if not np.isnan(mom_returns[t].iloc[i-1])]
                best_t = mom_returns[available].iloc[i-1].idxmax() if available else "BIL"
                target = best_t if mom_returns[best_t].iloc[i-1] > 0 else "BIL"
                ret = (daily_returns['SPY'].iloc[i] * 0.35 + daily_returns['GLD'].iloc[i] * 0.10 + 
                       daily_returns[target].iloc[i] * 0.55)
            
            # 비용 및 카운트 (상태가 변하거나 방어 자산 내에서 변할 때)
            c = 0
            if target != curr_tactical_asset:
                c = cost
                switch_count += 1
                curr_tactical_asset = target
            
            curr_status = new_status
            vals.append(vals[-1] * (1 + ret - c))
            
        return pd.Series(vals, index=data.index), switch_count

    single_cum, single_sw = run_sim("single")
    dual_cum, dual_sw = run_sim("dual")
    
    # 4. 결과 분석
    def get_metrics(s):
        rets = s.pct_change().dropna()
        cagr = (s.iloc[-1] ** (252 / len(s)) - 1) * 100
        mdd = (s / s.cummax() - 1).min() * 100
        vol = rets.std() * np.sqrt(252) * 100
        shrp = (cagr - 4.0) / vol if vol > 0 else 0
        return cagr, mdd, shrp

    res_single = get_metrics(single_cum)
    res_dual = get_metrics(dual_cum)
    
    print("\n[MA Logic Comparison: SMA 150 vs Dual SMA 150/200]")
    print(f"SMA 150 Only: CAGR {res_single[0]:.2f}%, MDD {res_single[1]:.2f}%, Switches {single_sw}")
    print(f"Dual SMA 150/200: CAGR {res_dual[0]:.2f}%, MDD {res_dual[1]:.2f}%, Switches {dual_sw}")
    
    # 5. 시각화
    plt.figure(figsize=(12, 7))
    plt.plot(single_cum, label=f"Price vs SMA 150 (Total Switches: {single_sw})", alpha=0.7)
    plt.plot(dual_cum, label=f"Price vs Dual SMA 150/200 (Total Switches: {dual_sw})", color='red', linewidth=2)
    plt.yscale('log')
    plt.title("MA Logic Comparison: Stability & Performance (2018-2026)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("ma_logic_comparison_plot.png")
    
    # CSV 저장용 데이터 생성
    summary = pd.DataFrame([
        ["SMA 150 Only", *res_single, single_sw],
        ["Dual SMA 150/200", *res_dual, dual_sw]
    ], columns=["Logic", "CAGR (%)", "MDD (%)", "Sharpe", "Total Switches"])
    summary.to_csv("ma_logic_comparison_results.csv", index=False)

if __name__ == "__main__":
    compare_ma_logic()
