# -*- coding: utf-8 -*-
"""
ML 하이브리드 vs 기존 기본 시그널 성과 비교 백테스트
- 비교군: Optimized Basic (M1 Only) vs ML Enhanced (M1 + M2 Refiner)
- 비용: 슬리피지(0.1%), 수수료(0.01%), 세금(연간 22%)
- 초기 자산: $100,000
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 데이터 로드 (ml_signal_refiner.py에서 생성됨)
df = pd.read_csv("ml_augmented_signals.csv", index_col=0)
df.index = pd.to_datetime(df.index)

# 설정
INITIAL_CAPITAL = 100000
TOTAL_COST = 0.0011 # 슬리피지 + 수수료
TAX_RATE = 0.22
TAX_DEDUCTION = 1724 # $2500/환율

def backtest(signal_col, name):
    cash = INITIAL_CAPITAL
    shares = 0
    in_position = False
    history = []
    
    # 단순화된 백테스트 (SPY만 대상으로 시뮬레이션 - 필터링 효과 확인 목적)
    for i in range(len(df)):
        price = df['SPY_Close'].iloc[i]
        signal = df[signal_col].iloc[i] # 1: Danger, 0: Normal
        
        # 0(Normal)이면 매수/보유, 1(Danger)이면 매도/현금
        if signal == 0 and not in_position:
            # 매수 (Normal 진입)
            shares = (cash * (1-TOTAL_COST)) / price
            cash = 0
            in_position = True
        elif signal == 1 and in_position:
            # 매도 (Danger 탈출)
            cash = shares * price * (1-TOTAL_COST)
            shares = 0
            in_position = False
        
        # 가치 기록
        current_val = cash + (shares * price if in_position else 0)
        history.append(current_val)
        
    return pd.Series(history, index=df.index)

def analyze(series):
    days = (series.index[-1] - series.index[0]).days
    cagr = (series.iloc[-1] / series.iloc[0]) ** (365/days) - 1
    mdd = (series / series.cummax() - 1).min()
    daily_ret = series.pct_change()
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252))
    return {'Final': series.iloc[-1], 'CAGR': cagr*100, 'MDD': mdd*100, 'Sharpe': sharpe}

# 실행
print("Comparing performance...")
res_m1 = backtest('M1_Signal', 'Basic')
res_ml = backtest('Hybrid_Signal', 'ML_Enhanced')

stats_m1 = analyze(res_m1)
stats_ml = analyze(res_ml)

print("\n" + "="*50)
print(f"{'Strategy':<15} | {'Final ($)':<12} | {'CAGR (%)':<10} | {'MDD (%)':<10} | {'Sharpe':<8}")
print("-" * 65)
print(f"{'Basic (M1)':<15} | {stats_m1['Final']:>12,.0f} | {stats_m1['CAGR']:>10.2f} | {stats_m1['MDD']:>10.2f} | {stats_m1['Sharpe']:>8.2f}")
print(f"{'ML Hybrid':<15} | {stats_ml['Final']:>12,.0f} | {stats_ml['CAGR']:>10.2f} | {stats_ml['MDD']:>10.2f} | {stats_ml['Sharpe']:>8.2f}")
print("="*50)

# 차트
plt.figure(figsize=(12, 7))
plt.plot(res_m1, label=f"Basic (Sharpe: {stats_m1['Sharpe']:.2f})", alpha=0.8)
plt.plot(res_ml, label=f"ML Hybrid (Sharpe: {stats_ml['Sharpe']:.2f})", linewidth=2)
plt.title("Performance Comparison: Basic vs ML Enhanced Signal")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("ml_comparison_results.png")
print("\nChart saved: ml_comparison_results.png")

# 결과 텍스트 저장
with open("ml_comparison_report.txt", "w", encoding="utf-8") as f:
    f.write("Performance Comparison Report\n")
    f.write(f"Basic Only: Final ${stats_m1['Final']:,.0f}, CAGR {stats_m1['CAGR']:.2f}%, MDD {stats_m1['MDD']:.2f}%\n")
    f.write(f"ML Enhanced: Final ${stats_ml['Final']:,.0f}, CAGR {stats_ml['CAGR']:.2f}%, MDD {stats_ml['MDD']:.2f}%\n")
