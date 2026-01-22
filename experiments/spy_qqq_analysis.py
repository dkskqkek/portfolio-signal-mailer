"""
SPY 50% + QQQ 50% 혼합 포트폴리오 분석
구조: SCHD 38% + (SPY 50% + QQQ 50%) 38% + XLRE 5%
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 분석 기간
end_date = datetime.now().date()
start_date = end_date - timedelta(days=3650)

print(f"분석 기간: {start_date} ~ {end_date}")
print("="*100)

# 데이터 다운로드
print("\n데이터 다운로드 중...\n")

tickers = ['SCHD', 'SPY', 'QQQ', 'XLRE']
data = {}

for ticker in tickers:
    try:
        print(f"  {ticker}...", end=" ", flush=True)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(df) > 0:
            if isinstance(df, pd.DataFrame):
                if 'Adj Close' in df.columns:
                    data[ticker] = df['Adj Close']
                else:
                    data[ticker] = df.iloc[:, 0]
            else:
                data[ticker] = df
            print("OK")
        else:
            print("FAIL")
    except Exception as e:
        print(f"FAIL ({str(e)[:30]})")

print(f"\n✅ 다운로드 완료: {list(data.keys())}")

# ============ 포트폴리오 계산 ============

def calculate_portfolio_value(weights, data_dict, initial_investment=100000):
    """포트폴리오 가치 계산"""
    # 공통 날짜로 정렬
    dates = None
    for prices in data_dict.values():
        if dates is None:
            dates = prices.index
        else:
            dates = dates.intersection(prices.index)
    
    portfolio_value = initial_investment
    portfolio_values = []
    
    for date in dates:
        daily_return = 0
        total_weight = sum(weights.values())
        
        for ticker, weight in weights.items():
            if ticker in data_dict and date in data_dict[ticker].index:
                price = data_dict[ticker].loc[date]
                
                if len(data_dict[ticker].loc[:date]) >= 2:
                    prev_price = data_dict[ticker].loc[:date].iloc[-2]
                    if prev_price > 0:
                        ret = (price - prev_price) / prev_price
                        daily_return += (weight / total_weight) * ret
        
        portfolio_value *= (1 + daily_return)
        portfolio_values.append(portfolio_value)
    
    return np.array(portfolio_values), dates

def calculate_metrics(portfolio_values, dates, benchmark_value=100000):
    """성과 지표 계산"""
    if len(portfolio_values) < 2:
        return {}
    
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    years = len(portfolio_values) / 252
    total_return = (portfolio_values[-1] - benchmark_value) / benchmark_value
    annual_return = (total_return + 1) ** (1 / years) - 1 if years > 0 else 0
    annual_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = (annual_return - 0.02) / annual_vol if annual_vol > 0 else 0
    
    cum_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cum_max) / cum_max
    max_dd = np.min(drawdown)
    
    cagr = (portfolio_values[-1] / benchmark_value) ** (1 / years) - 1 if years > 0 else 0
    
    return {
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'final_value': portfolio_values[-1],
        'cagr': cagr
    }

# ============ 3가지 포트폴리오 분석 ============

print("\n" + "="*100)
print("포트폴리오 구성 및 분석")
print("="*100)

portfolios = {
    'SPY 50% + QQQ 50%': {
        'SCHD': 0.38,
        'SPY': 0.19,  # 38% * 50%
        'QQQ': 0.19,  # 38% * 50%
        'XLRE': 0.05,
        # 합: 0.81 (KOSPI 19% 미포함)
    },
    'SPY 100%': {
        'SCHD': 0.38,
        'SPY': 0.38,
        'XLRE': 0.05,
    },
    'QQQ 100%': {
        'SCHD': 0.38,
        'QQQ': 0.38,
        'XLRE': 0.05,
    }
}

results = {}

for name, weights in portfolios.items():
    print(f"\n📊 {name}")
    print(f"   구성: SCHD {weights['SCHD']:.0%}", end="")
    
    if 'SPY' in weights:
        print(f" + SPY {weights['SPY']:.0%}", end="")
    if 'QQQ' in weights:
        print(f" + QQQ {weights['QQQ']:.0%}", end="")
    
    print(f" + XLRE {weights['XLRE']:.0%}")
    
    # 필요한 모든 데이터가 있는지 확인
    if all(t in data for t in weights.keys()):
        portfolio_values, dates = calculate_portfolio_value(weights, data)
        metrics = calculate_metrics(portfolio_values, dates)
        results[name] = metrics
        
        print(f"   연간 수익률: {metrics['annual_return']:.2%}")
        print(f"   연간 변동성: {metrics['annual_vol']:.2%}")
        print(f"   샤프 비율: {metrics['sharpe']:.3f}")
        print(f"   최대 낙폭: {metrics['max_dd']:.2%}")
        print(f"   최종 가치: ${metrics['final_value']:,.0f}")
    else:
        missing = [t for t in weights.keys() if t not in data]
        print(f"   ❌ 부족한 데이터: {missing}")

# ============ 비교 분석 ============

print("\n\n" + "="*100)
print("상세 비교 분석")
print("="*100)

comparison_order = ['SPY 50% + QQQ 50%', 'SPY 100%', 'QQQ 100%']

print(f"\n{'포트폴리오':<25} {'Sharpe':<10} {'Return':<12} {'Vol':<10} {'MDD':<10} {'최종가치':<15}")
print("-"*100)

for name in comparison_order:
    if name in results:
        m = results[name]
        print(f"{name:<25} {m['sharpe']:.3f}     {m['annual_return']:>9.2%}  {m['annual_vol']:>8.2%}  {m['max_dd']:>8.2%}  ${m['final_value']:>12,.0f}")

# ============ 이전 테스트 결과와 비교 ============

print("\n\n" + "="*100)
print("기존 단일 자산과 비교 (상위 5개 vs 혼합)")
print("="*100)

reference_data = {
    'XLP (현재)': {'sharpe': 0.716, 'return': 0.1154, 'vol': 0.1333, 'mdd': -0.1651, 'final': 185296},
    'VYMI (1위)': {'sharpe': 0.923, 'return': 0.1526, 'vol': 0.1437, 'mdd': -0.2023, 'final': 223006},
    'QQQ (단일)': {'sharpe': 0.859, 'return': 0.1653, 'vol': 0.1690, 'mdd': -0.2529, 'final': 237178},
    'VTV (3위)': {'sharpe': 0.836, 'return': 0.1449, 'vol': 0.1494, 'mdd': -0.1706, 'final': 214736},
}

print(f"\n{'전략':<30} {'Sharpe':<10} {'Return':<12} {'Vol':<10} {'MDD':<10} {'최종가치':<15}")
print("-"*100)

# 기존 데이터
for name, metrics in reference_data.items():
    print(f"{name:<30} {metrics['sharpe']:.3f}     {metrics['return']:>9.2%}  {metrics['vol']:>8.2%}  {metrics['mdd']:>8.2%}  ${metrics['final']:>12,.0f}")

# 신규 혼합 포트폴리오
print("\n[신규 혼합 포트폴리오]")
for name in comparison_order:
    if name in results:
        m = results[name]
        print(f"{name:<30} {m['sharpe']:.3f}     {m['annual_return']:>9.2%}  {m['annual_vol']:>8.2%}  {m['max_dd']:>8.2%}  ${m['final_value']:>12,.0f}")

# ============ 결론 ============

print("\n\n" + "="*100)
print("결론 및 평가")
print("="*100)

if 'SPY 50% + QQQ 50%' in results and 'QQQ 100%' in results:
    spy_qqq = results['SPY 50% + QQQ 50%']
    qqq_only = results['QQQ 100%']
    
    print(f"\n✅ SPY 50% + QQQ 50% (혼합) vs QQQ 100% (단일)")
    print(f"   혼합의 Sharpe: {spy_qqq['sharpe']:.3f}")
    print(f"   QQQ의 Sharpe: {qqq_only['sharpe']:.3f}")
    print(f"   → 차이: {(spy_qqq['sharpe'] - qqq_only['sharpe']):.3f}")
    print(f"\n   혼합의 변동성: {spy_qqq['annual_vol']:.2%}")
    print(f"   QQQ의 변동성: {qqq_only['annual_vol']:.2%}")
    print(f"   → 차이: {(spy_qqq['annual_vol'] - qqq_only['annual_vol']):.2%} (혼합이 낮음)")
    print(f"\n   혼합의 MDD: {spy_qqq['max_dd']:.2%}")
    print(f"   QQQ의 MDD: {qqq_only['max_dd']:.2%}")
    print(f"   → 차이: {(spy_qqq['max_dd'] - qqq_only['max_dd']):.2%} (혼합이 덜함)")

print("\n" + "="*100)
