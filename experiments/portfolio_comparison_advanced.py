"""
포트폴리오 비교 분석: 
1. SCHD 50% + QQQ 45% + GOLD 5%
2. SCHD 38% + QQQ 38% (하락신호시 JEPI 전환) + GOLD 5% + KOSPI 19%

배당금, 세금, 동적 전환 고려
최대 기간 분석 및 차트 생성
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# matplotlib 한글 폰트 설정
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1단계: 최대 기간 데이터 수집 ============

print("="*100)
print("데이터 수집 (최대 기간)")
print("="*100)

# 10년 데이터 시도
end_date = datetime.now().date()
start_date = end_date - timedelta(days=3650)

print(f"\n분석 기간 시도: {start_date} ~ {end_date}\n")

tickers_to_download = ['SCHD', 'QQQ', 'JEPI', 'GLD', '^KS200']
data = {}
failed = []

for ticker in tickers_to_download:
    try:
        print(f"  {ticker}...", end=" ", flush=True)
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if len(df) > 100:  # 최소 100일 이상
            # 단일 티커의 경우 Series가 반환되므로 처리
            if isinstance(df, pd.Series):
                df = df.to_frame('Adj Close')
            elif isinstance(df, pd.DataFrame):
                # 'Adj Close' 컬럼이 없으면 첫 번째 컬럼 사용
                if 'Adj Close' not in df.columns:
                    df = df[[df.columns[0]]].rename(columns={df.columns[0]: 'Adj Close'})
            
            data[ticker] = df
            print(f"OK ({len(df)} days)")
        else:
            print(f"FAIL (insufficient data)")
            failed.append(ticker)
    except Exception as e:
        print(f"FAIL ({str(e)[:30]})")
        failed.append(ticker)

print(f"\n✅ 성공: {len(data)}/{len(tickers_to_download)}")
print(f"❌ 실패: {failed}")

if len(data) < 4:
    print("\nERROR: 필수 데이터 부족")
    exit(1)

# 공통 날짜 찾기
common_dates = None
for ticker, df in data.items():
    dates = set(df.index)
    if common_dates is None:
        common_dates = dates
    else:
        common_dates = common_dates.intersection(dates)

print(f"\n공통 데이터 기간: {min(common_dates)} ~ {max(common_dates)}")
print(f"공통 거래일: {len(common_dates)}")

# ============ 2단계: 배당금 데이터 ============

print("\n" + "="*100)
print("배당금 데이터 수집")
print("="*100)

# 배당금 정보 (최근 연간 배당금)
dividend_yields = {
    'SCHD': 0.038,  # ~3.8% annual dividend
    'QQQ': 0.004,   # ~0.4% (기술주는 배당금 적음)
    'JEPI': 0.060,  # ~6.0% (프리미엄 인컴)
    'GLD': 0.0,     # 배당금 없음
    '^KS200': 0.016  # ~1.6% (한국 평균)
}

print("\n배당금 수익률 (Annual Dividend Yield):")
for ticker, div_yield in dividend_yields.items():
    print(f"  {ticker:<10}: {div_yield:.2%}")

# ============ 3단계: 포트폴리오 계산 ============

print("\n" + "="*100)
print("포트폴리오 계산")
print("="*100)

def calculate_portfolio_with_dividends(weights, data_dict, dividend_yields, dates, tax_rate=0.15):
    """
    배당금과 세금을 고려한 포트폴리오 가치 계산
    """
    sorted_dates = sorted(list(dates))
    portfolio_value = 100000
    portfolio_values = []
    daily_values = []
    
    for i, date in enumerate(sorted_dates):
        # 일일 수익률 계산
        daily_return = 0
        total_weight = sum(weights.values())
        
        for ticker, weight in weights.items():
            if ticker in data_dict and date in data_dict[ticker].index:
                if i > 0:
                    prev_date = sorted_dates[i-1]
                    if prev_date in data_dict[ticker].index:
                        current_price = data_dict[ticker].loc[date, 'Adj Close']
                        prev_price = data_dict[ticker].loc[prev_date, 'Adj Close']
                        
                        if prev_price > 0:
                            ret = (current_price - prev_price) / prev_price
                            daily_return += (weight / total_weight) * ret
        
        # 배당금 추가 (월간, 단순화)
        if i % 21 == 0 and i > 0:  # 월간 대략 21 거래일
            monthly_dividend = 0
            for ticker, weight in weights.items():
                if ticker in dividend_yields:
                    monthly_div = dividend_yields[ticker] / 12
                    # 세금 고려 (배당금에서 15% 세금)
                    monthly_div_after_tax = monthly_div * (1 - tax_rate)
                    monthly_dividend += (weight / total_weight) * monthly_div_after_tax
            
            daily_return += monthly_dividend
        
        # 포트폴리오 가치 업데이트
        portfolio_value *= (1 + daily_return)
        portfolio_values.append(portfolio_value)
        daily_values.append({'date': date, 'value': portfolio_value})
    
    return np.array(portfolio_values), sorted_dates, pd.DataFrame(daily_values)

# 포트폴리오 1: SCHD 50% + QQQ 45% + GLD 5%
print("\n[포트폴리오 1] SCHD 50% + QQQ 45% + GLD 5%")
weights_1 = {
    'SCHD': 0.50,
    'QQQ': 0.45,
    'GLD': 0.05
}

pf1_values, pf1_dates, pf1_df = calculate_portfolio_with_dividends(weights_1, data, dividend_yields, common_dates)
print(f"  최종 가치: ${pf1_values[-1]:,.0f}")
print(f"  총 수익률: {(pf1_values[-1] - 100000) / 100000:.2%}")
print(f"  CAGR: {(pf1_values[-1] / 100000) ** (252 / len(pf1_values)) - 1:.2%}")

# 포트폴리오 2: 동적 전환 (QQQ → JEPI)
print("\n[포트폴리오 2] SCHD 38% + QQQ 38% (하락신호시 JEPI 전환) + GLD 5% + KOSPI 19%")
print("  (동적 전환 로직 계산 중...)")

def calculate_dynamic_portfolio(weights_base, data_dict, dividend_yields, dates, tax_rate=0.15):
    """
    QQQ → JEPI 동적 전환 포트폴리오
    - QQQ가 50일 이동평균 아래로 내려가면 JEPI로 전환
    - JEPI가 50일 이동평균 위로 올라가면 다시 QQQ로 전환
    """
    sorted_dates = sorted(list(dates))
    portfolio_value = 100000
    portfolio_values = []
    switch_dates = []
    current_ticker = 'QQQ'  # 초기값
    
    for i, date in enumerate(sorted_dates):
        # 50일 이동평균 계산
        start_idx = max(0, i - 50)
        
        if i >= 50 and 'QQQ' in data_dict and 'JEPI' in data_dict:
            if date in data_dict['QQQ'].index and date in data_dict['JEPI'].index:
                qqq_prices = data_dict['QQQ'].loc[sorted_dates[start_idx:i+1], 'Adj Close'].values
                jepi_prices = data_dict['JEPI'].loc[sorted_dates[start_idx:i+1], 'Adj Close'].values
                
                qqq_current = qqq_prices[-1]
                qqq_ma50 = np.mean(qqq_prices)
                
                jepi_current = jepi_prices[-1]
                jepi_ma50 = np.mean(jepi_prices)
                
                # 전환 신호
                if current_ticker == 'QQQ' and qqq_current < qqq_ma50 * 0.95:  # 5% 이상 하락
                    current_ticker = 'JEPI'
                    switch_dates.append((date, 'QQQ→JEPI'))
                
                elif current_ticker == 'JEPI' and jepi_current > jepi_ma50 * 1.05:  # 5% 이상 상승
                    current_ticker = 'QQQ'
                    switch_dates.append((date, 'JEPI→QQQ'))
        
        # 일일 수익률 계산
        daily_return = 0
        
        # 동적으로 가중치 업데이트
        weights_dynamic = weights_base.copy()
        if current_ticker == 'JEPI':
            weights_dynamic['JEPI'] = weights_dynamic.pop('QQQ')
        
        total_weight = sum(weights_dynamic.values())
        
        for ticker, weight in weights_dynamic.items():
            if ticker in data_dict and date in data_dict[ticker].index:
                if i > 0:
                    prev_date = sorted_dates[i-1]
                    if prev_date in data_dict[ticker].index:
                        current_price = data_dict[ticker].loc[date, 'Adj Close']
                        prev_price = data_dict[ticker].loc[prev_date, 'Adj Close']
                        
                        if prev_price > 0:
                            ret = (current_price - prev_price) / prev_price
                            daily_return += (weight / total_weight) * ret
        
        # 배당금 추가 (월간)
        if i % 21 == 0 and i > 0:
            monthly_dividend = 0
            for ticker, weight in weights_dynamic.items():
                if ticker in dividend_yields:
                    monthly_div = dividend_yields[ticker] / 12
                    monthly_div_after_tax = monthly_div * (1 - tax_rate)
                    monthly_dividend += (weight / total_weight) * monthly_div_after_tax
            
            daily_return += monthly_dividend
        
        portfolio_value *= (1 + daily_return)
        portfolio_values.append(portfolio_value)
    
    return np.array(portfolio_values), sorted_dates, switch_dates

pf2_values, pf2_dates, switch_points = calculate_dynamic_portfolio(
    {'SCHD': 0.38, 'QQQ': 0.38, 'GLD': 0.05, '^KS200': 0.19},
    data, dividend_yields, common_dates
)

print(f"  최종 가치: ${pf2_values[-1]:,.0f}")
print(f"  총 수익률: {(pf2_values[-1] - 100000) / 100000:.2%}")
print(f"  CAGR: {(pf2_values[-1] / 100000) ** (252 / len(pf2_values)) - 1:.2%}")
print(f"  전환 횟수: {len(switch_points)}")
if switch_points:
    print(f"  최근 전환: {switch_points[-1]}")

# ============ 4단계: 차트 생성 ============

print("\n" + "="*100)
print("차트 생성")
print("="*100)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 차트 1: 포트폴리오 1 가치 추이
ax1 = axes[0, 0]
ax1.plot(pf1_dates, pf1_values, linewidth=2, label='Portfolio 1', color='blue')
ax1.fill_between(pf1_dates, 100000, pf1_values, alpha=0.3, color='blue')
ax1.set_title('Portfolio 1: SCHD 50% + QQQ 45% + GLD 5%', fontsize=12, fontweight='bold')
ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axhline(y=100000, color='red', linestyle='--', alpha=0.5, label='Initial Investment')

# 차트 2: 포트폴리오 2 가치 추이 (전환 시점 표시)
ax2 = axes[0, 1]
ax2.plot(pf2_dates, pf2_values, linewidth=2, label='Portfolio 2 (Dynamic)', color='green')
ax2.fill_between(pf2_dates, 100000, pf2_values, alpha=0.3, color='green')

# 전환 시점 표시
for switch_date, switch_type in switch_points:
    idx = list(pf2_dates).index(switch_date)
    if idx < len(pf2_values):
        ax2.scatter(switch_date, pf2_values[idx], color='red', s=100, zorder=5, marker='v' if 'JEPI' in switch_type else '^')

ax2.set_title('Portfolio 2: SCHD 38% + QQQ 38% (↔ JEPI) + GLD 5% + KOSPI 19%\n(Red markers = Switch points)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Portfolio Value ($)', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=100000, color='red', linestyle='--', alpha=0.5, label='Initial Investment')

# 차트 3: 직접 비교
ax3 = axes[1, 0]
ax3.plot(pf1_dates, pf1_values, linewidth=2, label='Portfolio 1 (SCHD 50% + QQQ 45% + GLD 5%)', color='blue')
ax3.plot(pf2_dates, pf2_values, linewidth=2, label='Portfolio 2 (Dynamic with KOSPI)', color='green')
ax3.set_title('Direct Comparison', fontsize=12, fontweight='bold')
ax3.set_ylabel('Portfolio Value ($)', fontsize=10)
ax3.set_xlabel('Date', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.axhline(y=100000, color='red', linestyle='--', alpha=0.5, label='Initial Investment')

# 차트 4: 누적 수익률
ax4 = axes[1, 1]
pf1_returns = (pf1_values - 100000) / 100000 * 100
pf2_returns = (pf2_values - 100000) / 100000 * 100
ax4.plot(pf1_dates, pf1_returns, linewidth=2, label='Portfolio 1', color='blue')
ax4.plot(pf2_dates, pf2_returns, linewidth=2, label='Portfolio 2', color='green')
ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax4.set_title('Cumulative Return (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Return (%)', fontsize=10)
ax4.set_xlabel('Date', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
plt.savefig('portfolio_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ 차트 저장: portfolio_comparison.png")
plt.close()

# ============ 5단계: 상세 통계 ============

print("\n" + "="*100)
print("상세 성과 통계")
print("="*100)

print("\n[포트폴리오 1: SCHD 50% + QQQ 45% + GLD 5%]")
pf1_return_pct = (pf1_values[-1] - 100000) / 100000 * 100
pf1_years = len(pf1_values) / 252
pf1_cagr = (pf1_values[-1] / 100000) ** (1 / pf1_years) - 1 if pf1_years > 0 else 0
pf1_daily_returns = np.diff(pf1_values) / pf1_values[:-1]
pf1_volatility = np.std(pf1_daily_returns) * np.sqrt(252)
pf1_sharpe = pf1_cagr / pf1_volatility if pf1_volatility > 0 else 0

print(f"  최종 가치: ${pf1_values[-1]:,.0f}")
print(f"  총 수익률: {pf1_return_pct:.2f}%")
print(f"  CAGR: {pf1_cagr:.2%}")
print(f"  연간 변동성: {pf1_volatility:.2%}")
print(f"  Sharpe 비율: {pf1_sharpe:.3f}")

print("\n[포트폴리오 2: SCHD 38% + QQQ 38% (↔JEPI) + GLD 5% + KOSPI 19%]")
pf2_return_pct = (pf2_values[-1] - 100000) / 100000 * 100
pf2_years = len(pf2_values) / 252
pf2_cagr = (pf2_values[-1] / 100000) ** (1 / pf2_years) - 1 if pf2_years > 0 else 0
pf2_daily_returns = np.diff(pf2_values) / pf2_values[:-1]
pf2_volatility = np.std(pf2_daily_returns) * np.sqrt(252)
pf2_sharpe = pf2_cagr / pf2_volatility if pf2_volatility > 0 else 0

print(f"  최종 가치: ${pf2_values[-1]:,.0f}")
print(f"  총 수익률: {pf2_return_pct:.2f}%")
print(f"  CAGR: {pf2_cagr:.2%}")
print(f"  연간 변동성: {pf2_volatility:.2%}")
print(f"  Sharpe 비율: {pf2_sharpe:.3f}")
print(f"  전환 횟수: {len(switch_points)}")

# 비교
print("\n[비교]")
value_diff = pf2_values[-1] - pf1_values[-1]
print(f"  포트폴리오 2가 {abs(value_diff):.0f}만큼 {'높음' if value_diff > 0 else '낮음'}")
print(f"  Sharpe 비율 차이: {pf2_sharpe - pf1_sharpe:.3f}")

print("\n" + "="*100)
