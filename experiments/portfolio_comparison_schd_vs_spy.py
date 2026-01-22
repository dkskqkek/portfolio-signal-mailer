# -*- coding: utf-8 -*-
"""
Portfolio Comparison: SCHD vs SPY
포트폴리오 A: SCHD 38% + QQQ/JEPI 38% + GOLD 5% + KOSPI 19%
포트폴리오 B: SPY 38% + QQQ/JEPI 38% + GOLD 5% + KOSPI 19%
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100000

# Tickers
TICKERS = {
    'SPY': 'SPY',
    'SCHD': 'SCHD',
    'QQQ': 'QQQ',
    'JEPI': 'JEPI',
    'GOLD': 'GLD',
    'KOSPI': '^KS200'
}

def fetch_data():
    """데이터 수집"""
    print("📊 데이터 수집 중...")
    data = {}
    
    for key, ticker in TICKERS.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                data[key] = df['Close'][ticker]
            else:
                data[key] = df['Close']
            print(f"  ✓ {ticker}: {len(data[key])} rows")
        except Exception as e:
            print(f"  ✗ {ticker} 로드 실패: {e}")
    
    # 날짜 정렬
    df_aligned = pd.DataFrame(data).dropna()
    print(f"\n정렬된 데이터: {len(df_aligned)} rows ({df_aligned.index[0].date()} ~ {df_aligned.index[-1].date()})")
    return df_aligned

def calculate_signals(df):
    """SPY 기반 위험 신호 계산"""
    print("\n🔍 시그널 계산 중...")
    
    spy = df['SPY']
    log_ret = np.log(spy / spy.shift(1))
    
    # 20일 이동평균 및 변동성
    window = 20
    roll_mean = log_ret.rolling(window=window).mean()
    roll_vol = log_ret.rolling(window=window).std()
    
    min_periods = 252  # 1년 워밍업
    signals = []
    history_ma = []
    history_vol = []
    
    for i in range(len(df)):
        if i < min_periods:
            signals.append(0)  # Normal
            if not np.isnan(roll_mean.iloc[i]):
                history_ma.append(roll_mean.iloc[i])
            if not np.isnan(roll_vol.iloc[i]):
                history_vol.append(roll_vol.iloc[i])
            continue
        
        current_ma = roll_mean.iloc[i]
        current_vol = roll_vol.iloc[i]
        
        p25_ma = np.nanpercentile(history_ma, 25)
        p75_vol = np.nanpercentile(history_vol, 75)
        
        # DANGER 조건: MA < 25% OR Vol > 75%
        is_danger = (current_ma < p25_ma) or (current_vol > p75_vol)
        signals.append(1 if is_danger else 0)
        
        history_ma.append(current_ma)
        history_vol.append(current_vol)
    
    df['is_danger'] = signals
    danger_days = sum(signals)
    print(f"  위험 신호 발생: {danger_days}일 ({danger_days/len(signals)*100:.1f}%)")
    return df

def backtest_portfolio(df, core_ticker, portfolio_name):
    """
    포트폴리오 백테스트
    
    Args:
        core_ticker: 'SCHD' or 'SPY'
        portfolio_name: 포트폴리오 이름
    
    Returns:
        포트폴리오 가치 시계열
    """
    print(f"\n💼 {portfolio_name} 백테스트 실행 중...")
    
    # 가중치: Core 38%, QQQ/JEPI 38%, GOLD 5%, KOSPI 19%
    weights = {
        'CORE': 0.38,
        'DYNAMIC': 0.38,  # QQQ or JEPI
        'GOLD': 0.05,
        'KOSPI': 0.19
    }
    
    capital = INITIAL_CAPITAL
    shares = {
        'CORE': 0,
        'QQQ': 0,
        'JEPI': 0,
        'GOLD': 0,
        'KOSPI': 0
    }
    
    # 초기 배분 (Normal 상태)
    first_prices = df.iloc[0]
    shares['CORE'] = (capital * weights['CORE']) / first_prices[core_ticker]
    shares['QQQ'] = (capital * weights['DYNAMIC']) / first_prices['QQQ']
    shares['GOLD'] = (capital * weights['GOLD']) / first_prices['GOLD']
    shares['KOSPI'] = (capital * weights['KOSPI']) / first_prices['KOSPI']
    shares['JEPI'] = 0
    
    current_mode = 0  # 0: Normal (QQQ), 1: Danger (JEPI)
    portfolio_values = []
    
    for i in range(len(df)):
        prices = df.iloc[i]
        signal = df['is_danger'].iloc[i]
        
        # 현재 포트폴리오 가치 계산
        core_value = shares['CORE'] * prices[core_ticker]
        dynamic_value = shares['QQQ'] * prices['QQQ'] + shares['JEPI'] * prices['JEPI']
        gold_value = shares['GOLD'] * prices['GOLD']
        kospi_value = shares['KOSPI'] * prices['KOSPI']
        
        total_value = core_value + dynamic_value + gold_value + kospi_value
        portfolio_values.append(total_value)
        
        # 신호 변경 시 리밸런싱
        if signal != current_mode:
            if signal == 1:  # Normal -> Danger: QQQ -> JEPI
                shares['JEPI'] = dynamic_value / prices['JEPI']
                shares['QQQ'] = 0
            else:  # Danger -> Normal: JEPI -> QQQ
                shares['QQQ'] = dynamic_value / prices['QQQ']
                shares['JEPI'] = 0
            
            current_mode = signal
    
    return pd.Series(portfolio_values, index=df.index, name=portfolio_name)

def analyze_performance(series):
    """성과 분석"""
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    
    # CAGR
    days = (series.index[-1] - series.index[0]).days
    cagr = (series.iloc[-1] / series.iloc[0]) ** (365/days) - 1
    
    # MDD
    peak = series.cummax()
    dd = (series - peak) / peak
    mdd = dd.min()
    
    # Sharpe
    daily_ret = series.pct_change()
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252))
    
    # Volatility
    annual_vol = daily_ret.std() * np.sqrt(252)
    
    return {
        'Final Value': series.iloc[-1],
        'Total Return': total_ret * 100,
        'CAGR': cagr * 100,
        'MDD': mdd * 100,
        'Sharpe': sharpe,
        'Volatility': annual_vol * 100
    }

def main():
    print("=" * 60)
    print("📈 포트폴리오 비교 백테스트: SCHD vs SPY")
    print("=" * 60)
    
    # 데이터 수집
    df = fetch_data()
    
    # 시그널 계산
    df = calculate_signals(df)
    
    # 포트폴리오 A: SCHD 기반
    portfolio_a = backtest_portfolio(df, 'SCHD', 'Portfolio A (SCHD)')
    
    # 포트폴리오 B: SPY 기반
    portfolio_b = backtest_portfolio(df, 'SPY', 'Portfolio B (SPY)')
    
    # 성과 분석
    stats_a = analyze_performance(portfolio_a)
    stats_b = analyze_performance(portfolio_b)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 백테스트 결과")
    print("=" * 60)
    print(f"{'Metric':<20} {'Portfolio A (SCHD)':<20} {'Portfolio B (SPY)':<20}")
    print("-" * 60)
    
    metrics = ['Final Value', 'Total Return', 'CAGR', 'MDD', 'Sharpe', 'Volatility']
    for metric in metrics:
        val_a = stats_a[metric]
        val_b = stats_b[metric]
        
        if metric == 'Final Value':
            print(f"{metric:<20} ${val_a:>18,.0f} ${val_b:>18,.0f}")
        elif metric in ['Sharpe']:
            print(f"{metric:<20} {val_a:>19.2f} {val_b:>19.2f}")
        else:
            print(f"{metric:<20} {val_a:>18.2f}% {val_b:>18.2f}%")
    
    # 승자 판정
    print("\n" + "=" * 60)
    print("🏆 승자 판정")
    print("=" * 60)
    
    winner_count = {'A': 0, 'B': 0}
    
    for metric in metrics:
        val_a = stats_a[metric]
        val_b = stats_b[metric]
        
        if metric == 'MDD':  # MDD는 낮을수록 좋음
            winner = 'A' if val_a > val_b else 'B'  # 덜 마이너스가 승리
        else:
            winner = 'A' if val_a > val_b else 'B'
        
        winner_count[winner] += 1
        winner_name = 'Portfolio A (SCHD)' if winner == 'A' else 'Portfolio B (SPY)'
        print(f"{metric:<20} 🏆 {winner_name}")
    
    print("\n" + "=" * 60)
    overall_winner = 'Portfolio A (SCHD)' if winner_count['A'] > winner_count['B'] else 'Portfolio B (SPY)'
    print(f"🎯 종합 승자: {overall_winner}")
    print(f"   (Portfolio A: {winner_count['A']}승, Portfolio B: {winner_count['B']}승)")
    print("=" * 60)
    
    # 차트 생성
    plt.figure(figsize=(14, 8))
    
    # 상단: 포트폴리오 가치 비교
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_a.index, portfolio_a, label='Portfolio A (SCHD 38%)', linewidth=2)
    plt.plot(portfolio_b.index, portfolio_b, label='Portfolio B (SPY 38%)', linewidth=2, alpha=0.8)
    plt.title('Portfolio Value Comparison: SCHD vs SPY', fontsize=14, fontweight='bold')
    plt.ylabel('Portfolio Value ($)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 하단: 위험 신호 표시
    plt.subplot(2, 1, 2)
    danger_zones = df[df['is_danger'] == 1].index
    plt.fill_between(df.index, 0, 1, where=df['is_danger']==1, alpha=0.3, color='red', label='Danger (JEPI)')
    plt.fill_between(df.index, 0, 1, where=df['is_danger']==0, alpha=0.3, color='green', label='Normal (QQQ)')
    plt.title('Market Regime (QQQ ↔ JEPI Switching)', fontsize=12)
    plt.ylabel('Signal')
    plt.xlabel('Date')
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('portfolio_comparison_schd_vs_spy.png', dpi=150)
    print(f"\n📈 차트 저장: portfolio_comparison_schd_vs_spy.png")
    
    # 리포트 파일 저장
    with open('portfolio_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("포트폴리오 비교 백테스트: SCHD vs SPY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"기간: {df.index[0].date()} ~ {df.index[-1].date()}\n")
        f.write(f"초기 자본: ${INITIAL_CAPITAL:,}\n\n")
        f.write("포트폴리오 구성:\n")
        f.write("  Portfolio A: SCHD 38% + QQQ/JEPI 38% + GOLD 5% + KOSPI 19%\n")
        f.write("  Portfolio B: SPY 38% + QQQ/JEPI 38% + GOLD 5% + KOSPI 19%\n\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Metric':<20} {'Portfolio A':<20} {'Portfolio B':<20}\n")
        f.write("-" * 60 + "\n")
        
        for metric in metrics:
            val_a = stats_a[metric]
            val_b = stats_b[metric]
            
            if metric == 'Final Value':
                f.write(f"{metric:<20} ${val_a:>18,.0f} ${val_b:>18,.0f}\n")
            elif metric in ['Sharpe']:
                f.write(f"{metric:<20} {val_a:>19.2f} {val_b:>19.2f}\n")
            else:
                f.write(f"{metric:<20} {val_a:>18.2f}% {val_b:>18.2f}%\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"종합 승자: {overall_winner}\n")
        f.write("=" * 60 + "\n")
    
    print(f"📄 리포트 저장: portfolio_comparison_report.txt\n")

if __name__ == "__main__":
    main()
