# -*- coding: utf-8 -*-
"""
시그널 임계값 최적화
- MA percentile, Volatility percentile 조합 최적화
- 다양한 윈도우 크기 테스트
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import product

# Configuration
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100000

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
    
    df_aligned = pd.DataFrame(data).dropna()
    print(f"\n정렬된 데이터: {len(df_aligned)} rows ({df_aligned.index[0].date()} ~ {df_aligned.index[-1].date()})")
    return df_aligned

def calculate_signal_with_params(df, ma_window=20, vol_window=20, ma_percentile=25, vol_percentile=75, min_periods=252):
    """
    파라미터 기반 시그널 계산
    
    Args:
        ma_window: MA 계산 윈도우
        vol_window: Volatility 계산 윈도우
        ma_percentile: MA 임계값 percentile (이하면 Danger)
        vol_percentile: Vol 임계값 percentile (이상이면 Danger)
        min_periods: 최소 워밍업 기간
    """
    # 파라미터 검증
    ma_window = int(max(1, ma_window))
    vol_window = int(max(1, vol_window))
    min_periods = int(max(ma_window, vol_window, min_periods))
    
    spy = df['SPY']
    log_ret = np.log(spy / spy.shift(1))
    
    ma = log_ret.rolling(window=ma_window).mean()
    vol = log_ret.rolling(window=vol_window).std()
    
    signals = []
    history_ma = []
    history_vol = []
    
    for i in range(len(df)):
        if i < min_periods:
            signals.append(0)
            if not np.isnan(ma.iloc[i]):
                history_ma.append(ma.iloc[i])
            if not np.isnan(vol.iloc[i]):
                history_vol.append(vol.iloc[i])
            continue
        
        current_ma = ma.iloc[i]
        current_vol = vol.iloc[i]
        
        if len(history_ma) > 0:
            p_ma = np.nanpercentile(history_ma, ma_percentile)
            p_vol = np.nanpercentile(history_vol, vol_percentile)
            
            is_danger = (current_ma < p_ma) or (current_vol > p_vol)
            signals.append(1 if is_danger else 0)
        else:
            signals.append(0)
        
        history_ma.append(current_ma)
        history_vol.append(current_vol)
    
    return pd.Series(signals, index=df.index, name='is_danger')

def backtest_with_signal(df, is_danger, core_ticker='SPY'):
    """시그널 기반 백테스트"""
    weights = {
        'CORE': 0.38,
        'DYNAMIC': 0.38,
        'GOLD': 0.05,
        'KOSPI': 0.19
    }
    
    shares = {
        'CORE': 0,
        'QQQ': 0,
        'JEPI': 0,
        'GOLD': 0,
        'KOSPI': 0
    }
    
    # 초기 배분
    first_prices = df.iloc[0]
    shares['CORE'] = (INITIAL_CAPITAL * weights['CORE']) / first_prices[core_ticker]
    shares['QQQ'] = (INITIAL_CAPITAL * weights['DYNAMIC']) / first_prices['QQQ']
    shares['GOLD'] = (INITIAL_CAPITAL * weights['GOLD']) / first_prices['GOLD']
    shares['KOSPI'] = (INITIAL_CAPITAL * weights['KOSPI']) / first_prices['KOSPI']
    
    current_mode = 0
    portfolio_values = []
    
    for i in range(len(df)):
        prices = df.iloc[i]
        signal = is_danger.iloc[i]
        
        # 포트폴리오 가치
        core_value = shares['CORE'] * prices[core_ticker]
        dynamic_value = shares['QQQ'] * prices['QQQ'] + shares['JEPI'] * prices['JEPI']
        gold_value = shares['GOLD'] * prices['GOLD']
        kospi_value = shares['KOSPI'] * prices['KOSPI']
        
        total_value = core_value + dynamic_value + gold_value + kospi_value
        portfolio_values.append(total_value)
        
        # 리밸런싱
        if signal != current_mode:
            if signal == 1:  # QQQ -> JEPI
                shares['JEPI'] = dynamic_value / prices['JEPI']
                shares['QQQ'] = 0
            else:  # JEPI -> QQQ
                shares['QQQ'] = dynamic_value / prices['QQQ']
                shares['JEPI'] = 0
            
            current_mode = signal
    
    return pd.Series(portfolio_values, index=df.index)

def analyze_performance(series):
    """성과 분석"""
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    
    days = (series.index[-1] - series.index[0]).days
    cagr = (series.iloc[-1] / series.iloc[0]) ** (365/days) - 1
    
    peak = series.cummax()
    dd = (series - peak) / peak
    mdd = dd.min()
    
    daily_ret = series.pct_change()
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252))
    
    annual_vol = daily_ret.std() * np.sqrt(252)
    
    return {
        'Final Value': series.iloc[-1],
        'Total Return': total_ret * 100,
        'CAGR': cagr * 100,
        'MDD': mdd * 100,
        'Sharpe': sharpe,
        'Volatility': annual_vol * 100
    }

def optimize_parameters(df):
    """파라미터 최적화"""
    print("\n🔍 파라미터 최적화 시작...")
    
    # 파라미터 그리드
    ma_windows = [10, 15, 20, 30]
    vol_windows = [10, 15, 20, 30]
    ma_percentiles = [15, 20, 25, 30, 35]
    vol_percentiles = [65, 70, 75, 80, 85]
    
    results = []
    total_combinations = len(list(product(ma_windows, vol_windows, ma_percentiles, vol_percentiles)))
    
    print(f"  총 {total_combinations}개 조합 테스트 중...")
    
    for i, (ma_w, vol_w, ma_p, vol_p) in enumerate(product(ma_windows, vol_windows, ma_percentiles, vol_percentiles)):
        if i % 50 == 0:
            print(f"  진행: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
        
        # 시그널 생성
        is_danger = calculate_signal_with_params(df, ma_w, vol_w, ma_p, vol_p)
        
        # 백테스트 (SPY 기준)
        portfolio = backtest_with_signal(df, is_danger, 'SPY')
        stats = analyze_performance(portfolio)
        
        # 위험 신호 비율
        danger_ratio = is_danger.sum() / len(is_danger) * 100
        
        # 거래 횟수
        switches = (is_danger.diff() != 0).sum()
        
        results.append({
            'ma_window': ma_w,
            'vol_window': vol_w,
            'ma_percentile': ma_p,
            'vol_percentile': vol_p,
            'danger_ratio': danger_ratio,
            'switches': switches,
            **stats
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n  ✓ 최적화 완료!")
    return results_df

def main():
    print("=" * 70)
    print("🎯 시그널 파라미터 최적화")
    print("=" * 70)
    
    # 데이터 수집
    df = fetch_data()
    
    # 최적화 실행
    results_df = optimize_parameters(df)
    
    # 결과 정렬 (Sharpe Ratio 기준)
    results_sharpe = results_df.sort_values('Sharpe', ascending=False)
    
    # CAGR 기준
    results_cagr = results_df.sort_values('CAGR', ascending=False)
    
    # Sharpe/MDD 균형 (Sharpe - |MDD|/10)
    results_df['Score'] = results_df['Sharpe'] - abs(results_df['MDD']) / 10
    results_balanced = results_df.sort_values('Score', ascending=False)
    
    # Top 10 출력 (Sharpe 기준)
    print("\n" + "=" * 70)
    print("🏆 Top 10 최적 파라미터 (Sharpe Ratio 기준)")
    print("=" * 70)
    print(f"{'Rank':<5} {'MA_W':<6} {'Vol_W':<7} {'MA_P':<6} {'Vol_P':<7} {'Danger%':<9} {'Switches':<9} {'Sharpe':<8} {'CAGR':<8}")
    print("-" * 70)
    
    for idx, (i, row) in enumerate(results_sharpe.head(10).iterrows()):
        print(f"{idx+1:<5} {row['ma_window']:<6.0f} {row['vol_window']:<7.0f} {row['ma_percentile']:<6.0f} "
              f"{row['vol_percentile']:<7.0f} {row['danger_ratio']:<9.1f} {row['switches']:<9.0f} "
              f"{row['Sharpe']:<8.2f} {row['CAGR']:<8.2f}")
    
    # 최적 파라미터 (Sharpe 기준)
    best_sharpe = results_sharpe.iloc[0]
    best_cagr = results_cagr.iloc[0]
    best_balanced = results_balanced.iloc[0]
    
    print("\n" + "=" * 70)
    print("🎯 최적 파라미터 (Sharpe 최대화)")
    print("=" * 70)
    print(f"  MA Window: {best_sharpe['ma_window']:.0f}")
    print(f"  Vol Window: {best_sharpe['vol_window']:.0f}")
    print(f"  MA Percentile: {best_sharpe['ma_percentile']:.0f}")
    print(f"  Vol Percentile: {best_sharpe['vol_percentile']:.0f}")
    print(f"  Danger Signal: {best_sharpe['danger_ratio']:.1f}%")
    print(f"  Switches: {best_sharpe['switches']:.0f}회")
    print(f"\n  성과:")
    print(f"    Final Value: ${best_sharpe['Final Value']:,.0f}")
    print(f"    CAGR: {best_sharpe['CAGR']:.2f}%")
    print(f"    MDD: {best_sharpe['MDD']:.2f}%")
    print(f"    Sharpe: {best_sharpe['Sharpe']:.2f}")
    
    print("\n" + "=" * 70)
    print("🎯 최적 파라미터 (CAGR 최대화)")
    print("=" * 70)
    print(f"  MA Window: {best_cagr['ma_window']:.0f}")
    print(f"  Vol Window: {best_cagr['vol_window']:.0f}")
    print(f"  MA Percentile: {best_cagr['ma_percentile']:.0f}")
    print(f"  Vol Percentile: {best_cagr['vol_percentile']:.0f}")
    print(f"  Danger Signal: {best_cagr['danger_ratio']:.1f}%")
    print(f"\n  성과:")
    print(f"    Final Value: ${best_cagr['Final Value']:,.0f}")
    print(f"    CAGR: {best_cagr['CAGR']:.2f}%")
    print(f"    MDD: {best_cagr['MDD']:.2f}%")
    print(f"    Sharpe: {best_cagr['Sharpe']:.2f}")
    
    # 최적 파라미터로 SCHD vs SPY 비교
    print("\n" + "=" * 70)
    print("📊 최적 파라미터 백테스트: SCHD vs SPY")
    print("=" * 70)
    
    optimal_signal = calculate_signal_with_params(
        df,
        best_sharpe['ma_window'],
        best_sharpe['vol_window'],
        best_sharpe['ma_percentile'],
        best_sharpe['vol_percentile']
    )
    
    portfolio_schd = backtest_with_signal(df, optimal_signal, 'SCHD')
    portfolio_spy = backtest_with_signal(df, optimal_signal, 'SPY')
    
    stats_schd = analyze_performance(portfolio_schd)
    stats_spy = analyze_performance(portfolio_spy)
    
    print(f"\n{'Metric':<20} {'SCHD (Optimized)':<20} {'SPY (Optimized)':<20}")
    print("-" * 60)
    print(f"{'Final Value':<20} ${stats_schd['Final Value']:>18,.0f} ${stats_spy['Final Value']:>18,.0f}")
    print(f"{'CAGR':<20} {stats_schd['CAGR']:>18.2f}% {stats_spy['CAGR']:>18.2f}%")
    print(f"{'MDD':<20} {stats_schd['MDD']:>18.2f}% {stats_spy['MDD']:>18.2f}%")
    print(f"{'Sharpe':<20} {stats_schd['Sharpe']:>19.2f} {stats_spy['Sharpe']:>19.2f}")
    
    # 차트 생성
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 포트폴리오 가치
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(portfolio_schd.index, portfolio_schd, label='SCHD (Optimized)', linewidth=2)
    ax1.plot(portfolio_spy.index, portfolio_spy, label='SPY (Optimized)', linewidth=2, alpha=0.8)
    ax1.set_title('최적화된 시그널 백테스트: SCHD vs SPY', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 시그널
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(df.index, 0, 1, where=optimal_signal==1, alpha=0.3, color='red', label='Danger (JEPI)')
    ax2.fill_between(df.index, 0, 1, where=optimal_signal==0, alpha=0.3, color='green', label='Normal (QQQ)')
    ax2.set_title(f'최적 시그널 (Danger: {best_sharpe["danger_ratio"]:.1f}%, Switches: {best_sharpe["switches"]:.0f}회)', fontsize=12)
    ax2.set_ylabel('Signal')
    ax2.set_xlabel('Date')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Sharpe Heatmap (MA Percentile vs Vol Percentile)
    ax3 = fig.add_subplot(gs[2, 0])
    pivot_sharpe = results_df.pivot_table(
        values='Sharpe',
        index='ma_percentile',
        columns='vol_percentile',
        aggfunc='mean'
    )
    sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax3, cbar_kws={'label': 'Sharpe'})
    ax3.set_title('Sharpe Ratio Heatmap', fontsize=12)
    ax3.set_xlabel('Vol Percentile')
    ax3.set_ylabel('MA Percentile')
    
    # 4. CAGR Heatmap
    ax4 = fig.add_subplot(gs[2, 1])
    pivot_cagr = results_df.pivot_table(
        values='CAGR',
        index='ma_percentile',
        columns='vol_percentile',
        aggfunc='mean'
    )
    sns.heatmap(pivot_cagr, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax4, cbar_kws={'label': 'CAGR (%)'})
    ax4.set_title('CAGR Heatmap', fontsize=12)
    ax4.set_xlabel('Vol Percentile')
    ax4.set_ylabel('MA Percentile')
    
    plt.savefig('signal_optimization_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 차트 저장: signal_optimization_results.png")
    
    # 결과 저장
    results_df.to_csv('signal_optimization_full_results.csv', index=False)
    print(f"📄 전체 결과 저장: signal_optimization_full_results.csv")
    
    with open('signal_optimization_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("시그널 파라미터 최적화 결과\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"최적 파라미터 (Sharpe 최대화):\n")
        f.write(f"  MA Window: {best_sharpe['ma_window']:.0f}\n")
        f.write(f"  Vol Window: {best_sharpe['vol_window']:.0f}\n")
        f.write(f"  MA Percentile: {best_sharpe['ma_percentile']:.0f}\n")
        f.write(f"  Vol Percentile: {best_sharpe['vol_percentile']:.0f}\n")
        f.write(f"  Danger Signal: {best_sharpe['danger_ratio']:.1f}%\n")
        f.write(f"  Switches: {best_sharpe['switches']:.0f}회\n\n")
        f.write(f"성과 (SPY 기준):\n")
        f.write(f"  Final Value: ${best_sharpe['Final Value']:,.0f}\n")
        f.write(f"  CAGR: {best_sharpe['CAGR']:.2f}%\n")
        f.write(f"  MDD: {best_sharpe['MDD']:.2f}%\n")
        f.write(f"  Sharpe: {best_sharpe['Sharpe']:.2f}\n")
    
    print(f"📄 요약 저장: signal_optimization_summary.txt\n")

if __name__ == "__main__":
    main()
