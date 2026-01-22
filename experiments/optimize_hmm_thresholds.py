# -*- coding: utf-8 -*-
"""
HMM 시그널 임계값 최적화
- 다양한 레짐 감지 임계값 조합 테스트
- Sharpe Ratio, CAGR, MDD를 기준으로 최적 파라미터 탐색
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from itertools import product

# crash_detection_system 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'crash_detection_system' / 'src'))

from main import CrashDetectionPipeline

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

def get_hmm_signals_base(df):
    """기본 HMM 시그널 데이터 가져오기"""
    print("\n🧠 HMM 파이프라인 실행 중...")
    
    try:
        pipeline = CrashDetectionPipeline(
            ticker='SPY',
            start_date=START_DATE,
            cache_dir=str(Path(__file__).parent / 'crash_detection_system' / 'data')
        )
        
        results = pipeline.run_full_pipeline()
        
        if results['status'] != 'SUCCESS':
            print(f"  ✗ HMM 파이프라인 실패")
            return None
        
        # 레짐 및 지표 데이터 추출
        regime_df = pipeline.indicators[['HMM_Regime']].copy()
        regime_df.index = pd.to_datetime(regime_df.index).tz_localize(None)
        
        # RSI, ADX 등 추가 지표
        signal_df = pipeline.signals.copy()
        signal_df.index = pd.to_datetime(signal_df.index).tz_localize(None)
        
        # 병합
        df = df.join(regime_df, how='left')
        df = df.join(signal_df[['RSI', 'ADX']], how='left')
        
        # 결측치 처리
        df['HMM_Regime'] = df['HMM_Regime'].fillna(method='ffill').fillna(0)
        df['RSI'] = df['RSI'].fillna(50)
        df['ADX'] = df['ADX'].fillna(20)
        
        print(f"  ✓ HMM 데이터 추출 완료")
        return df
        
    except Exception as e:
        print(f"  ✗ HMM 엔진 오류: {e}")
        return None

def create_signal_with_threshold(df, regime_threshold=1, rsi_lower=30, rsi_upper=70, adx_threshold=25):
    """
    임계값 기반 시그널 생성
    
    Args:
        regime_threshold: HMM 레짐 임계값 (0: Bull, 1: Correction, 2: Crisis)
                         이 값 이상이면 Danger
        rsi_lower: RSI 하한 (이하면 과매도 -> Danger)
        rsi_upper: RSI 상한 (이상이면 과매수 -> Normal 유지)
        adx_threshold: ADX 임계값 (이하면 추세 약함 -> 신호 무시)
    
    Returns:
        is_danger 시리즈
    """
    df_copy = df.copy()
    
    # 기본 레짐 기반 신호
    regime_danger = (df_copy['HMM_Regime'] >= regime_threshold).astype(int)
    
    # RSI 조건
    rsi_danger = (df_copy['RSI'] <= rsi_lower).astype(int)
    rsi_safe = (df_copy['RSI'] >= rsi_upper).astype(int)
    
    # ADX 필터 (추세가 약하면 신호 무시)
    strong_trend = (df_copy['ADX'] >= adx_threshold).astype(int)
    
    # 종합 신호
    # Danger: (레짐이 위험 OR RSI 과매도) AND 추세가 강함
    is_danger = ((regime_danger | rsi_danger) & strong_trend).astype(int)
    
    # RSI 과매수 시 강제 Normal
    is_danger = is_danger & ~rsi_safe.astype(bool)
    
    return is_danger

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

def optimize_thresholds(df):
    """임계값 최적화"""
    print("\n🔍 임계값 최적화 시작...")
    
    # 파라미터 그리드
    regime_thresholds = [0.5, 1.0, 1.5, 2.0]  # HMM 레짐 임계값
    rsi_lowers = [20, 25, 30, 35]  # RSI 하한
    rsi_uppers = [65, 70, 75, 80]  # RSI 상한
    adx_thresholds = [15, 20, 25, 30]  # ADX 임계값
    
    results = []
    total_combinations = len(list(product(regime_thresholds, rsi_lowers, rsi_uppers, adx_thresholds)))
    
    print(f"  총 {total_combinations}개 조합 테스트 중...")
    
    for i, (regime_th, rsi_l, rsi_u, adx_th) in enumerate(product(regime_thresholds, rsi_lowers, rsi_uppers, adx_thresholds)):
        if i % 20 == 0:
            print(f"  진행: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
        
        # 시그널 생성
        is_danger = create_signal_with_threshold(df, regime_th, rsi_l, rsi_u, adx_th)
        
        # 백테스트 (SPY 기준)
        portfolio = backtest_with_signal(df, is_danger, 'SPY')
        stats = analyze_performance(portfolio)
        
        # 위험 신호 비율
        danger_ratio = is_danger.sum() / len(is_danger) * 100
        
        results.append({
            'regime_threshold': regime_th,
            'rsi_lower': rsi_l,
            'rsi_upper': rsi_u,
            'adx_threshold': adx_th,
            'danger_ratio': danger_ratio,
            **stats
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n  ✓ 최적화 완료!")
    return results_df

def main():
    print("=" * 70)
    print("🎯 HMM 시그널 임계값 최적화")
    print("=" * 70)
    
    # 데이터 수집
    df = fetch_data()
    
    # HMM 기본 데이터 가져오기
    df = get_hmm_signals_base(df)
    
    if df is None or 'HMM_Regime' not in df.columns:
        print("\n❌ HMM 데이터 로드 실패")
        return
    
    # 최적화 실행
    results_df = optimize_thresholds(df)
    
    # 결과 정렬 (Sharpe Ratio 기준)
    results_df_sorted = results_df.sort_values('Sharpe', ascending=False)
    
    # Top 10 출력
    print("\n" + "=" * 70)
    print("🏆 Top 10 최적 파라미터 (Sharpe Ratio 기준)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Regime':<8} {'RSI_L':<7} {'RSI_U':<7} {'ADX':<6} {'Danger%':<9} {'Sharpe':<8} {'CAGR':<8} {'MDD':<8}")
    print("-" * 70)
    
    for i, row in results_df_sorted.head(10).iterrows():
        print(f"{i+1:<5} {row['regime_threshold']:<8.1f} {row['rsi_lower']:<7.0f} {row['rsi_upper']:<7.0f} "
              f"{row['adx_threshold']:<6.0f} {row['danger_ratio']:<9.1f} {row['Sharpe']:<8.2f} "
              f"{row['CAGR']:<8.2f} {row['MDD']:<8.2f}")
    
    # 최적 파라미터
    best = results_df_sorted.iloc[0]
    
    print("\n" + "=" * 70)
    print("🎯 최적 파라미터")
    print("=" * 70)
    print(f"  Regime Threshold: {best['regime_threshold']:.1f}")
    print(f"  RSI Lower: {best['rsi_lower']:.0f}")
    print(f"  RSI Upper: {best['rsi_upper']:.0f}")
    print(f"  ADX Threshold: {best['adx_threshold']:.0f}")
    print(f"  Danger Signal: {best['danger_ratio']:.1f}%")
    print(f"\n  성과:")
    print(f"    Final Value: ${best['Final Value']:,.0f}")
    print(f"    CAGR: {best['CAGR']:.2f}%")
    print(f"    MDD: {best['MDD']:.2f}%")
    print(f"    Sharpe: {best['Sharpe']:.2f}")
    print(f"    Volatility: {best['Volatility']:.2f}%")
    
    # 최적 파라미터로 백테스트 실행
    print("\n" + "=" * 70)
    print("📊 최적 파라미터 백테스트 실행")
    print("=" * 70)
    
    optimal_signal = create_signal_with_threshold(
        df,
        best['regime_threshold'],
        best['rsi_lower'],
        best['rsi_upper'],
        best['adx_threshold']
    )
    
    portfolio_schd = backtest_with_signal(df, optimal_signal, 'SCHD')
    portfolio_spy = backtest_with_signal(df, optimal_signal, 'SPY')
    
    stats_schd = analyze_performance(portfolio_schd)
    stats_spy = analyze_performance(portfolio_spy)
    
    print(f"\n{'Metric':<20} {'SCHD':<20} {'SPY':<20}")
    print("-" * 60)
    print(f"{'Final Value':<20} ${stats_schd['Final Value']:>18,.0f} ${stats_spy['Final Value']:>18,.0f}")
    print(f"{'CAGR':<20} {stats_schd['CAGR']:>18.2f}% {stats_spy['CAGR']:>18.2f}%")
    print(f"{'MDD':<20} {stats_schd['MDD']:>18.2f}% {stats_spy['MDD']:>18.2f}%")
    print(f"{'Sharpe':<20} {stats_schd['Sharpe']:>19.2f} {stats_spy['Sharpe']:>19.2f}")
    
    # 차트 생성
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 포트폴리오 가치
    axes[0].plot(portfolio_schd.index, portfolio_schd, label='SCHD (Optimized)', linewidth=2)
    axes[0].plot(portfolio_spy.index, portfolio_spy, label='SPY (Optimized)', linewidth=2, alpha=0.8)
    axes[0].set_title('최적화된 HMM 시그널 백테스트', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 시그널
    axes[1].fill_between(df.index, 0, 1, where=optimal_signal==1, alpha=0.3, color='red', label='Danger (JEPI)')
    axes[1].fill_between(df.index, 0, 1, where=optimal_signal==0, alpha=0.3, color='green', label='Normal (QQQ)')
    axes[1].set_title(f'최적 시그널 (Danger: {best["danger_ratio"]:.1f}%)', fontsize=12)
    axes[1].set_ylabel('Signal')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Sharpe Ratio 히트맵 (Regime vs RSI Lower)
    pivot = results_df.pivot_table(
        values='Sharpe',
        index='regime_threshold',
        columns='rsi_lower',
        aggfunc='mean'
    )
    
    im = axes[2].imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    axes[2].set_xticks(range(len(pivot.columns)))
    axes[2].set_yticks(range(len(pivot.index)))
    axes[2].set_xticklabels(pivot.columns)
    axes[2].set_yticklabels(pivot.index)
    axes[2].set_xlabel('RSI Lower Threshold')
    axes[2].set_ylabel('Regime Threshold')
    axes[2].set_title('Sharpe Ratio Heatmap (Regime vs RSI)', fontsize=12)
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('hmm_optimization_results.png', dpi=150)
    print(f"\n📈 차트 저장: hmm_optimization_results.png")
    
    # 결과 저장
    results_df_sorted.to_csv('hmm_optimization_full_results.csv', index=False)
    print(f"📄 전체 결과 저장: hmm_optimization_full_results.csv")
    
    with open('hmm_optimization_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("HMM 시그널 임계값 최적화 결과\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"최적 파라미터:\n")
        f.write(f"  Regime Threshold: {best['regime_threshold']:.1f}\n")
        f.write(f"  RSI Lower: {best['rsi_lower']:.0f}\n")
        f.write(f"  RSI Upper: {best['rsi_upper']:.0f}\n")
        f.write(f"  ADX Threshold: {best['adx_threshold']:.0f}\n")
        f.write(f"  Danger Signal: {best['danger_ratio']:.1f}%\n\n")
        f.write(f"성과 (SPY 기준):\n")
        f.write(f"  Final Value: ${best['Final Value']:,.0f}\n")
        f.write(f"  CAGR: {best['CAGR']:.2f}%\n")
        f.write(f"  MDD: {best['MDD']:.2f}%\n")
        f.write(f"  Sharpe: {best['Sharpe']:.2f}\n")
    
    print(f"📄 요약 저장: hmm_optimization_summary.txt\n")

if __name__ == "__main__":
    main()
