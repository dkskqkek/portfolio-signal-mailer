# -*- coding: utf-8 -*-
"""
HMM 전략 파라미터 최적화
- RSI, ADX, VIX 임계값 조합 최적화
- HMM 레짐 기반 신호 생성 규칙 최적화
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
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

def get_hmm_base_data(df):
    """HMM 기본 데이터 가져오기"""
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
        
        # 지표 데이터 추출
        indicators = pipeline.indicators.copy()
        indicators.index = pd.to_datetime(indicators.index).tz_localize(None)
        
        # 병합
        df = df.join(indicators[['HMM_Regime', 'RSI', 'ADX']], how='left')
        
        # VIX 데이터 추가
        try:
            vix_data = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
            if isinstance(vix_data.columns, pd.MultiIndex):
                df['VIX'] = vix_data['Close']['^VIX']
            else:
                df['VIX'] = vix_data['Close']
        except:
            df['VIX'] = 15  # 기본값
        
        # 결측치 처리
        df['HMM_Regime'] = df['HMM_Regime'].fillna(method='ffill').fillna(0)
        df['RSI'] = df['RSI'].fillna(50)
        df['ADX'] = df['ADX'].fillna(20)
        df['VIX'] = df['VIX'].fillna(15)
        
        print(f"  ✓ HMM 데이터 추출 완료")
        print(f"  - Regime 분포: {df['HMM_Regime'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"  ✗ HMM 엔진 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_hmm_signal(df, regime_threshold=1.5, rsi_crisis=45, rsi_normal=40, adx_min=20, vix_high=30):
    """
    HMM 기반 시그널 생성 (최적화 가능한 파라미터)
    
    Args:
        regime_threshold: HMM 레짐 임계값 (이상이면 위험)
        rsi_crisis: Crisis 상태에서 RSI 임계값
        rsi_normal: Normal 상태에서 RSI 임계값  
        adx_min: ADX 최소값 (이하면 신호 무시)
        vix_high: VIX 고점 임계값
    """
    df_copy = df.copy()
    
    # 기본 신호 (0: Normal, 1: Danger)
    signals = []
    
    for i in range(len(df_copy)):
        regime = df_copy['HMM_Regime'].iloc[i]
        rsi = df_copy['RSI'].iloc[i]
        adx = df_copy['ADX'].iloc[i]
        vix = df_copy['VIX'].iloc[i]
        
        is_danger = False
        
        # ADX 필터: 추세가 약하면 신호 무시
        if adx < adx_min:
            signals.append(0)
            continue
        
        # Crisis 레짐 (Regime == 2)
        if regime >= 2:
            if rsi < rsi_crisis:
                is_danger = True
        # Correction 레짐 (Regime == 1)
        elif regime >= regime_threshold:
            if rsi < rsi_normal or vix > vix_high:
                is_danger = True
        
        signals.append(1 if is_danger else 0)
    
    return pd.Series(signals, index=df_copy.index, name='is_danger')

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

def optimize_hmm_parameters(df):
    """HMM 파라미터 최적화"""
    print("\n🔍 HMM 파라미터 최적화 시작...")
    
    # 파라미터 그리드
    regime_thresholds = [1.0, 1.5, 2.0]  # HMM 레짐 임계값
    rsi_crisis_vals = [40, 45, 50]  # Crisis RSI
    rsi_normal_vals = [30, 35, 40]  # Normal RSI
    adx_mins = [15, 20, 25]  # ADX 최소값
    vix_highs = [25, 30, 35]  # VIX 고점
    
    results = []
    total_combinations = len(list(product(regime_thresholds, rsi_crisis_vals, rsi_normal_vals, adx_mins, vix_highs)))
    
    print(f"  총 {total_combinations}개 조합 테스트 중...")
    
    for i, (regime_th, rsi_c, rsi_n, adx_m, vix_h) in enumerate(product(regime_thresholds, rsi_crisis_vals, rsi_normal_vals, adx_mins, vix_highs)):
        if i % 20 == 0:
            print(f"  진행: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
        
        # 시그널 생성
        is_danger = create_hmm_signal(df, regime_th, rsi_c, rsi_n, adx_m, vix_h)
        
        # 백테스트 (SPY 기준)
        portfolio = backtest_with_signal(df, is_danger, 'SPY')
        stats = analyze_performance(portfolio)
        
        # 위험 신호 비율
        danger_ratio = is_danger.sum() / len(is_danger) * 100
        
        # 거래 횟수
        switches = (is_danger.diff() != 0).sum()
        
        results.append({
            'regime_threshold': regime_th,
            'rsi_crisis': rsi_c,
            'rsi_normal': rsi_n,
            'adx_min': adx_m,
            'vix_high': vix_h,
            'danger_ratio': danger_ratio,
            'switches': switches,
            **stats
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n  ✓ 최적화 완료!")
    return results_df

def main():
    print("=" * 70)
    print("🎯 HMM 전략 파라미터 최적화")
    print("=" * 70)
    
    # 데이터 수집
    df = fetch_data()
    
    # HMM 데이터 가져오기
    df = get_hmm_base_data(df)
    
    if df is None or 'HMM_Regime' not in df.columns:
        print("\n❌ HMM 데이터 로드 실패")
        return
    
    # 최적화 실행
    results_df = optimize_hmm_parameters(df)
    
    # 결과 정렬 (Sharpe 기준)
    results_sharpe = results_df.sort_values('Sharpe', ascending=False)
    results_cagr = results_df.sort_values('CAGR', ascending=False)
    
    # Top 10 출력
    print("\n" + "=" * 70)
    print("🏆 Top 10 최적 파라미터 (Sharpe Ratio 기준)")
    print("=" * 70)
    print(f"{'Rank':<5} {'Regime':<8} {'RSI_C':<7} {'RSI_N':<7} {'ADX':<6} {'VIX':<6} {'Danger%':<9} {'Sharpe':<8} {'CAGR':<8}")
    print("-" * 70)
    
    for idx, (i, row) in enumerate(results_sharpe.head(10).iterrows()):
        print(f"{idx+1:<5} {row['regime_threshold']:<8.1f} {row['rsi_crisis']:<7.0f} {row['rsi_normal']:<7.0f} "
              f"{row['adx_min']:<6.0f} {row['vix_high']:<6.0f} {row['danger_ratio']:<9.1f} "
              f"{row['Sharpe']:<8.2f} {row['CAGR']:<8.2f}")
    
    # 최적 파라미터
    best = results_sharpe.iloc[0]
    
    print("\n" + "=" * 70)
    print("🎯 최적 파라미터 (Sharpe 최대화)")
    print("=" * 70)
    print(f"  Regime Threshold: {best['regime_threshold']:.1f}")
    print(f"  RSI Crisis: {best['rsi_crisis']:.0f}")
    print(f"  RSI Normal: {best['rsi_normal']:.0f}")
    print(f"  ADX Min: {best['adx_min']:.0f}")
    print(f"  VIX High: {best['vix_high']:.0f}")
    print(f"  Danger Signal: {best['danger_ratio']:.1f}%")
    print(f"  Switches: {best['switches']:.0f}회")
    print(f"\n  성과:")
    print(f"    Final Value: ${best['Final Value']:,.0f}")
    print(f"    CAGR: {best['CAGR']:.2f}%")
    print(f"    MDD: {best['MDD']:.2f}%")
    print(f"    Sharpe: {best['Sharpe']:.2f}")
    
    # 최적 파라미터로 SCHD vs SPY 비교
    print("\n" + "=" * 70)
    print("📊 최적 HMM 파라미터 백테스트: SCHD vs SPY")
    print("=" * 70)
    
    optimal_signal = create_hmm_signal(
        df,
        best['regime_threshold'],
        best['rsi_crisis'],
        best['rsi_normal'],
        best['adx_min'],
        best['vix_high']
    )
    
    portfolio_schd = backtest_with_signal(df, optimal_signal, 'SCHD')
    portfolio_spy = backtest_with_signal(df, optimal_signal, 'SPY')
    
    stats_schd = analyze_performance(portfolio_schd)
    stats_spy = analyze_performance(portfolio_spy)
    
    print(f"\n{'Metric':<20} {'SCHD (HMM Opt)':<20} {'SPY (HMM Opt)':<20}")
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
    ax1.plot(portfolio_schd.index, portfolio_schd, label='SCHD (HMM Optimized)', linewidth=2)
    ax1.plot(portfolio_spy.index, portfolio_spy, label='SPY (HMM Optimized)', linewidth=2, alpha=0.8)
    ax1.set_title('최적화된 HMM 전략 백테스트: SCHD vs SPY', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 시그널
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(df.index, 0, 1, where=optimal_signal==1, alpha=0.3, color='red', label='Danger (JEPI)')
    ax2.fill_between(df.index, 0, 1, where=optimal_signal==0, alpha=0.3, color='green', label='Normal (QQQ)')
    ax2.set_title(f'최적 HMM 시그널 (Danger: {best["danger_ratio"]:.1f}%, Switches: {best["switches"]:.0f}회)', fontsize=12)
    ax2.set_ylabel('Signal')
    ax2.set_xlabel('Date')
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Sharpe Heatmap (Regime vs RSI Crisis)
    ax3 = fig.add_subplot(gs[2, 0])
    pivot_sharpe = results_df.pivot_table(
        values='Sharpe',
        index='regime_threshold',
        columns='rsi_crisis',
        aggfunc='mean'
    )
    sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax3, cbar_kws={'label': 'Sharpe'})
    ax3.set_title('Sharpe Heatmap (Regime vs RSI Crisis)', fontsize=12)
    ax3.set_xlabel('RSI Crisis')
    ax3.set_ylabel('Regime Threshold')
    
    # 4. CAGR Heatmap
    ax4 = fig.add_subplot(gs[2, 1])
    pivot_cagr = results_df.pivot_table(
        values='CAGR',
        index='adx_min',
        columns='vix_high',
        aggfunc='mean'
    )
    sns.heatmap(pivot_cagr, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax4, cbar_kws={'label': 'CAGR (%)'})
    ax4.set_title('CAGR Heatmap (ADX vs VIX)', fontsize=12)
    ax4.set_xlabel('VIX High')
    ax4.set_ylabel('ADX Min')
    
    plt.savefig('hmm_strategy_optimization_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 차트 저장: hmm_strategy_optimization_results.png")
    
    # 결과 저장
    results_df.to_csv('hmm_strategy_optimization_full_results.csv', index=False)
    print(f"📄 전체 결과 저장: hmm_strategy_optimization_full_results.csv")
    
    with open('hmm_strategy_optimization_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("HMM 전략 파라미터 최적화 결과\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"최적 파라미터:\n")
        f.write(f"  Regime Threshold: {best['regime_threshold']:.1f}\n")
        f.write(f"  RSI Crisis: {best['rsi_crisis']:.0f}\n")
        f.write(f"  RSI Normal: {best['rsi_normal']:.0f}\n")
        f.write(f"  ADX Min: {best['adx_min']:.0f}\n")
        f.write(f"  VIX High: {best['vix_high']:.0f}\n")
        f.write(f"  Danger Signal: {best['danger_ratio']:.1f}%\n")
        f.write(f"  Switches: {best['switches']:.0f}회\n\n")
        f.write(f"성과 (SPY 기준):\n")
        f.write(f"  Final Value: ${best['Final Value']:,.0f}\n")
        f.write(f"  CAGR: {best['CAGR']:.2f}%\n")
        f.write(f"  MDD: {best['MDD']:.2f}%\n")
        f.write(f"  Sharpe: {best['Sharpe']:.2f}\n")
    
    print(f"📄 요약 저장: hmm_strategy_optimization_summary.txt\n")

if __name__ == "__main__":
    main()
