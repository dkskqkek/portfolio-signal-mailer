# -*- coding: utf-8 -*-
"""
종합 전략 비교 분석
- 베이스: SCHD vs SPY
- 시그널: 최적화 기본시그널 vs 최적화 HMM시그널
- 배당 재투자 포함
- 거래세 포함
- 초기 자본: $100,000
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# crash_detection_system 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'crash_detection_system' / 'src'))

from main import CrashDetectionPipeline

# Configuration
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100000

# 거래 비용
US_TRADING_FEE = 0.0001  # 0.01%
KR_TRADING_FEE = 0.003   # 0.3%
DIVIDEND_TAX_RATE = 0.154  # 15.4%

TICKERS = {
    'SPY': 'SPY',
    'SCHD': 'SCHD',
    'QQQ': 'QQQ',
    'JEPI': 'JEPI',
    'GOLD': 'GLD',
    'KOSPI': '^KS200'
}

def fetch_data_with_dividends():
    """가격 및 배당 데이터 수집"""
    print("📊 데이터 수집 중 (배당 포함)...")
    price_data = {}
    dividend_data = {}
    
    for key, ticker in TICKERS.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=START_DATE, end=END_DATE)
            if hist.empty:
                continue
            
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            
            price_data[key] = hist['Close']
            
            if 'Dividends' in hist.columns:
                dividend_data[key] = hist['Dividends']
            else:
                dividend_data[key] = pd.Series(0, index=hist.index)
            
            print(f"  ✓ {ticker}: {len(hist)} rows")
            
        except Exception as e:
            print(f"  ✗ {ticker} 로드 실패: {e}")
    
    df_prices = pd.DataFrame(price_data).fillna(method='ffill').dropna()
    
    for key in price_data.keys():
        if key not in dividend_data:
            dividend_data[key] = pd.Series(0, index=df_prices.index)
        else:
            dividend_data[key] = dividend_data[key].reindex(df_prices.index).fillna(0)
    
    df_dividends = pd.DataFrame(dividend_data)
    
    print(f"\n정렬된 데이터: {len(df_prices)} rows ({df_prices.index[0].date()} ~ {df_prices.index[-1].date()})")
    return df_prices, df_dividends

def get_basic_signal_optimized(df):
    """최적화된 기본 시그널 (15/30/25/65)"""
    print("\n🔍 최적화된 기본 시그널 계산 중...")
    
    spy = df['SPY']
    log_ret = np.log(spy / spy.shift(1))
    
    ma15 = log_ret.rolling(window=15).mean()
    std30 = log_ret.rolling(window=30).std()
    
    min_periods = 252
    signals = []
    history_ma = []
    history_vol = []
    
    for i in range(len(df)):
        if i < min_periods:
            signals.append(0)
            if not np.isnan(ma15.iloc[i]):
                history_ma.append(ma15.iloc[i])
            if not np.isnan(std30.iloc[i]):
                history_vol.append(std30.iloc[i])
            continue
        
        current_ma = ma15.iloc[i]
        current_vol = std30.iloc[i]
        
        if len(history_ma) > 0:
            p25_ma = np.nanpercentile(history_ma, 25)
            p65_vol = np.nanpercentile(history_vol, 65)
            
            is_danger = (current_ma < p25_ma) or (current_vol > p65_vol)
            signals.append(1 if is_danger else 0)
        else:
            signals.append(0)
        
        history_ma.append(current_ma)
        history_vol.append(current_vol)
    
    print(f"  ✓ 기본 시그널 생성 완료 (Danger: {sum(signals)/len(signals)*100:.1f}%)")
    return pd.Series(signals, index=df.index, name='is_danger')

def get_hmm_signal_optimized(df):
    """최적화된 HMM 시그널"""
    print("\n🧠 최적화된 HMM 시그널 계산 중...")
    
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
        
        indicators = pipeline.indicators.copy()
        indicators.index = pd.to_datetime(indicators.index).tz_localize(None)
        
        df_temp = df.join(indicators[['HMM_Regime', 'RSI', 'ADX']], how='left')
        
        # VIX 추가
        try:
            vix_data = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
            if isinstance(vix_data.columns, pd.MultiIndex):
                df_temp['VIX'] = vix_data['Close']['^VIX']
            else:
                df_temp['VIX'] = vix_data['Close']
        except:
            df_temp['VIX'] = 15
        
        df_temp['HMM_Regime'] = df_temp['HMM_Regime'].fillna(method='ffill').fillna(0)
        df_temp['RSI'] = df_temp['RSI'].fillna(50)
        df_temp['ADX'] = df_temp['ADX'].fillna(20)
        df_temp['VIX'] = df_temp['VIX'].fillna(15)
        
        # 최적 파라미터
        regime_threshold = 1.0
        rsi_crisis = 45
        rsi_normal = 40
        adx_min = 15
        vix_high = 25
        
        signals = []
        
        for i in range(len(df_temp)):
            regime = df_temp['HMM_Regime'].iloc[i]
            rsi = df_temp['RSI'].iloc[i]
            adx = df_temp['ADX'].iloc[i]
            vix = df_temp['VIX'].iloc[i]
            
            is_danger = False
            
            if adx < adx_min:
                is_danger = False
            elif regime >= 2:
                is_danger = True if rsi < rsi_crisis else True
            elif regime >= regime_threshold:
                if rsi < rsi_normal or vix > vix_high:
                    is_danger = True
            
            signals.append(1 if is_danger else 0)
        
        print(f"  ✓ HMM 시그널 생성 완료 (Danger: {sum(signals)/len(signals)*100:.1f}%)")
        return pd.Series(signals, index=df.index, name='is_danger')
        
    except Exception as e:
        print(f"  ✗ HMM 엔진 오류: {e}")
        return None

def backtest_strategy(df_prices, df_dividends, is_danger, core_ticker):
    """전략 백테스트"""
    weights = {
        'CORE': 0.38,
        'DYNAMIC': 0.38,
        'GOLD': 0.05,
        'KOSPI': 0.19
    }
    
    cash = 0
    shares = {
        'CORE': 0,
        'QQQ': 0,
        'JEPI': 0,
        'GOLD': 0,
        'KOSPI': 0
    }
    
    total_fees = 0
    total_dividends = 0
    
    # 초기 배분
    first_prices = df_prices.iloc[0]
    
    core_value = INITIAL_CAPITAL * weights['CORE']
    fee = core_value * US_TRADING_FEE
    shares['CORE'] = (core_value - fee) / first_prices[core_ticker]
    total_fees += fee
    
    qqq_value = INITIAL_CAPITAL * weights['DYNAMIC']
    fee = qqq_value * US_TRADING_FEE
    shares['QQQ'] = (qqq_value - fee) / first_prices['QQQ']
    total_fees += fee
    
    gold_value = INITIAL_CAPITAL * weights['GOLD']
    fee = gold_value * US_TRADING_FEE
    shares['GOLD'] = (gold_value - fee) / first_prices['GOLD']
    total_fees += fee
    
    kospi_value = INITIAL_CAPITAL * weights['KOSPI']
    fee = kospi_value * KR_TRADING_FEE
    shares['KOSPI'] = (kospi_value - fee) / first_prices['KOSPI']
    total_fees += fee
    
    current_mode = 0
    portfolio_values = []
    
    for i in range(len(df_prices)):
        prices = df_prices.iloc[i]
        dividends = df_dividends.iloc[i]
        signal = is_danger.iloc[i]
        
        # 배당 재투자
        for ticker in ['CORE', 'QQQ', 'JEPI', 'GOLD']:
            if shares[ticker] > 0:
                ticker_key = core_ticker if ticker == 'CORE' else ticker
                div_amount = dividends[ticker_key] * shares[ticker]
                
                if div_amount > 0:
                    tax = div_amount * DIVIDEND_TAX_RATE
                    net_dividend = div_amount - tax
                    total_dividends += div_amount
                    
                    if prices[ticker_key] > 0:
                        fee = net_dividend * US_TRADING_FEE
                        additional_shares = (net_dividend - fee) / prices[ticker_key]
                        shares[ticker] += additional_shares
                        total_fees += fee
        
        # KOSPI 배당
        if shares['KOSPI'] > 0:
            div_amount = dividends['KOSPI'] * shares['KOSPI']
            if div_amount > 0:
                tax = div_amount * DIVIDEND_TAX_RATE
                net_dividend = div_amount - tax
                total_dividends += div_amount
                
                if prices['KOSPI'] > 0:
                    fee = net_dividend * KR_TRADING_FEE
                    additional_shares = (net_dividend - fee) / prices['KOSPI']
                    shares['KOSPI'] += additional_shares
                    total_fees += fee
        
        # 포트폴리오 가치
        core_value = shares['CORE'] * prices[core_ticker]
        dynamic_value = shares['QQQ'] * prices['QQQ'] + shares['JEPI'] * prices['JEPI']
        gold_value = shares['GOLD'] * prices['GOLD']
        kospi_value = shares['KOSPI'] * prices['KOSPI']
        
        total_value = core_value + dynamic_value + gold_value + kospi_value + cash
        portfolio_values.append(total_value)
        
        # 리밸런싱
        if signal != current_mode:
            if signal == 1:  # QQQ -> JEPI
                if shares['QQQ'] > 0:
                    sell_value = shares['QQQ'] * prices['QQQ']
                    sell_fee = sell_value * US_TRADING_FEE
                    net_proceeds = sell_value - sell_fee
                    total_fees += sell_fee
                    
                    buy_fee = net_proceeds * US_TRADING_FEE
                    shares['JEPI'] = (net_proceeds - buy_fee) / prices['JEPI']
                    shares['QQQ'] = 0
                    total_fees += buy_fee
                    
            else:  # JEPI -> QQQ
                if shares['JEPI'] > 0:
                    sell_value = shares['JEPI'] * prices['JEPI']
                    sell_fee = sell_value * US_TRADING_FEE
                    net_proceeds = sell_value - sell_fee
                    total_fees += sell_fee
                    
                    buy_fee = net_proceeds * US_TRADING_FEE
                    shares['QQQ'] = (net_proceeds - buy_fee) / prices['QQQ']
                    shares['JEPI'] = 0
                    total_fees += buy_fee
            
            current_mode = signal
    
    return pd.Series(portfolio_values, index=df_prices.index), total_fees, total_dividends

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

def main():
    print("=" * 80)
    print("🎯 종합 전략 비교 분석")
    print("=" * 80)
    print(f"초기 자본: ${INITIAL_CAPITAL:,}")
    print(f"기간: {START_DATE} ~ {END_DATE}")
    print("\n4가지 조합:")
    print("  1. SCHD + 최적화 기본시그널 (15/30/25/65)")
    print("  2. SCHD + 최적화 HMM시그널")
    print("  3. SPY + 최적화 기본시그널 (15/30/25/65)")
    print("  4. SPY + 최적화 HMM시그널")
    
    # 데이터 수집
    df_prices, df_dividends = fetch_data_with_dividends()
    
    # 시그널 생성
    basic_signal = get_basic_signal_optimized(df_prices)
    hmm_signal = get_hmm_signal_optimized(df_prices)
    
    if hmm_signal is None:
        print("\n❌ HMM 시그널 생성 실패")
        return
    
    # 4가지 백테스트 실행
    print("\n" + "=" * 80)
    print("백테스트 실행 중...")
    print("=" * 80)
    
    results = {}
    
    # 1. SCHD + Basic
    print("\n[1/4] SCHD + 기본시그널")
    port1, fees1, div1 = backtest_strategy(df_prices, df_dividends, basic_signal, 'SCHD')
    stats1 = analyze_performance(port1)
    results['SCHD_Basic'] = {'portfolio': port1, 'stats': stats1, 'fees': fees1, 'dividends': div1}
    
    # 2. SCHD + HMM
    print("\n[2/4] SCHD + HMM시그널")
    port2, fees2, div2 = backtest_strategy(df_prices, df_dividends, hmm_signal, 'SCHD')
    stats2 = analyze_performance(port2)
    results['SCHD_HMM'] = {'portfolio': port2, 'stats': stats2, 'fees': fees2, 'dividends': div2}
    
    # 3. SPY + Basic
    print("\n[3/4] SPY + 기본시그널")
    port3, fees3, div3 = backtest_strategy(df_prices, df_dividends, basic_signal, 'SPY')
    stats3 = analyze_performance(port3)
    results['SPY_Basic'] = {'portfolio': port3, 'stats': stats3, 'fees': fees3, 'dividends': div3}
    
    # 4. SPY + HMM
    print("\n[4/4] SPY + HMM시그널")
    port4, fees4, div4 = backtest_strategy(df_prices, df_dividends, hmm_signal, 'SPY')
    stats4 = analyze_performance(port4)
    results['SPY_HMM'] = {'portfolio': port4, 'stats': stats4, 'fees': fees4, 'dividends': div4}
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("📊 종합 비교 결과")
    print("=" * 80)
    
    print(f"\n{'Strategy':<20} {'Final Value':<15} {'CAGR':<10} {'MDD':<10} {'Sharpe':<10} {'Fees':<12} {'Dividends':<12}")
    print("-" * 80)
    
    for name, data in results.items():
        stats = data['stats']
        print(f"{name:<20} ${stats['Final Value']:>13,.0f} {stats['CAGR']:>8.2f}% {stats['MDD']:>8.2f}% {stats['Sharpe']:>9.2f} ${data['fees']:>10,.0f} ${data['dividends']:>10,.0f}")
    
    # 최고 성과 찾기
    best_strategy = max(results.items(), key=lambda x: x[1]['stats']['Final Value'])
    
    print("\n" + "=" * 80)
    print(f"🏆 최고 성과: {best_strategy[0]}")
    print(f"   최종 자산: ${best_strategy[1]['stats']['Final Value']:,.0f}")
    print(f"   CAGR: {best_strategy[1]['stats']['CAGR']:.2f}%")
    print("=" * 80)
    
    # 차트 생성
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. SCHD + Basic
    axes[0, 0].plot(port1.index, port1, linewidth=2, color='blue')
    axes[0, 0].set_title(f'SCHD + 기본시그널\nCAGR: {stats1["CAGR"]:.2f}% | Sharpe: {stats1["Sharpe"]:.2f}', fontweight='bold')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    
    # 2. SCHD + HMM
    axes[0, 1].plot(port2.index, port2, linewidth=2, color='green')
    axes[0, 1].set_title(f'SCHD + HMM시그널\nCAGR: {stats2["CAGR"]:.2f}% | Sharpe: {stats2["Sharpe"]:.2f}', fontweight='bold')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    
    # 3. SPY + Basic
    axes[1, 0].plot(port3.index, port3, linewidth=2, color='orange')
    axes[1, 0].set_title(f'SPY + 기본시그널\nCAGR: {stats3["CAGR"]:.2f}% | Sharpe: {stats3["Sharpe"]:.2f}', fontweight='bold')
    axes[1, 0].set_ylabel('Portfolio Value ($)')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    
    # 4. SPY + HMM
    axes[1, 1].plot(port4.index, port4, linewidth=2, color='red')
    axes[1, 1].set_title(f'SPY + HMM시그널\nCAGR: {stats4["CAGR"]:.2f}% | Sharpe: {stats4["Sharpe"]:.2f}', fontweight='bold')
    axes[1, 1].set_ylabel('Portfolio Value ($)')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('comprehensive_strategy_comparison.png', dpi=150)
    print(f"\n📈 차트 저장: comprehensive_strategy_comparison.png")
    
    # 리포트 저장
    with open('comprehensive_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("종합 전략 비교 분석\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"초기 자본: ${INITIAL_CAPITAL:,}\n")
        f.write(f"기간: {df_prices.index[0].date()} ~ {df_prices.index[-1].date()}\n\n")
        f.write(f"{'Strategy':<20} {'Final Value':<15} {'CAGR':<10} {'MDD':<10} {'Sharpe':<10}\n")
        f.write("-" * 80 + "\n")
        
        for name, data in results.items():
            stats = data['stats']
            f.write(f"{name:<20} ${stats['Final Value']:>13,.0f} {stats['CAGR']:>8.2f}% {stats['MDD']:>8.2f}% {stats['Sharpe']:>9.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"최고 성과: {best_strategy[0]}\n")
        f.write(f"최종 자산: ${best_strategy[1]['stats']['Final Value']:,.0f}\n")
        f.write("=" * 80 + "\n")
    
    print(f"📄 리포트 저장: comprehensive_comparison_report.txt\n")

if __name__ == "__main__":
    main()
