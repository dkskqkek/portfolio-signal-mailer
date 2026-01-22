# -*- coding: utf-8 -*-
"""
최적화된 HMM 전략 역사적 신호 분석
- 신호 횟수 카운트
- 매수/매도 시점 차트 표시
- 거래세 상세 분석
- 초기 자본: 1,000만원 (KRW)
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
INITIAL_CAPITAL_KRW = 10_000_000  # 1,000만원
USD_KRW_RATE = 1450  # 환율 (1 USD = 1450 KRW)
INITIAL_CAPITAL_USD = INITIAL_CAPITAL_KRW / USD_KRW_RATE

# 거래 비용
US_TRADING_FEE = 0.0001  # 0.01% (미국 주식)
KR_TRADING_FEE = 0.003   # 0.3% (한국 주식)

TICKERS = {
    'SPY': 'SPY',
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
    
    df_aligned = pd.DataFrame(data).fillna(method='ffill').dropna()
    print(f"\n정렬된 데이터: {len(df_aligned)} rows ({df_aligned.index[0].date()} ~ {df_aligned.index[-1].date()})")
    return df_aligned

def get_hmm_signals(df):
    """최적화된 HMM 전략 시그널 생성"""
    print("\n🧠 최적화된 HMM 전략 실행 중...")
    
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
            df['VIX'] = 15
        
        # 결측치 처리
        df['HMM_Regime'] = df['HMM_Regime'].fillna(method='ffill').fillna(0)
        df['RSI'] = df['RSI'].fillna(50)
        df['ADX'] = df['ADX'].fillna(20)
        df['VIX'] = df['VIX'].fillna(15)
        
        # 최적화된 파라미터로 시그널 생성
        regime_threshold = 1.0
        rsi_crisis = 45
        rsi_normal = 40
        adx_min = 15
        vix_high = 25
        
        signals = []
        
        for i in range(len(df)):
            regime = df['HMM_Regime'].iloc[i]
            rsi = df['RSI'].iloc[i]
            adx = df['ADX'].iloc[i]
            vix = df['VIX'].iloc[i]
            
            is_danger = False
            
            # ADX 필터
            if adx < adx_min:
                is_danger = False
            # Crisis 레짐
            elif regime >= 2:
                if rsi < rsi_crisis:
                    is_danger = True
                else:
                    is_danger = True
            # Correction 레짐
            elif regime >= regime_threshold:
                if rsi < rsi_normal or vix > vix_high:
                    is_danger = True
            
            signals.append(1 if is_danger else 0)
        
        df['is_danger'] = signals
        
        print(f"  ✓ HMM 시그널 생성 완료")
        print(f"  - 위험 신호: {sum(signals)}일 ({sum(signals)/len(signals)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"  ✗ HMM 엔진 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def backtest_with_signal_tracking(df):
    """
    시그널 추적 백테스트
    - 매수/매도 시점 기록
    - 거래세 상세 추적
    """
    print(f"\n💼 백테스트 실행 중 (초기 자본: ₩{INITIAL_CAPITAL_KRW:,})...")
    
    weights = {
        'SPY': 0.38,
        'DYNAMIC': 0.38,
        'GOLD': 0.05,
        'KOSPI': 0.19
    }
    
    # 초기 자본 (USD)
    cash_usd = INITIAL_CAPITAL_USD
    shares = {
        'SPY': 0,
        'QQQ': 0,
        'JEPI': 0,
        'GOLD': 0,
        'KOSPI': 0
    }
    
    # 추적 변수
    portfolio_values_krw = []
    total_trading_fees = 0
    trade_history = []  # 거래 내역
    signal_changes = []  # 신호 변경 시점
    
    # 초기 배분
    first_prices = df.iloc[0]
    
    # SPY 매수
    spy_value = cash_usd * weights['SPY']
    fee = spy_value * US_TRADING_FEE
    shares['SPY'] = (spy_value - fee) / first_prices['SPY']
    total_trading_fees += fee
    trade_history.append({
        'date': df.index[0],
        'action': 'BUY',
        'ticker': 'SPY',
        'shares': shares['SPY'],
        'price': first_prices['SPY'],
        'value': spy_value - fee,
        'fee': fee
    })
    
    # QQQ 매수
    qqq_value = cash_usd * weights['DYNAMIC']
    fee = qqq_value * US_TRADING_FEE
    shares['QQQ'] = (qqq_value - fee) / first_prices['QQQ']
    total_trading_fees += fee
    trade_history.append({
        'date': df.index[0],
        'action': 'BUY',
        'ticker': 'QQQ',
        'shares': shares['QQQ'],
        'price': first_prices['QQQ'],
        'value': qqq_value - fee,
        'fee': fee
    })
    
    # GOLD 매수
    gold_value = cash_usd * weights['GOLD']
    fee = gold_value * US_TRADING_FEE
    shares['GOLD'] = (gold_value - fee) / first_prices['GOLD']
    total_trading_fees += fee
    trade_history.append({
        'date': df.index[0],
        'action': 'BUY',
        'ticker': 'GOLD',
        'shares': shares['GOLD'],
        'price': first_prices['GOLD'],
        'value': gold_value - fee,
        'fee': fee
    })
    
    # KOSPI 매수
    kospi_value = cash_usd * weights['KOSPI']
    fee = kospi_value * KR_TRADING_FEE
    shares['KOSPI'] = (kospi_value - fee) / first_prices['KOSPI']
    total_trading_fees += fee
    trade_history.append({
        'date': df.index[0],
        'action': 'BUY',
        'ticker': 'KOSPI',
        'shares': shares['KOSPI'],
        'price': first_prices['KOSPI'],
        'value': kospi_value - fee,
        'fee': fee
    })
    
    cash_usd = 0
    current_mode = 0
    
    for i in range(len(df)):
        prices = df.iloc[i]
        signal = df['is_danger'].iloc[i]
        
        # 포트폴리오 가치 (KRW)
        spy_value = shares['SPY'] * prices['SPY']
        dynamic_value = shares['QQQ'] * prices['QQQ'] + shares['JEPI'] * prices['JEPI']
        gold_value = shares['GOLD'] * prices['GOLD']
        kospi_value = shares['KOSPI'] * prices['KOSPI']
        
        total_value_usd = spy_value + dynamic_value + gold_value + kospi_value + cash_usd
        total_value_krw = total_value_usd * USD_KRW_RATE
        portfolio_values_krw.append(total_value_krw)
        
        # 신호 변경 시 리밸런싱
        if signal != current_mode:
            signal_changes.append({
                'date': df.index[i],
                'from': 'QQQ' if current_mode == 0 else 'JEPI',
                'to': 'JEPI' if signal == 1 else 'QQQ',
                'signal': 'DANGER' if signal == 1 else 'NORMAL'
            })
            
            if signal == 1:  # Normal -> Danger: QQQ -> JEPI
                if shares['QQQ'] > 0:
                    # QQQ 매도
                    sell_value = shares['QQQ'] * prices['QQQ']
                    sell_fee = sell_value * US_TRADING_FEE
                    net_proceeds = sell_value - sell_fee
                    total_trading_fees += sell_fee
                    
                    trade_history.append({
                        'date': df.index[i],
                        'action': 'SELL',
                        'ticker': 'QQQ',
                        'shares': shares['QQQ'],
                        'price': prices['QQQ'],
                        'value': sell_value,
                        'fee': sell_fee
                    })
                    
                    # JEPI 매수
                    buy_fee = net_proceeds * US_TRADING_FEE
                    shares['JEPI'] = (net_proceeds - buy_fee) / prices['JEPI']
                    shares['QQQ'] = 0
                    total_trading_fees += buy_fee
                    
                    trade_history.append({
                        'date': df.index[i],
                        'action': 'BUY',
                        'ticker': 'JEPI',
                        'shares': shares['JEPI'],
                        'price': prices['JEPI'],
                        'value': net_proceeds - buy_fee,
                        'fee': buy_fee
                    })
                    
            else:  # Danger -> Normal: JEPI -> QQQ
                if shares['JEPI'] > 0:
                    # JEPI 매도
                    sell_value = shares['JEPI'] * prices['JEPI']
                    sell_fee = sell_value * US_TRADING_FEE
                    net_proceeds = sell_value - sell_fee
                    total_trading_fees += sell_fee
                    
                    trade_history.append({
                        'date': df.index[i],
                        'action': 'SELL',
                        'ticker': 'JEPI',
                        'shares': shares['JEPI'],
                        'price': prices['JEPI'],
                        'value': sell_value,
                        'fee': sell_fee
                    })
                    
                    # QQQ 매수
                    buy_fee = net_proceeds * US_TRADING_FEE
                    shares['QQQ'] = (net_proceeds - buy_fee) / prices['QQQ']
                    shares['JEPI'] = 0
                    total_trading_fees += buy_fee
                    
                    trade_history.append({
                        'date': df.index[i],
                        'action': 'BUY',
                        'ticker': 'QQQ',
                        'shares': shares['QQQ'],
                        'price': prices['QQQ'],
                        'value': net_proceeds - buy_fee,
                        'fee': buy_fee
                    })
            
            current_mode = signal
    
    print(f"  ✓ 백테스트 완료")
    print(f"  - 총 거래 횟수: {len(trade_history)}회")
    print(f"  - 신호 변경: {len(signal_changes)}회")
    print(f"  - 총 거래세: ${total_trading_fees:,.2f} (₩{total_trading_fees * USD_KRW_RATE:,.0f})")
    
    return pd.Series(portfolio_values_krw, index=df.index), trade_history, signal_changes, total_trading_fees

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
    
    return {
        'Final Value': series.iloc[-1],
        'Total Return': total_ret * 100,
        'CAGR': cagr * 100,
        'MDD': mdd * 100,
        'Sharpe': sharpe
    }

def main():
    print("=" * 70)
    print("🎯 최적화된 HMM 전략 역사적 신호 분석")
    print("=" * 70)
    print(f"초기 자본: ₩{INITIAL_CAPITAL_KRW:,} (${INITIAL_CAPITAL_USD:,.2f})")
    print(f"환율: 1 USD = ₩{USD_KRW_RATE}")
    
    # 데이터 수집
    df = fetch_data()
    
    # HMM 시그널 생성
    df = get_hmm_signals(df)
    
    if df is None or 'is_danger' not in df.columns:
        print("\n❌ HMM 시그널 생성 실패")
        return
    
    # 백테스트 실행
    portfolio, trade_history, signal_changes, total_fees = backtest_with_signal_tracking(df)
    
    # 성과 분석
    stats = analyze_performance(portfolio)
    
    # 신호 통계
    buy_signals = [s for s in signal_changes if s['to'] == 'QQQ']
    sell_signals = [s for s in signal_changes if s['to'] == 'JEPI']
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("📊 백테스트 결과")
    print("=" * 70)
    print(f"최종 자산: ₩{stats['Final Value']:,.0f} (${stats['Final Value']/USD_KRW_RATE:,.2f})")
    print(f"총 수익률: {stats['Total Return']:.2f}%")
    print(f"CAGR: {stats['CAGR']:.2f}%")
    print(f"MDD: {stats['MDD']:.2f}%")
    print(f"Sharpe: {stats['Sharpe']:.2f}")
    
    print("\n" + "=" * 70)
    print("📈 신호 통계")
    print("=" * 70)
    print(f"총 신호 변경: {len(signal_changes)}회")
    print(f"  - 매수 신호 (QQQ): {len(buy_signals)}회")
    print(f"  - 매도 신호 (JEPI): {len(sell_signals)}회")
    print(f"총 거래 횟수: {len(trade_history)}회")
    
    print("\n" + "=" * 70)
    print("💰 거래세 분석")
    print("=" * 70)
    print(f"총 거래세: ${total_fees:,.2f} (₩{total_fees * USD_KRW_RATE:,.0f})")
    print(f"초기 자본 대비: {total_fees / INITIAL_CAPITAL_USD * 100:.2f}%")
    print(f"최종 자산 대비: {total_fees / (stats['Final Value'] / USD_KRW_RATE) * 100:.2f}%")
    
    # 거래 내역 상세
    buy_trades = [t for t in trade_history if t['action'] == 'BUY']
    sell_trades = [t for t in trade_history if t['action'] == 'SELL']
    
    total_buy_fees = sum(t['fee'] for t in buy_trades)
    total_sell_fees = sum(t['fee'] for t in sell_trades)
    
    print(f"\n거래세 세부:")
    print(f"  - 매수 거래세: ${total_buy_fees:,.2f} (₩{total_buy_fees * USD_KRW_RATE:,.0f})")
    print(f"  - 매도 거래세: ${total_sell_fees:,.2f} (₩{total_sell_fees * USD_KRW_RATE:,.0f})")
    
    # 차트 생성
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 1. 포트폴리오 가치 + 매수/매도 시점
    axes[0].plot(portfolio.index, portfolio, label='Portfolio Value (KRW)', linewidth=2, color='black')
    
    # 매수 시점 (초록색 ▲)
    buy_dates = [s['date'] for s in signal_changes if s['to'] == 'QQQ']
    buy_values = [portfolio.loc[d] for d in buy_dates]
    axes[0].scatter(buy_dates, buy_values, color='green', marker='^', s=150, label='BUY (QQQ)', zorder=5)
    
    # 매도 시점 (빨간색 ▼)
    sell_dates = [s['date'] for s in signal_changes if s['to'] == 'JEPI']
    sell_values = [portfolio.loc[d] for d in sell_dates]
    axes[0].scatter(sell_dates, sell_values, color='red', marker='v', s=150, label='SELL (JEPI)', zorder=5)
    
    axes[0].set_title(f'HMM 전략 백테스트 (초기 자본: ₩{INITIAL_CAPITAL_KRW:,})', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value (KRW)')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=INITIAL_CAPITAL_KRW, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # 2. 시그널
    axes[1].fill_between(df.index, 0, 1, where=df['is_danger']==1, alpha=0.3, color='red', label='Danger (JEPI)')
    axes[1].fill_between(df.index, 0, 1, where=df['is_danger']==0, alpha=0.3, color='green', label='Normal (QQQ)')
    axes[1].set_title(f'HMM 시그널 (총 {len(signal_changes)}회 변경)', fontsize=12)
    axes[1].set_ylabel('Signal')
    axes[1].set_xlabel('Date')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hmm_signal_history_analysis.png', dpi=150)
    print(f"\n📈 차트 저장: hmm_signal_history_analysis.png")
    
    # 리포트 저장
    with open('hmm_signal_history_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("최적화된 HMM 전략 역사적 신호 분석\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"초기 자본: ₩{INITIAL_CAPITAL_KRW:,}\n")
        f.write(f"기간: {df.index[0].date()} ~ {df.index[-1].date()}\n\n")
        f.write(f"최종 자산: ₩{stats['Final Value']:,.0f}\n")
        f.write(f"CAGR: {stats['CAGR']:.2f}%\n")
        f.write(f"MDD: {stats['MDD']:.2f}%\n")
        f.write(f"Sharpe: {stats['Sharpe']:.2f}\n\n")
        f.write("신호 통계:\n")
        f.write(f"  총 신호 변경: {len(signal_changes)}회\n")
        f.write(f"  매수 신호: {len(buy_signals)}회\n")
        f.write(f"  매도 신호: {len(sell_signals)}회\n\n")
        f.write("거래세:\n")
        f.write(f"  총 거래세: ₩{total_fees * USD_KRW_RATE:,.0f}\n")
        f.write(f"  매수 거래세: ₩{total_buy_fees * USD_KRW_RATE:,.0f}\n")
        f.write(f"  매도 거래세: ₩{total_sell_fees * USD_KRW_RATE:,.0f}\n")
    
    print(f"📄 리포트 저장: hmm_signal_history_report.txt\n")

if __name__ == "__main__":
    main()
