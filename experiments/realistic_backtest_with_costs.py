# -*- coding: utf-8 -*-
"""
현실적인 백테스트: 배당 + 거래비용 + 세금 포함
- 배당 수익 재투자
- 거래 수수료 반영
- 배당세 및 양도소득세 반영
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

# 거래 비용
US_TRADING_FEE = 0.0001  # 0.01% (미국 주식)
KR_TRADING_FEE = 0.003   # 0.3% (한국 주식)

# 세금
DIVIDEND_TAX_RATE = 0.154  # 15.4% (미국 배당세)
CAPITAL_GAINS_TAX_THRESHOLD = 250 * 14.5  # 250만원 (USD 환산, 1USD=1450원)
CAPITAL_GAINS_TAX_RATE = 0.22  # 22%

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
            # yfinance Ticker 객체
            t = yf.Ticker(ticker)
            
            # 가격 데이터
            hist = t.history(start=START_DATE, end=END_DATE)
            if hist.empty:
                continue
            
            # 타임존 제거
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            
            price_data[key] = hist['Close']
            
            # 배당 데이터
            if 'Dividends' in hist.columns:
                dividend_data[key] = hist['Dividends']
            else:
                dividend_data[key] = pd.Series(0, index=hist.index)
            
            print(f"  ✓ {ticker}: {len(hist)} rows, 배당: {dividend_data[key].sum():.2f}")
            
        except Exception as e:
            print(f"  ✗ {ticker} 로드 실패: {e}")
    
    # 날짜 정렬
    df_prices = pd.DataFrame(price_data).fillna(method='ffill').dropna()
    
    # 배당 데이터 정렬
    for key in price_data.keys():
        if key not in dividend_data:
            dividend_data[key] = pd.Series(0, index=df_prices.index)
        else:
            dividend_data[key] = dividend_data[key].reindex(df_prices.index).fillna(0)
    
    df_dividends = pd.DataFrame(dividend_data)
    
    print(f"\n정렬된 데이터: {len(df_prices)} rows ({df_prices.index[0].date()} ~ {df_prices.index[-1].date()})")
    return df_prices, df_dividends

def calculate_signal_optimized(df):
    """최적화된 기본 시그널 (15/30/25/65)"""
    print("\n🔍 시그널 계산 중 (최적 파라미터)...")
    
    spy = df['SPY']
    log_ret = np.log(spy / spy.shift(1))
    
    ma = log_ret.rolling(window=15).mean()
    vol = log_ret.rolling(window=30).std()
    
    min_periods = 252
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
            p25_ma = np.nanpercentile(history_ma, 25)
            p65_vol = np.nanpercentile(history_vol, 65)
            
            is_danger = (current_ma < p25_ma) or (current_vol > p65_vol)
            signals.append(1 if is_danger else 0)
        else:
            signals.append(0)
        
        history_ma.append(current_ma)
        history_vol.append(current_vol)
    
    df['is_danger'] = signals
    print(f"  위험 신호: {sum(signals)}일 ({sum(signals)/len(signals)*100:.1f}%)")
    return df

def backtest_realistic(df_prices, df_dividends, is_danger, core_ticker='SPY'):
    """
    현실적인 백테스트 (배당 + 거래비용 + 세금)
    """
    print(f"\n💼 현실적인 백테스트 실행 중 ({core_ticker})...")
    
    weights = {
        'CORE': 0.38,
        'DYNAMIC': 0.38,
        'GOLD': 0.05,
        'KOSPI': 0.19
    }
    
    # 초기 자본
    cash = INITIAL_CAPITAL
    shares = {
        'CORE': 0,
        'QQQ': 0,
        'JEPI': 0,
        'GOLD': 0,
        'KOSPI': 0
    }
    
    # 추적 변수
    portfolio_values = []
    total_dividends_received = 0
    total_trading_fees = 0
    total_dividend_tax = 0
    initial_investment = {}  # 양도소득세 계산용
    
    # 초기 배분
    first_prices = df_prices.iloc[0]
    
    # CORE 매수
    core_value = cash * weights['CORE']
    fee = core_value * US_TRADING_FEE
    shares['CORE'] = (core_value - fee) / first_prices[core_ticker]
    total_trading_fees += fee
    initial_investment['CORE'] = core_value - fee
    
    # QQQ 매수
    qqq_value = cash * weights['DYNAMIC']
    fee = qqq_value * US_TRADING_FEE
    shares['QQQ'] = (qqq_value - fee) / first_prices['QQQ']
    total_trading_fees += fee
    initial_investment['QQQ'] = qqq_value - fee
    
    # GOLD 매수
    gold_value = cash * weights['GOLD']
    fee = gold_value * US_TRADING_FEE
    shares['GOLD'] = (gold_value - fee) / first_prices['GOLD']
    total_trading_fees += fee
    initial_investment['GOLD'] = gold_value - fee
    
    # KOSPI 매수
    kospi_value = cash * weights['KOSPI']
    fee = kospi_value * KR_TRADING_FEE
    shares['KOSPI'] = (kospi_value - fee) / first_prices['KOSPI']
    total_trading_fees += fee
    initial_investment['KOSPI'] = kospi_value - fee
    
    cash = 0
    current_mode = 0
    
    for i in range(len(df_prices)):
        prices = df_prices.iloc[i]
        dividends = df_dividends.iloc[i]
        signal = is_danger.iloc[i]
        
        # 1. 배당 수령 (세후)
        for ticker in ['CORE', 'QQQ', 'JEPI', 'GOLD']:
            if shares[ticker] > 0:
                ticker_key = core_ticker if ticker == 'CORE' else ticker
                div_amount = dividends[ticker_key] * shares[ticker]
                
                if div_amount > 0:
                    # 배당세 차감
                    tax = div_amount * DIVIDEND_TAX_RATE
                    net_dividend = div_amount - tax
                    
                    total_dividends_received += div_amount
                    total_dividend_tax += tax
                    
                    # 배당 재투자 (같은 종목에)
                    if prices[ticker_key] > 0:
                        fee = net_dividend * US_TRADING_FEE
                        additional_shares = (net_dividend - fee) / prices[ticker_key]
                        shares[ticker] += additional_shares
                        total_trading_fees += fee
        
        # KOSPI 배당 (한국 주식)
        if shares['KOSPI'] > 0:
            div_amount = dividends['KOSPI'] * shares['KOSPI']
            if div_amount > 0:
                tax = div_amount * DIVIDEND_TAX_RATE
                net_dividend = div_amount - tax
                
                total_dividends_received += div_amount
                total_dividend_tax += tax
                
                # 배당 재투자
                if prices['KOSPI'] > 0:
                    fee = net_dividend * KR_TRADING_FEE
                    additional_shares = (net_dividend - fee) / prices['KOSPI']
                    shares['KOSPI'] += additional_shares
                    total_trading_fees += fee
        
        # 2. 포트폴리오 가치 계산
        core_value = shares['CORE'] * prices[core_ticker]
        dynamic_value = shares['QQQ'] * prices['QQQ'] + shares['JEPI'] * prices['JEPI']
        gold_value = shares['GOLD'] * prices['GOLD']
        kospi_value = shares['KOSPI'] * prices['KOSPI']
        
        total_value = core_value + dynamic_value + gold_value + kospi_value + cash
        portfolio_values.append(total_value)
        
        # 3. 신호 변경 시 리밸런싱
        if signal != current_mode:
            if signal == 1:  # Normal -> Danger: QQQ -> JEPI
                if shares['QQQ'] > 0:
                    # QQQ 매도
                    sell_value = shares['QQQ'] * prices['QQQ']
                    sell_fee = sell_value * US_TRADING_FEE
                    
                    # 양도소득세 계산
                    capital_gain = sell_value - initial_investment.get('QQQ', sell_value)
                    if capital_gain > CAPITAL_GAINS_TAX_THRESHOLD:
                        capital_gains_tax = (capital_gain - CAPITAL_GAINS_TAX_THRESHOLD) * CAPITAL_GAINS_TAX_RATE
                    else:
                        capital_gains_tax = 0
                    
                    net_proceeds = sell_value - sell_fee - capital_gains_tax
                    total_trading_fees += sell_fee
                    
                    # JEPI 매수
                    buy_fee = net_proceeds * US_TRADING_FEE
                    shares['JEPI'] = (net_proceeds - buy_fee) / prices['JEPI']
                    shares['QQQ'] = 0
                    total_trading_fees += buy_fee
                    initial_investment['JEPI'] = net_proceeds - buy_fee
                    
            else:  # Danger -> Normal: JEPI -> QQQ
                if shares['JEPI'] > 0:
                    # JEPI 매도
                    sell_value = shares['JEPI'] * prices['JEPI']
                    sell_fee = sell_value * US_TRADING_FEE
                    
                    # 양도소득세 계산
                    capital_gain = sell_value - initial_investment.get('JEPI', sell_value)
                    if capital_gain > CAPITAL_GAINS_TAX_THRESHOLD:
                        capital_gains_tax = (capital_gain - CAPITAL_GAINS_TAX_THRESHOLD) * CAPITAL_GAINS_TAX_RATE
                    else:
                        capital_gains_tax = 0
                    
                    net_proceeds = sell_value - sell_fee - capital_gains_tax
                    total_trading_fees += sell_fee
                    
                    # QQQ 매수
                    buy_fee = net_proceeds * US_TRADING_FEE
                    shares['QQQ'] = (net_proceeds - buy_fee) / prices['QQQ']
                    shares['JEPI'] = 0
                    total_trading_fees += buy_fee
                    initial_investment['QQQ'] = net_proceeds - buy_fee
            
            current_mode = signal
    
    print(f"  ✓ 총 배당 수령: ${total_dividends_received:,.2f}")
    print(f"  ✓ 배당세: ${total_dividend_tax:,.2f}")
    print(f"  ✓ 거래 수수료: ${total_trading_fees:,.2f}")
    
    return pd.Series(portfolio_values, index=df_prices.index), {
        'total_dividends': total_dividends_received,
        'dividend_tax': total_dividend_tax,
        'trading_fees': total_trading_fees
    }

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
    print("=" * 70)
    print("💰 현실적인 백테스트: 배당 + 거래비용 + 세금 포함")
    print("=" * 70)
    
    # 데이터 수집
    df_prices, df_dividends = fetch_data_with_dividends()
    
    # 시그널 계산
    df_prices = calculate_signal_optimized(df_prices)
    is_danger = df_prices['is_danger']
    
    # 백테스트 실행
    portfolio_schd, costs_schd = backtest_realistic(df_prices, df_dividends, is_danger, 'SCHD')
    portfolio_spy, costs_spy = backtest_realistic(df_prices, df_dividends, is_danger, 'SPY')
    
    # 성과 분석
    stats_schd = analyze_performance(portfolio_schd)
    stats_spy = analyze_performance(portfolio_spy)
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("📊 현실적인 백테스트 결과 (배당 + 비용 + 세금 포함)")
    print("=" * 70)
    print(f"{'Metric':<25} {'SCHD':<20} {'SPY':<20}")
    print("-" * 70)
    print(f"{'Final Value':<25} ${stats_schd['Final Value']:>18,.0f} ${stats_spy['Final Value']:>18,.0f}")
    print(f"{'Total Return':<25} {stats_schd['Total Return']:>18.2f}% {stats_spy['Total Return']:>18.2f}%")
    print(f"{'CAGR':<25} {stats_schd['CAGR']:>18.2f}% {stats_spy['CAGR']:>18.2f}%")
    print(f"{'MDD':<25} {stats_schd['MDD']:>18.2f}% {stats_spy['MDD']:>18.2f}%")
    print(f"{'Sharpe':<25} {stats_schd['Sharpe']:>19.2f} {stats_spy['Sharpe']:>19.2f}")
    print(f"{'Volatility':<25} {stats_schd['Volatility']:>18.2f}% {stats_spy['Volatility']:>18.2f}%")
    
    print("\n" + "-" * 70)
    print("💵 비용 및 수익 분석")
    print("-" * 70)
    print(f"{'Item':<25} {'SCHD':<20} {'SPY':<20}")
    print("-" * 70)
    print(f"{'Total Dividends':<25} ${costs_schd['total_dividends']:>18,.2f} ${costs_spy['total_dividends']:>18,.2f}")
    print(f"{'Dividend Tax (15.4%)':<25} ${costs_schd['dividend_tax']:>18,.2f} ${costs_spy['dividend_tax']:>18,.2f}")
    print(f"{'Trading Fees':<25} ${costs_schd['trading_fees']:>18,.2f} ${costs_spy['trading_fees']:>18,.2f}")
    
    net_dividend_schd = costs_schd['total_dividends'] - costs_schd['dividend_tax']
    net_dividend_spy = costs_spy['total_dividends'] - costs_spy['dividend_tax']
    print(f"{'Net Dividends':<25} ${net_dividend_schd:>18,.2f} ${net_dividend_spy:>18,.2f}")
    
    # 승자 판정
    print("\n" + "=" * 70)
    print("🏆 승자 판정")
    print("=" * 70)
    
    winner = 'SCHD' if stats_schd['Final Value'] > stats_spy['Final Value'] else 'SPY'
    diff = abs(stats_schd['Final Value'] - stats_spy['Final Value'])
    
    print(f"🎯 최종 승자: {winner}")
    print(f"   차이: ${diff:,.0f} ({diff/min(stats_schd['Final Value'], stats_spy['Final Value'])*100:.2f}%)")
    
    # 차트 생성
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. 포트폴리오 가치
    axes[0].plot(portfolio_schd.index, portfolio_schd, label='SCHD (배당+비용 포함)', linewidth=2)
    axes[0].plot(portfolio_spy.index, portfolio_spy, label='SPY (배당+비용 포함)', linewidth=2, alpha=0.8)
    axes[0].set_title('현실적인 백테스트: SCHD vs SPY (배당 재투자 + 거래비용 + 세금)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. 시그널
    axes[1].fill_between(df_prices.index, 0, 1, where=is_danger==1, alpha=0.3, color='red', label='Danger (JEPI)')
    axes[1].fill_between(df_prices.index, 0, 1, where=is_danger==0, alpha=0.3, color='green', label='Normal (QQQ)')
    axes[1].set_title('시그널 (최적 파라미터: 15/30/25/65)', fontsize=12)
    axes[1].set_ylabel('Signal')
    axes[1].set_xlabel('Date')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_backtest_with_dividends.png', dpi=150)
    print(f"\n📈 차트 저장: realistic_backtest_with_dividends.png")
    
    # 리포트 저장
    with open('realistic_backtest_report.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("현실적인 백테스트 결과 (배당 + 거래비용 + 세금 포함)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"기간: {df_prices.index[0].date()} ~ {df_prices.index[-1].date()}\n")
        f.write(f"초기 자본: ${INITIAL_CAPITAL:,}\n\n")
        f.write(f"{'Metric':<25} {'SCHD':<20} {'SPY':<20}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Final Value':<25} ${stats_schd['Final Value']:>18,.0f} ${stats_spy['Final Value']:>18,.0f}\n")
        f.write(f"{'CAGR':<25} {stats_schd['CAGR']:>18.2f}% {stats_spy['CAGR']:>18.2f}%\n")
        f.write(f"{'MDD':<25} {stats_schd['MDD']:>18.2f}% {stats_spy['MDD']:>18.2f}%\n")
        f.write(f"{'Sharpe':<25} {stats_schd['Sharpe']:>19.2f} {stats_spy['Sharpe']:>19.2f}\n\n")
        f.write("비용 및 수익:\n")
        f.write(f"{'Total Dividends':<25} ${costs_schd['total_dividends']:>18,.2f} ${costs_spy['total_dividends']:>18,.2f}\n")
        f.write(f"{'Net Dividends':<25} ${net_dividend_schd:>18,.2f} ${net_dividend_spy:>18,.2f}\n")
        f.write(f"{'Trading Fees':<25} ${costs_schd['trading_fees']:>18,.2f} ${costs_spy['trading_fees']:>18,.2f}\n\n")
        f.write(f"승자: {winner}\n")
    
    print(f"📄 리포트 저장: realistic_backtest_report.txt\n")

if __name__ == "__main__":
    main()
