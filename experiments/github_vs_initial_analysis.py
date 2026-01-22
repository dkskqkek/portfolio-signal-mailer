# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import time
import requests
import os
from datetime import datetime, timedelta
from logging import getLogger, basicConfig, INFO

# --- 시스템 경로 추가 및 모듈 임포트 ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'crash_detection_system'))
sys.path.append(os.path.join(current_dir, 'signal_mailer'))

from src.signal_processor import SignalProcessor
from src.strategy import Strategy, SignalType
from signal_detector import SignalDetector as GithubSignalDetector # 이름 충돌 방지
print("Successfully imported crash_detection_system and signal_mailer modules.")

# --- 로깅 및 폰트 설정 ---
basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)

def setup_korean_font():
    font_path = next((font for font in fm.findSystemFonts(fontpaths=None, fontext='ttf') if 'Malgun Gothic' in font or 'AppleGothic' in font), None)
    if font_path:
        fm.fontManager.addfont(font_path)
        plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
        plt.rc('axes', unicode_minus=False)
    else:
        logger.warning("Korean font not found.")

import time
import requests # Needed for handling requests exceptions
# ... (rest of the imports) ...

# --- 데이터 다운로드 ---
def get_data(tickers, ohlcv_tickers, start_date, end_date, max_retries_per_ticker=3, initial_delay_per_ticker=2):
    logger.info(f"Downloading data for {tickers} individually...")
    
    all_raw_data = {}
    
    for ticker in tickers:
        ticker_data = None
        for attempt in range(max_retries_per_ticker):
            try:
                logger.info(f"Downloading {ticker} (Attempt {attempt + 1})...")
                # Download individual ticker data
                ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not ticker_data.empty and 'Close' in ticker_data.columns:
                    all_raw_data[ticker] = ticker_data
                    logger.info(f"Successfully downloaded {ticker}.")
                    break # Success for this ticker
                elif attempt < max_retries_per_ticker - 1:
                    logger.warning(f"Attempt {attempt + 1}: yfinance returned empty data for {ticker}. Retrying in {initial_delay_per_ticker * (2 ** attempt)} seconds.")
                    time.sleep(initial_delay_per_ticker * (2 ** attempt))
                else:
                    logger.error(f"Failed to download data for {ticker} after {max_retries_per_ticker} attempts. Returned empty or unexpected data. Skipping this ticker.")
                    break # Give up on this ticker
            except requests.exceptions.RequestException as e: # Catch network errors
                if attempt < max_retries_per_ticker - 1:
                    logger.warning(f"Attempt {attempt + 1}: Request failed for {ticker} due to {e}. Retrying in {initial_delay_per_ticker * (2 ** attempt)} seconds.")
                    time.sleep(initial_delay_per_ticker * (2 ** attempt))
                else:
                    logger.error(f"Failed to download data for {ticker} after {max_retries_per_ticker} attempts due to network error: {e}. Skipping this ticker.")
                    break
            except Exception as e: # Catch other potential yfinance or pandas errors
                logger.error(f"Attempt {attempt + 1}: An unexpected error occurred downloading {ticker}: {e}. Skipping this ticker.")
                break # Give up on this ticker

        # Short delay between tickers to be more polite to yfinance
        time.sleep(1) 

    if not all_raw_data:
        raise ValueError("Failed to download data for any tickers.")

    # Combine all individual ticker data into a single MultiIndex DataFrame
    combined_raw_data_list = []
    for ticker_name, df_ticker in all_raw_data.items():
        # Ensure column names are unique for concat (e.g. Open, High, Close, etc.)
        # Then create MultiIndex (Ticker, Measure)
        df_ticker.columns = pd.MultiIndex.from_product([[ticker_name], df_ticker.columns])
        combined_raw_data_list.append(df_ticker)
    
    raw_data_multi_index = pd.concat(combined_raw_data_list, axis=1)
    # Swap levels so it's (Measure, Ticker) like yf.download(list) returns
    raw_data_multi_index.columns = raw_data_multi_index.columns.swaplevel(0, 1)
    raw_data_multi_index.sort_index(axis=1, level=0, inplace=True)


    # 1. 백테스팅을 위한 종가 데이터프레임 생성
    adj_close = pd.DataFrame(index=raw_data_multi_index.index)
    for ticker in tickers:
        if ('Adj Close', ticker) in raw_data_multi_index.columns:
            adj_close[ticker] = raw_data_multi_index[('Adj Close', ticker)]
        elif ('Close', ticker) in raw_data_multi_index.columns:
            adj_close[ticker] = raw_data_multi_index[('Close', ticker)]
        else:
            logger.warning(f"No 'Adj Close' or 'Close' data found for {ticker} in combined raw data. Skipping {ticker} for backtesting.")
            if ticker in adj_close.columns: # Remove if partially added
                del adj_close[ticker]
    
    # 2. 신호 생성을 위한 OHLCV 데이터 추출
    ohlcv_cols = []
    measures = ['Open', 'High', 'Low', 'Close', 'Volume']
    for ticker in ohlcv_tickers:
        for measure in measures:
            if (measure, ticker) in raw_data_multi_index.columns:
                ohlcv_cols.append((measure, ticker))
    
    ohlcv_data = raw_data_multi_index[ohlcv_cols].ffill()
    adj_close.ffill(inplace=True)
    
    # Align indices and remove rows with any NaN after ffill
    common_index = adj_close.index.intersection(ohlcv_data.index)
    adj_close = adj_close.loc[common_index].dropna()
    ohlcv_data = ohlcv_data.loc[common_index].dropna()
    
    if adj_close.empty or ohlcv_data.empty:
        raise ValueError("After data cleaning, one or both dataframes are empty. Check data availability and tickers.")

    logger.info("Data download and preprocessing complete.")
    return adj_close, ohlcv_data

# --- 초기 전략 신호 생성 (crash_detection_system 기반) ---
def generate_initial_signals(ohlcv_data, signal_ticker='QQQ', vix_ticker='^VIX'):
    logger.info(f"Generating Initial Strategy signals for {signal_ticker}...")
    
    processor = SignalProcessor()
    strategy = Strategy()
    price_data = ohlcv_data.loc[:, (slice(None), signal_ticker)]; price_data.columns = price_data.columns.droplevel(1)
    vix_data = ohlcv_data.loc[:, (slice(None), vix_ticker)]; vix_data.columns = vix_data.columns.droplevel(1)
    
    log_returns = np.log(price_data['Close'] / price_data['Close'].shift(1)).dropna()
    realized_vol = processor.calculate_realized_volatility(log_returns, window=20)
    
    if len(log_returns) < 252: raise ValueError("Not enough data for HMM.")
    
    processor.train_hmm_regime_detector(log_returns, realized_vol)
    hmm_regime = processor.predict_regime(log_returns, realized_vol)
    
    rsi = processor.calculate_rsi(price_data['Close'])
    adx = processor.calculate_adx(price_data['High'], price_data['Low'], price_data['Close'])
    
    df = pd.concat([price_data['Close'], vix_data['Close'].rename('VIX_Close'), hmm_regime, rsi.rename('RSI'), adx.rename('ADX')], axis=1).dropna()
    
    base_signals = strategy.generate_signals(
        df=df, kalman_smoothed_price=df['Close'], rsi=df['RSI'], adx=df['ADX'],
        hmm_regime=df['HMM_Regime'], vix=df['VIX_Close']
    )
    
    final_signal = (base_signals['signal'] >= SignalType.NEUTRAL.value).astype(int)
    final_signal.name = 'Position'
    details = df.join(final_signal).join(base_signals['signal_reason']).dropna() # Join for easy plotting
    return final_signal, details

# --- GitHub 로직 신호 생성 ---
def generate_github_signals(spy_close_data, signal_ticker='QQQ', portfolio_assets_close=None):
    logger.info("Generating GitHub Strategy signals...")

    # SignalDetector의 calculate_danger_signal 로직을 재현
    # spy_data는 fetch_data에서 Close만 가져오므로 재구성
    
    # 데이터 정렬 및 로그 수익률 계산
    log_returns = np.log(spy_close_data.iloc[1:] / spy_close_data.iloc[:-1])
    
    # 20일 이동평균과 변동성
    ma20_returns = log_returns.rolling(20).mean()
    std20_returns = log_returns.rolling(20).std()
    
    # 임계값 (전체 데이터 기준 percentile)
    ma_threshold = np.nanpercentile(ma20_returns.dropna(), 25)
    vol_threshold = np.nanpercentile(std20_returns.dropna(), 75)
    
    # 신호 생성
    github_signal = pd.Series(1, index=spy_close_data.index, name='Position') # 1: NORMAL (QQQ), 0: DANGER (XLP)
    for i in range(len(spy_close_data)):
        if i < 20: # 20일 데이터 부족
            github_signal.iloc[i] = 1 # 초기에는 NORMAL
            continue

        latest_ma = ma20_returns.iloc[i-1] # 현재 일자의 전날까지 계산
        latest_vol = std20_returns.iloc[i-1] # 현재 일자의 전날까지 계산
        
        is_danger = False
        if latest_ma < ma_threshold:
            is_danger = True
        if latest_vol > vol_threshold:
            is_danger = True

        github_signal.iloc[i] = 0 if is_danger else 1
    
    # signal_details를 위한 DataFrame 생성
    details_df = pd.DataFrame(index=spy_close_data.index)
    details_df['Close'] = portfolio_assets_close[signal_ticker] if portfolio_assets_close is not None else np.nan
    details_df['MA20_Returns'] = ma20_returns
    details_df['Vol20_Returns'] = std20_returns
    details_df['MA_Threshold'] = ma_threshold
    details_df['Vol_Threshold'] = vol_threshold
    details_df['Position'] = github_signal

    return github_signal, details_df

# --- 백테스트 실행 ---
def run_backtest(adj_close_data, strategy_configs, initial_capital=100_000_000):
    logger.info("Starting backtests...")
    results = {}
    for name, config in strategy_configs.items():
        logger.info(f"Running backtest for {name}...")
        returns = adj_close_data.pct_change().fillna(0)
        portfolio_returns = pd.Series(index=adj_close_data.index, data=0.0)
        static_weights = config['static_weights']
        dynamic_asset = config['dynamic_asset']
        switch_asset = config['switch_asset']
        signal = config['signal']
        
        for asset, weight in static_weights.items():
            portfolio_returns += returns[asset] * weight
        dynamic_weight = 1.0 - sum(static_weights.values())
        dynamic_returns = np.where(signal.shift(1) == 1, returns[dynamic_asset] * dynamic_weight, returns[switch_asset] * dynamic_weight)
        portfolio_returns += dynamic_returns
        results[name] = initial_capital * (1 + portfolio_returns.fillna(0)).cumprod()
    logger.info("All backtests complete.")
    return pd.DataFrame(results)

# --- 성과 지표 계산 ---
def calculate_performance_metrics(portfolio_values):
    metrics = {}
    for name, values in portfolio_values.items():
        returns = values.pct_change().dropna()
        cagr = (values.iloc[-1] / values.iloc[0]) ** (252 / len(values)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = cagr / volatility if volatility > 0 else 0
        rolling_max = values.cummax()
        drawdown = (values - rolling_max) / rolling_max
        mdd = drawdown.min()
        
        metrics[name] = {
            "CAGR": f"{cagr:.2%}",
            "Volatility": f"{volatility:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "MDD": f"{mdd:.2%}"
        }
    return pd.DataFrame(metrics)

# --- 전환 날짜 가져오기 ---
def get_transition_dates(signal_details, position_col='Position'):
    pos_changes = signal_details[position_col].diff()
    buy_dates = pos_changes[pos_changes > 0].index
    sell_dates = pos_changes[pos_changes < 0].index
    return buy_dates, sell_dates

# --- 차트 생성 ---
def plot_results(portfolio_values, initial_signal_details, github_signal_details):
    logger.info("Generating plots...")
    setup_korean_font()
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 18), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # 1. 포트폴리오 누적 수익률 비교
    portfolio_values.plot(ax=axes[0])
    axes[0].set_title('Portfolio Strategy Comparison: Initial vs GitHub', fontsize=16)
    axes[0].set_ylabel('Portfolio Value')
    axes[0].grid(True)
    
    # 2. 초기 전략 신호
    initial_signal_details['Close'].plot(ax=axes[1], label='QQQ Close', color='k', alpha=0.8)
    initial_buy, initial_sell = get_transition_dates(initial_signal_details)
    axes[1].plot(initial_buy, initial_signal_details['Close'].loc[initial_buy], '^', ms=10, color='g', label='Initial: Buy QQQ (JEPI->QQQ)')
    axes[1].plot(initial_sell, initial_signal_details['Close'].loc[initial_sell], 'v', ms=10, color='r', label='Initial: Sell QQQ (QQQ->JEPI)')
    axes[1].set_title('Initial Strategy Trade Signals (QQQ <-> JEPI)', fontsize=16)
    axes[1].legend(loc='upper left'); axes[1].grid(True)

    # 3. GitHub 전략 신호
    github_signal_details['Close'].plot(ax=axes[2], label='QQQ Close', color='k', alpha=0.8)
    github_buy, github_sell = get_transition_dates(github_signal_details)
    axes[2].plot(github_buy, github_signal_details['Close'].loc[github_buy], '^', ms=10, color='orange', label='GitHub: Buy QQQ (XLP->QQQ)')
    axes[2].plot(github_sell, github_signal_details['Close'].loc[github_sell], 'v', ms=10, color='blue', label='GitHub: Sell QQQ (QQQ->XLP)')
    axes[2].set_title('GitHub Strategy Trade Signals (QQQ <-> XLP)', fontsize=16)
    axes[2].set_xlabel('Date'); axes[2].legend(loc='upper left'); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('github_vs_initial_comparison.png')
    logger.info("Plot saved to 'github_vs_initial_comparison.png'")

# --- 메인 실행 함수 ---
def main():
    # --- 기본 설정 ---
    # JEPI가 2020-05-21에 상장했으므로 모든 전략의 시작일은 이 날짜로 통일
    start_date = '2020-05-21' 
    end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    initial_capital = 100_000_000 # 초기 자본 1억원

    # --- 데이터 로드 ---
    # GitHub 전략에 XLP 필요, 초기 전략에 JEPI 필요. 공통으로 QQQ, SCHD, GLD, ^KS11, ^VIX, SPY
    all_tickers = ['SCHD', 'QQQ', 'GLD', 'JEPI', 'XLP', '^KS11', '^VIX', 'SPY']
    ohlcv_tickers_initial = ['QQQ', '^VIX'] # 초기 전략에 필요한 OHLCV
    ohlcv_tickers_github = ['SPY'] # GitHub 전략에 필요한 OHLCV (SPY는 Close만 필요) 
    
    # get_data 함수는 ohlcv_tickers 인자를 MultiIndex로 만들어주는 역할
    # 여기서는 GitHub 전략을 위해 SPY의 Close 데이터만 필요하므로, adj_close의 SPY 데이터를 활용
    adj_close_full, ohlcv_data_initial = get_data(all_tickers, ohlcv_tickers_initial, start_date, end_date)
    
    # --- 초기 전략 신호 생성 (QQQ <-> JEPI) ---
    initial_signal, initial_signal_details = generate_initial_signals(
        ohlcv_data_initial, signal_ticker='QQQ', vix_ticker='^VIX'
    )
    # 초기 전략의 포트폴리오 자산 (QQQ, JEPI, SCHD, GLD, ^KS11)
    initial_portfolio_assets = ['QQQ', 'JEPI', 'SCHD', 'GLD', '^KS11']


    # --- GitHub 전략 신호 생성 (QQQ <-> XLP) ---
    # GitHub 전략은 SPY의 Close 데이터를 기반으로 신호 생성
    # generate_github_signals에 백테스팅에 사용될 QQQ 종가를 넘겨줘야 차트에 QQQ 가격이 그려짐
    github_signal, github_signal_details = generate_github_signals(
        adj_close_full['SPY'], signal_ticker='QQQ', portfolio_assets_close=adj_close_full
    )
    # GitHub 전략의 포트폴리오 자산 (QQQ, XLP, SCHD, ^KS11, GLD)
    github_portfolio_assets = ['QQQ', 'XLP', 'SCHD', '^KS11', 'GLD']


    # --- 백테스트 실행 ---
    strategy_configs = {
        'Initial Strategy (QQQ <-> JEPI)': {
            'static_weights': {'SCHD': 0.38, 'GLD': 0.05, '^KS11': 0.19},
            'dynamic_asset': 'QQQ', 'switch_asset': 'JEPI', 'signal': initial_signal
        },
        'GitHub Strategy (QQQ <-> XLP)': {
            'static_weights': {'SCHD': 0.34, 'GLD': 0.15, '^KS11': 0.17},
            'dynamic_asset': 'QQQ', 'switch_asset': 'XLP', 'signal': github_signal
        }
    }
    
    # 백테스트에 필요한 모든 자산의 데이터만 필터링
    # 초기 전략과 GitHub 전략에 필요한 자산이 다름. Union으로 합치기.
    common_backtest_assets = list(set(initial_portfolio_assets + github_portfolio_assets))
    adj_close_filtered = adj_close_full[common_backtest_assets].dropna()

    # 신호 기간에 맞춰 데이터 필터링
    initial_valid_start = initial_signal.index.min()
    github_valid_start = github_signal.index.min()
    min_overall_start = max(initial_valid_start, github_valid_start)
    
    adj_close_for_backtest = adj_close_filtered.loc[min_overall_start:]
    
    # 신호 시리즈도 백테스트 데이터 기간에 맞추기
    initial_signal_final = initial_signal.loc[min_overall_start:]
    github_signal_final = github_signal.loc[min_overall_start:]

    # 업데이트된 신호 시리즈로 config 재정의
    strategy_configs['Initial Strategy (QQQ <-> JEPI)']['signal'] = initial_signal_final
    strategy_configs['GitHub Strategy (QQQ <-> XLP)']['signal'] = github_signal_final
    
    portfolio_values = run_backtest(adj_close_for_backtest, strategy_configs, initial_capital)
    
    # --- 결과 출력 ---
    print("\n--- Performance Comparison (Initial vs GitHub) ---")
    performance_df = calculate_performance_metrics(portfolio_values)
    print(performance_df)

    print("\n--- Signal Transition Dates Comparison ---")
    initial_buy_dates, initial_sell_dates = get_transition_dates(initial_signal_details)
    github_buy_dates, github_sell_dates = get_transition_dates(github_signal_details)

    print("\nInitial Strategy Sell Dates (QQQ -> JEPI):")
    print(initial_sell_dates.strftime('%Y-%m-%d').to_list())
    print(f"(Total: {len(initial_sell_dates)} transitions)")

    print("\nGitHub Strategy Sell Dates (QQQ -> XLP):")
    print(github_sell_dates.strftime('%Y-%m-%d').to_list())
    print(f"(Total: {len(github_sell_dates)} transitions)")
    
    print("\nInitial Strategy Buy Dates (JEPI -> QQQ):")
    print(initial_buy_dates.strftime('%Y-%m-%d').to_list())
    print(f"(Total: {len(initial_buy_dates)} transitions)")

    print("\nGitHub Strategy Buy Dates (XLP -> QQQ):")
    print(github_buy_dates.strftime('%Y-%m-%d').to_list())
    print(f"(Total: {len(github_buy_dates)} transitions)")

    # --- 시각화 ---
    # 차트 생성을 위해 signal_details도 백테스트 기간에 맞춰 잘라야 함
    initial_signal_details_final = initial_signal_details.loc[min_overall_start:]
    github_signal_details_final = github_signal_details.loc[min_overall_start:]

    plot_results(portfolio_values, initial_signal_details_final, github_signal_details_final)

if __name__ == '__main__':
    main()
