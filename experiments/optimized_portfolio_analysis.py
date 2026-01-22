# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os
from logging import getLogger, basicConfig, INFO

# --- 시스템 경로 추가 ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'crash_detection_system'))

# --- crash_detection_system 모듈 임포트 ---
try:
    from src.signal_processor import SignalProcessor
    from src.strategy import Strategy, SignalType
    print("Successfully imported crash_detection_system modules.")
except ImportError as e:
    print(f"Error importing crash_detection_system modules: {e}")
    sys.exit(1)

# --- 로깅 및 폰트 설정 ---
basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)

def setup_korean_font():
    font_path = next((font for font in fm.findSystemFonts(fontpaths=None, fontext='ttf') if 'Malgun Gothic' in font or 'AppleGothic' in font), None)
    if font_path:
        fm.fontManager.addfont(font_path)
        plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
        plt.rc('axes', unicode_minus=False)
        logger.info(f"Korean font set to: {fm.FontProperties(fname=font_path).get_name()}")
    else:
        logger.warning("Korean font not found.")

# --- 데이터 다운로드 ---
def get_data(tickers, ohlcv_tickers, start_date, end_date):
    logger.info(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    adj_close = pd.DataFrame(index=raw_data.index)
    for ticker in tickers:
        if ('Adj Close', ticker) in raw_data.columns:
            adj_close[ticker] = raw_data[('Adj Close', ticker)]
        elif ('Close', ticker) in raw_data.columns:
            adj_close[ticker] = raw_data[('Close', ticker)]
    
    ohlcv_cols = [(m, t) for t in ohlcv_tickers for m in ['Open', 'High', 'Low', 'Close', 'Volume'] if (m, t) in raw_data.columns]
    ohlcv_data = raw_data[ohlcv_cols]

    adj_close.ffill(inplace=True)
    ohlcv_data.ffill(inplace=True)
    
    common_index = adj_close.index.intersection(ohlcv_data.index)
    adj_close, ohlcv_data = adj_close.loc[common_index], ohlcv_data.loc[common_index]
    
    adj_close.dropna(inplace=True)
    ohlcv_data.dropna(inplace=True)
    
    logger.info("Data download and preprocessing complete.")
    return adj_close, ohlcv_data

# --- 최적화된 신호 생성 로직 ---
def generate_optimized_signals(ohlcv_data, signal_ticker='QQQ', vix_ticker='^VIX'):
    logger.info("Generating OPTIMIZED signals with Triple-Lock Strategy...")
    
    # 1. 원본 데이터 및 지표 생성
    processor = SignalProcessor()
    price_data = ohlcv_data.loc[:, (slice(None), signal_ticker)]
    price_data.columns = price_data.columns.droplevel(1)
    vix_data = ohlcv_data.loc[:, (slice(None), vix_ticker)]
    vix_data.columns = vix_data.columns.droplevel(1)
    
    log_returns = np.log(price_data['Close'] / price_data['Close'].shift(1)).dropna()
    realized_vol = processor.calculate_realized_volatility(log_returns, window=20)
    
    if len(log_returns) < 252: raise ValueError("Not enough data to train HMM.")
    
    processor.train_hmm_regime_detector(log_returns, realized_vol)
    hmm_regime = processor.predict_regime(log_returns, realized_vol)
    
    rsi = processor.calculate_rsi(price_data['Close'])
    adx = processor.calculate_adx(price_data['High'], price_data['Low'], price_data['Close'])
    
    df = pd.concat([price_data['Close'], vix_data['Close'].rename('vix'), hmm_regime, rsi.rename('rsi'), adx.rename('adx')], axis=1).dropna()
    
    # --- 3중 잠금 최적화 로직 적용 ---
    # Solution B: HMM 스무딩 (3일의 법칙)
    df['HMM_Smooth'] = df['HMM_Regime'].rolling(window=3).median().fillna(method='bfill')
    
    signals = pd.Series(1, index=df.index, name='Position') # 1: QQQ(보유)
    current_position = 1

    for i in range(1, len(df)):
        # --- 매도 조건 (QQQ -> JEPI) ---
        is_crisis_regime = (df['HMM_Smooth'].iloc[i] == 2)
        sell_signal = False
        
        # 조건 1: 스무딩된 '위기' 국면 + RSI < 45
        if is_crisis_regime and df['rsi'].iloc[i] < 45:
            sell_signal = True
        
        # 필터: ADX < 25 이면 매도 신호 무시 (Solution C: ADX 필터 강화)
        if df['adx'].iloc[i] < 25:
            sell_signal = False
            
        # --- 매수 조건 (JEPI -> QQQ) ---
        buy_signal = False
        if current_position == 0: # 현재 JEPI 보유 중일 때만 복귀 고려
            # Solution A: 히스테리시스 (Deadband) - RSI가 50을 '초과'해야 복귀
            if df['rsi'].iloc[i] > 50:
                buy_signal = True
        
        # --- 최종 포지션 결정 ---
        if sell_signal:
            current_position = 0 # JEPI로 전환
        elif buy_signal:
            current_position = 1 # QQQ로 복귀
            
        signals.iloc[i] = current_position

    logger.info("Optimized signals generated successfully.")
    details = df.join(signals).dropna()
    return signals, details

# --- 백테스트 및 결과 분석 (기존 함수 재사용) ---
def run_backtest(adj_close_data, strategy_configs, initial_capital=1000000):
    logger.info("Starting backtests...")
    results = {}
    for name, config in strategy_configs.items():
        logger.info(f"Running backtest for {name}...")
        returns = adj_close_data.pct_change().fillna(0)
        portfolio_returns = pd.Series(index=adj_close_data.index, data=0.0)
        
        static_weights = config['static_weights']
        dynamic_asset, switch_asset = config['dynamic_asset'], config['switch_asset']
        signal = config['signal']
        
        for asset, weight in static_weights.items():
            portfolio_returns += returns[asset] * weight
            
        dynamic_weight = 1.0 - sum(static_weights.values())
        dynamic_returns = np.where(signal.shift(1) == 1, 
                                   returns[dynamic_asset] * dynamic_weight, 
                                   returns[switch_asset] * dynamic_weight)
        portfolio_returns += dynamic_returns
        
        results[name] = initial_capital * (1 + portfolio_returns.fillna(0)).cumprod()
    logger.info("All backtests complete.")
    return pd.DataFrame(results)

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
        metrics[name] = {"CAGR": f"{cagr:.2%}", "Volatility": f"{volatility:.2%}", "Sharpe Ratio": f"{sharpe_ratio:.2f}", "MDD": f"{mdd:.2%}"}
    return pd.DataFrame(metrics)

def get_transition_dates(signal_details):
    position_changes = signal_details['Position'].diff()
    buy_dates = position_changes[position_changes > 0].index
    sell_dates = position_changes[position_changes < 0].index
    return buy_dates, sell_dates

def plot_results(portfolio_values, signal_details, signal_ticker='QQQ'):
    logger.info("Generating plots...")
    setup_korean_font()
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 16), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    
    portfolio_values.plot(ax=axes[0])
    axes[0].set_title('Optimized Portfolio Strategy Comparison', fontsize=16)
    axes[0].set_ylabel('Portfolio Value')
    axes[0].grid(True)
    
    signal_details['Close'].plot(ax=axes[1], label=f'{signal_ticker} Close', color='black', alpha=0.8)
    buy_dates, sell_dates = get_transition_dates(signal_details)
    axes[1].plot(buy_dates, signal_details['Close'].loc[buy_dates], '^', markersize=10, color='g', label='Switch to QQQ')
    axes[1].plot(sell_dates, signal_details['Close'].loc[sell_dates], 'v', markersize=10, color='r', label='Switch to JEPI')
    axes[1].set_title(f'{signal_ticker} Trade Signals (Optimized)', fontsize=16)
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(signal_details.index, signal_details['HMM_Regime'], label='HMM Original', color='orange', alpha=0.5)
    axes[2].plot(signal_details.index, signal_details['HMM_Smooth'], label='HMM Smoothed (3d Median)', color='purple', drawstyle='steps-post')
    axes[2].set_yticks([0, 1, 2]); axes[2].set_yticklabels(['Bull (0)', 'Correction (1)', 'Crisis (2)'])
    axes[2].set_title('HMM Regime (Original vs. Smoothed)', fontsize=16)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('optimized_portfolio_comparison.png')
    logger.info("Plots saved to 'optimized_portfolio_comparison.png'")
    plt.show()

def main():
    # --- 기본 설정 ---
    tickers = ['SCHD', 'QQQ', 'GLD', 'JEPI', '^KS11', '^VIX']
    ohlcv_tickers = ['QQQ', '^VIX']
    start_date, end_date = '2020-05-21', pd.to_datetime('today').strftime('%Y-%m-%d')

    # --- 데이터 로드 ---
    adj_close, ohlcv_data = get_data(tickers, ohlcv_tickers, start_date, end_date)
    
    # --- 최적화된 신호 생성 ---
    optimized_signal, signal_details = generate_optimized_signals(ohlcv_data)
    
    # --- 백테스트 실행 ---
    strategy_config = {
        'Optimized Portfolio': {
            'static_weights': {'SCHD': 0.38, 'GLD': 0.05, '^KS11': 0.19},
            'dynamic_asset': 'QQQ', 'switch_asset': 'JEPI', 'signal': optimized_signal
        }
    }
    valid_start = optimized_signal.index.min()
    portfolio_values = run_backtest(adj_close.loc[valid_start:], strategy_config)
    
    # --- 결과 비교 및 출력 ---
    print("\n--- Optimized Strategy Performance ---")
    perf_metrics = calculate_performance_metrics(portfolio_values)
    print(perf_metrics)

    print("\n--- Optimized Transition Dates ---")
    buy_dates, sell_dates = get_transition_dates(signal_details)
    print("\nSell Dates (QQQ -> JEPI):")
    print(sell_dates.strftime('%Y-%m-%d').to_list())
    print("\nBuy Dates (JEPI -> QQQ):")
    print(buy_dates.strftime('%Y-%m-%d').to_list())
    
    # --- 시각화 ---
    plot_results(portfolio_values, signal_details)

if __name__ == '__main__':
    main()
