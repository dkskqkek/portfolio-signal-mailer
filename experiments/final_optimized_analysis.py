# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os
from logging import getLogger, basicConfig, INFO

# --- 시스템 경로 추가 및 모듈 임포트 ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'crash_detection_system'))

from src.signal_processor import SignalProcessor
print("Successfully imported crash_detection_system modules.")

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

# --- 데이터 다운로드 ---
def get_data(tickers, ohlcv_tickers, start_date, end_date):
    logger.info(f"Downloading data for {tickers}...")
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    adj_close = pd.DataFrame(index=raw_data.index)
    for ticker in tickers:
        if ('Adj Close', ticker) in raw_data.columns:
            adj_close[ticker] = raw_data[('Adj Close', ticker)]
        elif ('Close', ticker) in raw_data.columns:
            adj_close[ticker] = raw_data[('Close', ticker)]
    
    ohlcv_cols = [(m, t) for t in ohlcv_tickers for m in ['Open', 'High', 'Low', 'Close', 'Volume'] if (m, t) in raw_data.columns]
    ohlcv_data = raw_data[ohlcv_cols].ffill()
    adj_close.ffill(inplace=True)
    
    common_index = adj_close.index.intersection(ohlcv_data.index)
    adj_close, ohlcv_data = adj_close.loc[common_index].dropna(), ohlcv_data.loc[common_index].dropna()
    return adj_close, ohlcv_data

# --- 최종 최적화 신호 생성 로직 ---
def generate_final_optimized_signals(ohlcv_data, signal_ticker='QQQ', vix_ticker='^VIX'):
    logger.info("Generating FINAL OPTIMIZED signals (Triple-Lock with AND condition)...")
    
    # 데이터 및 지표 생성
    processor = SignalProcessor()
    price_data = ohlcv_data.loc[:, (slice(None), signal_ticker)]
    price_data.columns = price_data.columns.droplevel(1)
    vix_data = ohlcv_data.loc[:, (slice(None), vix_ticker)]
    vix_data.columns = vix_data.columns.droplevel(1)
    
    log_returns = np.log(price_data['Close'] / price_data['Close'].shift(1)).dropna()
    realized_vol = processor.calculate_realized_volatility(log_returns, window=20)
    
    if len(log_returns) < 252: raise ValueError("Not enough data for HMM.")
    
    processor.train_hmm_regime_detector(log_returns, realized_vol)
    hmm_regime = processor.predict_regime(log_returns, realized_vol)
    
    rsi = processor.calculate_rsi(price_data['Close'])
    adx = processor.calculate_adx(price_data['High'], price_data['Low'], price_data['Close'])
    
    df = pd.concat([price_data['Close'], vix_data['Close'].rename('vix'), hmm_regime, rsi.rename('rsi'), adx.rename('adx')], axis=1).dropna()
    
    # --- 최종 3중 잠금 로직 ---
    df['HMM_Smooth'] = df['HMM_Regime'].rolling(window=3).median().fillna(method='bfill')
    
    signals = pd.Series(1, index=df.index, name='Position')
    current_position = 1

    for i in range(1, len(df)):
        # --- 매도 조건 (QQQ -> JEPI) ---
        sell_signal = False
        is_crisis = (df['HMM_Smooth'].iloc[i] == 2)
        
        # 조건 1: 스무딩된 '위기' 국면 + RSI < 45
        if is_crisis and df['rsi'].iloc[i] < 45:
            sell_signal = True
        
        # 조건 2: 정상 국면이지만 모멘텀/변동성 '동시에' 붕괴 (AND 조건)
        elif not is_crisis:
            cond_a = (df['rsi'].iloc[i-1] > 70) and (df['rsi'].iloc[i] < 40) # RSI 급락
            cond_c = (df['vix'].iloc[i] > 30) # VIX 급등
            if cond_a and cond_c: # ★★★ 핵심 수정: OR -> AND ★★★
                sell_signal = True
        
        # 필터: ADX < 25 이면 매도 신호 무시
        if df['adx'].iloc[i] < 25:
            sell_signal = False
            
        # --- 매수 조건 (JEPI -> QQQ) ---
        buy_signal = False
        if current_position == 0:
            if df['rsi'].iloc[i] > 50:
                buy_signal = True
        
        # --- 최종 포지션 결정 ---
        if sell_signal:
            current_position = 0
        elif buy_signal:
            current_position = 1
        signals.iloc[i] = current_position

    logger.info("Final optimized signals generated.")
    details = df.join(signals).dropna()
    return signals, details

# --- 백테스트 및 결과 분석 (기존 함수 재사용) ---
def run_backtest(adj_close_data, strategy_configs):
    results = {}
    for name, config in strategy_configs.items():
        returns = adj_close_data.pct_change().fillna(0)
        portfolio_returns = pd.Series(index=adj_close_data.index, data=0.0)
        static_weights = config['static_weights']
        dynamic_asset, switch_asset = config['dynamic_asset'], config['switch_asset']
        signal = config['signal']
        
        for asset, weight in static_weights.items():
            portfolio_returns += returns[asset] * weight
        dynamic_weight = 1.0 - sum(static_weights.values())
        dynamic_returns = np.where(signal.shift(1) == 1, returns[dynamic_asset] * dynamic_weight, returns[switch_asset] * dynamic_weight)
        portfolio_returns += dynamic_returns
        results[name] = (1 + portfolio_returns.fillna(0)).cumprod()
    return pd.DataFrame(results)

def get_transition_dates(signal_details):
    pos_changes = signal_details['Position'].diff()
    return pos_changes[pos_changes > 0].index, pos_changes[pos_changes < 0].index

# --- 메인 실행 함수 ---
def main():
    tickers = ['SCHD', 'QQQ', 'GLD', 'JEPI', '^KS11', '^VIX']
    ohlcv_tickers = ['QQQ', '^VIX']
    start_date, end_date = '2020-05-21', pd.to_datetime('today').strftime('%Y-%m-%d')

    adj_close, ohlcv_data = get_data(tickers, ohlcv_tickers, start_date, end_date)
    final_signal, signal_details = generate_final_optimized_signals(ohlcv_data)
    
    strategy_config = { 'Final Optimized': { 'static_weights': {'SCHD': 0.38, 'GLD': 0.05, '^KS11': 0.19}, 'dynamic_asset': 'QQQ', 'switch_asset': 'JEPI', 'signal': final_signal } } 
    
    portfolio_values = run_backtest(adj_close.loc[final_signal.index.min():], strategy_config)
    
    print("\n--- Final Optimized Strategy Performance ---")
    cagr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252 / len(portfolio_values)) - 1
    vol = portfolio_values.pct_change().std() * np.sqrt(252)
    sharpe = cagr / vol
    rolling_max = portfolio_values.cummax()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    mdd = drawdown.min()
    
    print(f"CAGR: {cagr.values[0]:.2%}")
    print(f"Volatility: {vol.values[0]:.2%}")
    print(f"Sharpe Ratio: {sharpe.values[0]:.2f}")
    print(f"MDD: {mdd.values[0]:.2%}")

    buy_dates, sell_dates = get_transition_dates(signal_details)
    print("\n--- Final Optimized Transition Dates ---")
    print(f"Total Sell Signals: {len(sell_dates)}")
    print(f"Total Buy Signals: {len(buy_dates)}")
    print("\nSell Dates (QQQ -> JEPI):\n", sell_dates.strftime('%Y-%m-%d').to_list())
    print("\nBuy Dates (JEPI -> QQQ):\n", buy_dates.strftime('%Y-%m-%d').to_list())
    
    setup_korean_font()
    fig, axes = plt.subplots(3, 1, figsize=(18, 16), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
    (portfolio_values * 100_000_000).plot(ax=axes[0])
    axes[0].set_title('Final Optimized Portfolio Performance', fontsize=16)
    axes[0].set_ylabel('Portfolio Value'); axes[0].grid(True)
    
    signal_details['Close'].plot(ax=axes[1], label='QQQ Close', color='k', alpha=0.8)
    axes[1].plot(buy_dates, signal_details['Close'].loc[buy_dates], '^', ms=10, color='g', label='Switch to QQQ')
    axes[1].plot(sell_dates, signal_details['Close'].loc[sell_dates], 'v', ms=10, color='r', label='Switch to JEPI')
    axes[1].set_title('Final Trade Signals', fontsize=16); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(signal_details.index, signal_details['HMM_Regime'], label='HMM Original', color='orange', alpha=0.5)
    axes[2].plot(signal_details.index, signal_details['HMM_Smooth'], label='HMM Smoothed', color='purple', drawstyle='steps-post')
    axes[2].set_title('HMM Regime (Original vs. Smoothed)', fontsize=16); axes[2].legend(); axes[2].set_yticks([0,1,2])
    
    plt.tight_layout()
    plt.savefig('final_optimized_portfolio_comparison.png')
    logger.info("Plot saved to 'final_optimized_portfolio_comparison.png'")

if __name__ == '__main__':
    main()
