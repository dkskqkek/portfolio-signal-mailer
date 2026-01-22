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
from src.strategy import Strategy, SignalType
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

# --- '쿨다운' 신호 생성 로직 ---
def generate_cooldown_signals(ohlcv_data, signal_ticker='QQQ', vix_ticker='^VIX', cooldown_period=5):
    logger.info(f"Generating signals with {cooldown_period}-day Cooldown Strategy...")
    
    # 1. 원본 신호 생성 (최초의 성공적인 로직 사용)
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
    
    # 원본 매도 신호 (SELL 또는 STRONG_SELL)
    is_sell_triggered = (base_signals['signal'] <= SignalType.SELL.value)

    # 2. '쿨다운' 로직 적용
    positions = pd.Series(1, index=df.index, name='Position')
    cooldown_counter = 0
    for i in range(1, len(df)):
        if cooldown_counter > 0:
            positions.iloc[i] = 0 # 쿨다운 중에는 JEPI 유지
            cooldown_counter -= 1
        else:
            # 원본 매도 신호가 발생했고, 현재 QQQ 보유 중이면
            if is_sell_triggered.iloc[i] and positions.iloc[i-1] == 1:
                positions.iloc[i] = 0 # JEPI로 전환
                cooldown_counter = cooldown_period # 쿨다운 시작
            # 매도 신호가 없고, 현재 JEPI 보유 중이면
            elif not is_sell_triggered.iloc[i] and positions.iloc[i-1] == 0:
                positions.iloc[i] = 1 # QQQ로 복귀
            # 그 외의 경우는 이전 포지션 유지
            else:
                positions.iloc[i] = positions.iloc[i-1]

    logger.info("Cooldown signals generated.")
    details = df.join(positions).join(base_signals['signal_reason']).dropna()
    return positions, details

# --- 백테스트 및 결과 분석 (기존 함수 재사용) ---
def run_backtest(adj_close_data, strategy_configs):
    results = {}
    for name, config in strategy_configs.items():
        returns = adj_close_data.pct_change().fillna(0)
        portfolio_returns = pd.Series(index=adj_close_data.index, data=0.0)
        static_weights, dyn_asset, sw_asset, signal = config.values()
        
        for asset, weight in static_weights.items():
            portfolio_returns += returns[asset] * weight
        dyn_weight = 1.0 - sum(static_weights.values())
        dyn_returns = np.where(signal.shift(1) == 1, returns[dyn_asset] * dyn_weight, returns[sw_asset] * dyn_weight)
        portfolio_returns += dyn_returns
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
    cooldown_signal, signal_details = generate_cooldown_signals(ohlcv_data, cooldown_period=5)
    
    strategy_config = { 'Cooldown Strategy': { 'static_weights': {'SCHD': 0.38, 'GLD': 0.05, '^KS11': 0.19}, 'dynamic_asset': 'QQQ', 'switch_asset': 'JEPI', 'signal': cooldown_signal } }
    
    portfolio_values = run_backtest(adj_close.loc[cooldown_signal.index.min():], strategy_config)
    
    print("\n--- Cooldown Strategy Performance ---")
    cagr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252 / len(portfolio_values)) - 1
    vol = portfolio_values.pct_change().std() * np.sqrt(252)
    sharpe = cagr / vol
    mdd = ((portfolio_values.cummax() - portfolio_values) / portfolio_values.cummax()).min()
    
    print(f"CAGR: {cagr.values[0]:.2%}")
    print(f"Volatility: {vol.values[0]:.2%}")
    print(f"Sharpe Ratio: {sharpe.values[0]:.2f}")
    print(f"MDD: {mdd.values[0]:.2%}")

    buy_dates, sell_dates = get_transition_dates(signal_details)
    print("\n--- Cooldown Strategy Transition Dates ---")
    print(f"Total Sell Signals: {len(sell_dates)}")
    print(f"Total Buy Signals: {len(buy_dates)}")
    print("\nSell Dates (QQQ -> JEPI):\n", sell_dates.strftime('%Y-%m-%d').to_list())
    print("\nBuy Dates (JEPI -> QQQ):\n", buy_dates.strftime('%Y-%m-%d').to_list())
    
    # --- 시각화 ---
    setup_korean_font()
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    (portfolio_values * 100_000_000).plot(ax=axes[0], legend=True)
    axes[0].set_title('Cooldown Strategy Portfolio Performance', fontsize=16)
    axes[0].set_ylabel('Portfolio Value'); axes[0].grid(True)
    
    signal_details['Close'].plot(ax=axes[1], label='QQQ Close', color='k', alpha=0.8)
    axes[1].plot(buy_dates, signal_details['Close'].loc[buy_dates], '^', ms=10, color='g', label='Switch to QQQ')
    axes[1].plot(sell_dates, signal_details['Close'].loc[sell_dates], 'v', ms=10, color='r', label='Switch to JEPI')
    axes[1].set_title('Cooldown Strategy Trade Signals', fontsize=16); axes[1].legend(); axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('cooldown_strategy_comparison.png')
    logger.info("Plot saved to 'cooldown_strategy_comparison.png'")

if __name__ == '__main__':
    main()
