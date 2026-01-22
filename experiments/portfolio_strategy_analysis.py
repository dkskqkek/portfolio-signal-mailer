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
# 현재 파일의 디렉토리를 기준으로 crash_detection_system 경로 설정
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd() # 대화형 환경용
sys.path.append(os.path.join(current_dir, 'crash_detection_system'))

# --- crash_detection_system 모듈 임포트 ---
try:
    from src.signal_processor import SignalProcessor
    from src.strategy import Strategy, SignalType
    print("Successfully imported crash_detection_system modules.")
except ImportError as e:
    print(f"Error importing crash_detection_system modules: {e}")
    print("Please ensure the script is run from the project root directory 'gg'.")
    sys.exit(1)

# --- 로깅 설정 ---
basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)

# --- 한글 폰트 설정 ---
def setup_korean_font():
    """matplotlib에 한글 폰트를 설정합니다."""
    font_path = None
    for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
        if 'Malgun Gothic' in font or 'AppleGothic' in font:
            font_path = font
            break
    
    if font_path:
        fm.fontManager.addfont(font_path)
        plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
        plt.rc('axes', unicode_minus=False)
        logger.info(f"Korean font set to: {fm.FontProperties(fname=font_path).get_name()}")
    else:
        logger.warning("Malgun Gothic or AppleGothic font not found. Please install it for Korean characters.")

# --- 데이터 다운로드 ---
def get_data(tickers, ohlcv_tickers, start_date, end_date):
    """Yahoo Finance에서 주가 데이터를 다운로드하고 전처리합니다. (최종 수정)"""
    logger.info(f"Downloading data for {tickers} from {start_date} to {end_date}...")
    
    # 모든 티커의 전체 데이터 다운로드
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # 1. 백테스팅을 위한 종가 데이터프레임 생성
    adj_close = pd.DataFrame(index=raw_data.index)
    for ticker in tickers:
        # 'Adj Close'가 있는지 확인. MultiIndex 접근은 튜플로.
        if ('Adj Close', ticker) in raw_data.columns:
            adj_close[ticker] = raw_data[('Adj Close', ticker)]
        # 'Adj Close'가 없으면 'Close' 사용
        elif ('Close', ticker) in raw_data.columns:
            adj_close[ticker] = raw_data[('Close', ticker)]
        else:
            logger.warning(f"No 'Adj Close' or 'Close' data found for {ticker}")

    # 2. 신호 생성을 위한 OHLCV 데이터 추출
    ohlcv_cols = []
    measures = ['Open', 'High', 'Low', 'Close', 'Volume']
    for ticker in ohlcv_tickers:
        for measure in measures:
            if (measure, ticker) in raw_data.columns:
                ohlcv_cols.append((measure, ticker))
    
    ohlcv_data = raw_data[ohlcv_cols]

    # 결측치 처리 및 인덱스 정렬
    adj_close = adj_close.ffill().dropna()
    ohlcv_data = ohlcv_data.ffill().dropna()

    common_index = adj_close.index.intersection(ohlcv_data.index)
    adj_close = adj_close.loc[common_index]
    ohlcv_data = ohlcv_data.loc[common_index]
    
    logger.info("Data download and preprocessing complete.")
    return adj_close, ohlcv_data

# --- 충돌 회피 신호 생성 ---
def generate_crash_avoidance_signal(ohlcv_data, signal_ticker='QQQ', vix_ticker='^VIX'):
    """
    crash_detection_system 로직을 사용하여 QQQ에 대한 매매 신호를 생성합니다.
    """
    logger.info(f"Generating crash avoidance signals for {signal_ticker}...")
    
    processor = SignalProcessor()
    strategy = Strategy()
    
    # 신호 생성에 필요한 데이터 추출
    price_data = ohlcv_data.loc[:, (slice(None), signal_ticker)]
    price_data.columns = price_data.columns.droplevel(1) # MultiIndex -> SingleIndex

    vix_data = ohlcv_data.loc[:, (slice(None), vix_ticker)]
    vix_data.columns = vix_data.columns.droplevel(1) # MultiIndex -> SingleIndex
    
    # 1. 수익률 및 변동성 계산
    log_returns = np.log(price_data['Close'] / price_data['Close'].shift(1)).dropna()
    realized_vol = processor.calculate_realized_volatility(log_returns, window=20)
    
    # 2. HMM 모델 학습 및 국면 예측
    # HMM 학습에 충분한 데이터가 있는지 확인
    if len(log_returns) < 252: # 최소 1년 데이터 필요
        raise ValueError("Not enough data to train HMM model. Need at least 1 year of data.")
    
    processor.train_hmm_regime_detector(log_returns, realized_vol)
    hmm_regime = processor.predict_regime(log_returns, realized_vol)
    
    # 3. 기타 기술적 지표 계산
    rsi = processor.calculate_rsi(price_data['Close'])
    adx = processor.calculate_adx(price_data['High'], price_data['Low'], price_data['Close'])
    
    # 4. 최종 신호 생성
    # 모든 지표가 포함된 DataFrame 생성
    combined_df = pd.concat([price_data, vix_data['Close'].rename('VIX_Close'), hmm_regime, rsi.rename('RSI'), adx.rename('ADX')], axis=1).dropna()

    signals_df = strategy.generate_signals(
        df=combined_df,
        kalman_smoothed_price=combined_df['Close'], # Kalman 필터 대신 원본 사용 간소화
        rsi=combined_df['RSI'],
        adx=combined_df['ADX'],
        hmm_regime=combined_df['HMM_Regime'],
        vix=combined_df['VIX_Close']
    )
    
    # 매매 포지션 결정 (SELL/STRONG_SELL 이면 0, 아니면 1)
    # 1: QQQ 보유, 0: JEPI로 전환
    final_signal = (signals_df['signal'] >= SignalType.NEUTRAL.value).astype(int)
    final_signal.name = 'Position'
    
    logger.info("Crash avoidance signals generated successfully.")
    
    # 신호 분석을 위한 상세 데이터 반환
    details = pd.concat([combined_df[['Close', 'HMM_Regime', 'RSI', 'ADX', 'VIX_Close']], signals_df, final_signal], axis=1).dropna()
    return final_signal, details

# --- 백테스트 로직 ---
def run_backtest(adj_close_data, strategy_configs, initial_capital=1000000):
    """두 가지 전략에 대한 백테스트를 실행합니다."""
    logger.info("Starting backtests for all strategies...")
    results = {}

    for name, config in strategy_configs.items():
        logger.info(f"Running backtest for {name}...")
        
        # 일일 수익률 계산
        returns = adj_close_data.pct_change().fillna(0)
        
        # 포트폴리오 수익률 시리즈 초기화
        portfolio_returns = pd.Series(index=adj_close_data.index, data=0.0)

        if config['type'] == 'static':
            # 정적 전략
            for asset, weight in config['weights'].items():
                portfolio_returns += returns[asset] * weight
        
        elif config['type'] == 'dynamic':
            # 동적 전략
            static_weights = config['static_weights']
            dynamic_asset = config['dynamic_asset']
            switch_asset = config['switch_asset']
            signal = config['signal']

            # 정적 자산군 수익률
            for asset, weight in static_weights.items():
                portfolio_returns += returns[asset] * weight
            
            # 동적 자산군 수익률
            dynamic_weight = 1.0 - sum(static_weights.values())
            
            # 신호에 따라 동적 자산의 수익률을 선택 (신호는 하루 전 기준)
            dynamic_returns_asset = returns[dynamic_asset] * dynamic_weight
            dynamic_returns_switch = returns[switch_asset] * dynamic_weight
            
            # signal을 기준으로 동적 수익률 결정
            # signal이 1이면 dynamic_asset(QQQ) 보유, 0이면 switch_asset(JEPI) 보유
            portfolio_returns += np.where(signal.shift(1) == 1, dynamic_returns_asset, dynamic_returns_switch)

        # 누적 수익률 및 포트폴리오 가치 계산
        cumulative_returns = (1 + portfolio_returns.fillna(0)).cumprod()
        results[name] = initial_capital * cumulative_returns

    logger.info("All backtests complete.")
    return pd.DataFrame(results)

# --- 성과 지표 계산 ---
def calculate_performance_metrics(portfolio_values):
    """성과 지표(CAGR, MDD 등)를 계산하고 출력합니다."""
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
            "최종 자산": f"{values.iloc[-1]:,.0f} 원",
            "연평균 수익률 (CAGR)": f"{cagr:.2%}",
            "연간 변동성": f"{volatility:.2%}",
            "샤프 지수": f"{sharpe_ratio:.2f}",
            "최대 낙폭 (MDD)": f"{mdd:.2%}"
        }
    
    metrics_df = pd.DataFrame(metrics)
    logger.info("\n--- 포트폴리오 성과 ---\n" + metrics_df.to_string())
    return metrics_df

# --- 차트 생성 ---
def plot_results(portfolio_values, signal_details, signal_ticker='QQQ'):
    """결과 비교 및 신호 상세 차트를 생성합니다."""
    logger.info("Generating plots...")
    setup_korean_font()
    
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    
    # 1. 포트폴리오 누적 수익률 비교
    ax1 = fig.add_subplot(gs[0])
    portfolio_values.plot(ax=ax1)
    ax1.set_title('포트폴리오 전략별 누적 수익률 비교 (배당 재투자 가정)', fontsize=16)
    ax1.set_ylabel('포트폴리오 가치 (원)')
    ax1.legend(['포트폴리오 1 (정적)', '포트폴리오 2 (동적)'])
    ax1.grid(True)
    
    # 2. QQQ 매매 신호 차트
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    signal_details['Close'].plot(ax=ax2, label=f'{signal_ticker} 종가', color='black', alpha=0.8)
    
    # 매매 시점 계산
    position_changes = signal_details['Position'].diff()
    buy_signals = position_changes[position_changes == 1].index
    sell_signals = position_changes[position_changes == -1].index
    
    ax2.plot(buy_signals, signal_details['Close'].loc[buy_signals], '^', markersize=10, color='g', label='보유 신호 (JEPI → QQQ)')
    ax2.plot(sell_signals, signal_details['Close'].loc[sell_signals], 'v', markersize=10, color='r', label='매도 신호 (QQQ → JEPI)')
    
    ax2.set_title(f'{signal_ticker} 매매 신호 (충돌 회피 전략 기반)', fontsize=16)
    ax2.set_ylabel(f'{signal_ticker} 가격')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    # 3. HMM 시장 국면 차트
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(signal_details.index, signal_details['HMM_Regime'], label='HMM Regime', color='purple', drawstyle='steps-post')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['상승장 (0)', '조정장 (1)', '위기 (2)'])
    ax3.set_title('HMM 기반 시장 국면 분석', fontsize=16)
    ax3.set_ylabel('시장 국면')
    ax3.set_xlabel('날짜')
    ax3.legend(loc='upper left')
    ax3.grid(axis='y')

    plt.tight_layout()
    plt.savefig('portfolio_strategy_comparison.png')
    logger.info("Plots saved to 'portfolio_strategy_comparison.png'")
    plt.show()

# --- 메인 실행 함수 ---
def main():
    """메인 분석 실행 함수"""
    # --- 설정 ---
    tickers = ['SCHD', 'QQQ', 'GLD', 'JEPI', '^KS11', '^VIX']
    ohlcv_tickers = ['QQQ', '^VIX'] # 신호 생성에 OHLCV 데이터가 필요한 티커
    start_date = '2020-05-21' # JEPI 상장일 기준
    end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    initial_capital = 100_000_000 # 초기 자본 1억원

    try:
        # --- 데이터 로드 ---
        adj_close, ohlcv_data = get_data(tickers, ohlcv_tickers, start_date, end_date)
        
        # --- 동적 신호 생성 ---
        dynamic_signal, signal_details = generate_crash_avoidance_signal(ohlcv_data, signal_ticker='QQQ', vix_ticker='^VIX')
        
        # --- 전략 정의 ---
        strategy_configs = {
            'Portfolio 1 (Static)': {
                'type': 'static',
                'weights': {'SCHD': 0.50, 'QQQ': 0.45, 'GLD': 0.05}
            },
            'Portfolio 2 (Dynamic)': {
                'type': 'dynamic',
                'static_weights': {'SCHD': 0.38, 'GLD': 0.05, '^KS11': 0.19},
                'dynamic_asset': 'QQQ',
                'switch_asset': 'JEPI',
                'signal': dynamic_signal
            }
        }
        
        # --- 백테스트 실행 ---
        # 신호가 생성된 기간으로 데이터 필터링
        valid_start_date = dynamic_signal.index.min()
        adj_close_filtered = adj_close[adj_close.index >= valid_start_date]
        
        portfolio_values = run_backtest(adj_close_filtered, strategy_configs, initial_capital)
        
        # --- 결과 분석 및 시각화 ---
        calculate_performance_metrics(portfolio_values)
        plot_results(portfolio_values, signal_details, signal_ticker='QQQ')

    except Exception as e:
        logger.error(f"An error occurred during the analysis: {e}", exc_info=True)

if __name__ == '__main__':
    main()
