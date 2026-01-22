"""
Main Pipeline - Integration and Backtesting Module
파일 목적: 모든 컴포넌트 통합, 백테스팅 실행, 결과 시각화
주요 기능: 엔드-투-엔드 파이프라인 실행 및 검증
"""

import logging
import sys
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
# matplotlib and seaborn will be imported lazily in visualization methods

# Custom modules
sys.path.insert(0, str(Path(__file__).parent))
from data_fetcher import DataFetcher
from signal_processor import SignalProcessor
from strategy import Strategy, SignalType

# Backtesting
try:
    import vectorbt as vbt
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("vectorbt not available, using fallback backtesting")
    vbt = None

# Setup logging
import sys
from pathlib import Path

# Ensure logs directory exists
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_dir / 'crash_detection.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class CrashDetectionPipeline:
    """
    통합 충돌회피 감지 파이프라인
    
    Responsibility:
    - 전체 워크플로우 조율 (Data → Signal → Backtest)
    - 결과 검증 및 리포팅
    - 멱등성 확보 (재실행 안전성)
    """
    
    def __init__(
        self,
        ticker: str = 'SPY',
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None,
        cache_dir: str = './data'
    ):
        """
        Initialize the pipeline.
        
        Args:
            ticker: Stock ticker to analyze
            start_date: Historical data start date
            end_date: Historical data end date (default: today)
            cache_dir: Data cache directory
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.cache_dir = cache_dir
        
        # Components
        self.data_fetcher = DataFetcher(output_dir=cache_dir)
        self.signal_processor = SignalProcessor()
        self.strategy = Strategy()
        
        # Results storage
        self.data: Optional[pd.DataFrame] = None
        self.indicators: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        self.backtest_results: Optional[Dict] = None
        
        logger.info(f"CrashDetectionPipeline initialized: {ticker} ({start_date} to {self.end_date})")
    
    def fetch_and_prepare_data(self) -> pd.DataFrame:
        """
        단계 1: 데이터 수집 및 전처리
        
        Responsibility:
        - 모든 데이터 소스에서 데이터 수집
        - 데이터 무결성 검증 (Full Context)
        - 기본 지표 계산 (Returns, Volatility)
        
        Returns:
            Comprehensive dataset
        """
        logger.info("="*60)
        logger.info("STAGE 1: Data Fetching and Preparation")
        logger.info("="*60)
        
        try:
            # 데이터 수집
            self.data = self.data_fetcher.create_comprehensive_dataset(
                ticker=self.ticker,
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            logger.info(f"Data shape: {self.data.shape}")
            logger.info(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            
            # 기본 지표 계산
            self.data['Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
            self.data['Realized_Volatility'] = self.signal_processor.calculate_realized_volatility(
                self.data['Returns'],
                window=20
            )
            
            logger.info(f"Basic indicators computed")
            logger.info(f"Returns mean: {self.data['Returns'].mean():.6f}, std: {self.data['Returns'].std():.6f}")
            logger.info(f"Volatility mean: {self.data['Realized_Volatility'].mean():.6f}")
            
            # 데이터 검증: NaN 확인
            nan_count = self.data.isnull().sum().sum()
            logger.info(f"NaN count after preparation: {nan_count}")
            
            if nan_count > 0:
                logger.warning(f"Remaining NaN values: {nan_count}")
                self.data = self.data.ffill().bfill()
                logger.info(f"NaN values after filling: {self.data.isnull().sum().sum()}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise
    
    def process_signals(self) -> pd.DataFrame:
        """
        단계 2: 신호 처리 및 생성
        
        Responsibility:
        - Kalman smoothing 적용
        - HMM 학습 및 레짐 예측
        - 기술 지표 계산 (RSI, ADX)
        - 최종 신호 생성
        
        Returns:
            Signals dataframe
        """
        logger.info("="*60)
        logger.info("STAGE 2: Signal Processing")
        logger.info("="*60)
        
        try:
            if self.data is None:
                raise ValueError("Data not fetched. Call fetch_and_prepare_data first.")
            
            # 1. Kalman Filtering (Noise reduction)
            logger.info("Applying Kalman Filter...")
            kalman_smoothed = self.signal_processor.apply_kalman_filter(
                self.data['Close'],
                process_variance=1e-4,
                measurement_variance=1e-1
            )
            
            # 2. HMM Training and Prediction
            logger.info("Training HMM for regime detection...")
            self.signal_processor.train_hmm_regime_detector(
                self.data['Returns'].fillna(0),
                self.data['Realized_Volatility'].fillna(self.data['Realized_Volatility'].mean()),
                n_states=3,
                n_iter=1000
            )
            
            logger.info("Predicting regimes...")
            hmm_regime = self.signal_processor.predict_regime(
                self.data['Returns'].fillna(0),
                self.data['Realized_Volatility'].fillna(self.data['Realized_Volatility'].mean())
            )
            
            # 3. Technical Indicators
            logger.info("Calculating technical indicators...")
            rsi = self.signal_processor.calculate_rsi(
                self.data['Close'],
                period=14,
                smoothing=kalman_smoothed
            )
            
            adx = self.signal_processor.calculate_adx(
                self.data['High'],
                self.data['Low'],
                self.data['Close'],
                period=14
            )
            
            # 4. Signal Generation
            logger.info("Generating hybrid signals...")
            self.signals = self.strategy.generate_signals(
                df=self.data,
                kalman_smoothed_price=kalman_smoothed,
                rsi=rsi,
                adx=adx,
                hmm_regime=hmm_regime,
                vix=self.data.get('VIX', pd.Series(np.nan, index=self.data.index)),
                vvix=self.data.get('VVIX', None)
            )
            
            # Indicator storage
            self.indicators = pd.DataFrame({
                'Kalman_Price': kalman_smoothed,
                'RSI': rsi,
                'ADX': adx,
                'HMM_Regime': hmm_regime,
                'VIX': self.data.get('VIX', np.nan),
                'VVIX': self.data.get('VVIX', np.nan)
            }, index=self.data.index)
            
            logger.info(f"Signal generation complete:")
            logger.info(f"Signal distribution:\n{self.signals['signal'].value_counts().sort_index()}")
            
            return self.signals
            
        except Exception as e:
            logger.error(f"Error in signal processing: {e}")
            raise
    
    def backtest(self, initial_capital: float = 100000.0) -> Dict:
        """
        단계 3: 백테스트 및 성과 평가
        
        Responsibility:
        - 신호 기반 포트폴리오 구성
        - 수익률 및 드로우다운 계산
        - 성과 지표 산출 (Sharpe, Sortino, etc.)
        
        Args:
            initial_capital: 초기 자본금
        
        Returns:
            Backtest results dictionary
        """
        logger.info("="*60)
        logger.info("STAGE 3: Backtesting")
        logger.info("="*60)
        
        try:
            if self.signals is None:
                raise ValueError("Signals not generated. Call process_signals first.")
            
            # 포지션 결정: 신호 기반
            # SELL/STRONG_SELL 신호 → 0% 포지션 (현금)
            # NEUTRAL → 100% 포지션 (SPY)
            positions = pd.Series(1.0, index=self.signals.index)
            positions[self.signals['signal'] < 0] = 0.0  # SELL signals
            
            # Position smoothing (급격한 변경 방지)
            positions = positions.ffill()
            
            # 수익률 계산
            returns = self.data['Close'].pct_change()
            strategy_returns = returns * positions.shift(1)
            
            # 누적 수익률
            cumulative_returns = (1 + strategy_returns).cumprod()
            benchmark_returns = (1 + returns).cumprod()
            
            # 성과 지표 계산
            total_return = (cumulative_returns.iloc[-1] - 1) * 100
            benchmark_return = (benchmark_returns.iloc[-1] - 1) * 100
            
            # CAGR (연평균 성장률)
            n_years = (self.data.index[-1] - self.data.index[0]).days / 365.25
            cagr = ((cumulative_returns.iloc[-1] ** (1 / n_years)) - 1) * 100
            benchmark_cagr = ((benchmark_returns.iloc[-1] ** (1 / n_years)) - 1) * 100
            
            # Max Drawdown
            cummax = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - cummax) / cummax
            max_dd = drawdown.min() * 100
            
            # Sharpe Ratio
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() > 0 else 0
            
            # Sortino Ratio (downside deviation only)
            downside_returns = strategy_returns[strategy_returns < 0]
            sortino = (strategy_returns.mean() / downside_returns.std()) * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Win rate
            win_rate = (strategy_returns > 0).sum() / len(strategy_returns) * 100
            
            self.backtest_results = {
                'total_return': total_return,
                'benchmark_return': benchmark_return,
                'cagr': cagr,
                'benchmark_cagr': benchmark_cagr,
                'max_drawdown': max_dd,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'win_rate': win_rate,
                'cumulative_returns': cumulative_returns,
                'benchmark_returns': benchmark_returns,
                'strategy_returns': strategy_returns
            }
            
            # 로그 기록
            logger.info(f"Backtest Results:")
            logger.info(f"  Total Return: {total_return:.2f}% (Benchmark: {benchmark_return:.2f}%)")
            logger.info(f"  CAGR: {cagr:.2f}% (Benchmark: {benchmark_cagr:.2f}%)")
            logger.info(f"  Max Drawdown: {max_dd:.2f}%")
            logger.info(f"  Sharpe Ratio: {sharpe:.3f}")
            logger.info(f"  Sortino Ratio: {sortino:.3f}")
            logger.info(f"  Win Rate: {win_rate:.2f}%")
            
            # 검증: Sortino > 2.0, MDD < 50%
            if sortino > 2.0:
                logger.info("[PASS] Sortino Ratio target (>2.0) ACHIEVED")
            else:
                logger.warning(f"[WARN] Sortino Ratio target not met (current: {sortino:.3f})")
            
            if abs(max_dd) < 50:
                logger.info("[PASS] Max Drawdown target (<50%) ACHIEVED")
            else:
                logger.warning(f"[WARN] Max Drawdown target not met (current: {abs(max_dd):.2f}%)")
            
            return self.backtest_results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            raise
    
    def visualize_results(self, output_dir: str = 'results') -> None:
        """
        Visualize results: cumulative returns, signals, indicators
        
        Args:
            output_dir: Output directory for plots
        """
        logger.info("="*60)
        logger.info("STAGE 4: Visualization")
        logger.info("="*60)
        
        try:
            # Lazy import visualization libraries
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Figure 1: Cumulative Returns
            fig, ax = plt.subplots(figsize=(14, 6))
            
            self.backtest_results['cumulative_returns'].plot(
                ax=ax, label='Strategy', color='blue', linewidth=2.5, linestyle='-'
            )
            self.backtest_results['benchmark_returns'].plot(
                ax=ax, label='Benchmark (Buy&Hold)', color='red', linewidth=2.5, 
                linestyle='--', alpha=0.9
            )
            
            ax.set_title(f'{self.ticker} - Crash Detection Strategy vs Benchmark', fontsize=14, fontweight='bold')
            ax.set_ylabel('Cumulative Return (%)', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.legend(fontsize=11, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/01_cumulative_returns.png', dpi=150)
            logger.info(f"Saved: {output_dir}/01_cumulative_returns.png")
            plt.close()
            
            # Figure 2: Signals & Price
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            
            # Price with signals
            ax1.plot(self.data.index, self.data['Close'], label='Close', color='black', linewidth=1)
            
            # Mark sell signals
            sell_signals = self.signals[self.signals['signal'] < 0]
            ax1.scatter(
                sell_signals.index,
                self.data.loc[sell_signals.index, 'Close'],
                color='red', s=100, marker='v', label='SELL Signal', zorder=5
            )
            
            ax1.set_title(f'{self.ticker} Price with Signals', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Price', fontsize=10)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Indicators
            ax2.plot(self.indicators.index, self.indicators['RSI'], label='RSI', color='green', linewidth=1.5)
            ax2.axhline(y=40, color='red', linestyle='--', linewidth=1, alpha=0.7, label='RSI=40')
            ax2.axhline(y=70, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='RSI=70')
            ax2.set_title('RSI Indicator', fontsize=12, fontweight='bold')
            ax2.set_ylabel('RSI', fontsize=10)
            ax2.set_xlabel('Date', fontsize=10)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 100])
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/02_signals_and_indicators.png', dpi=150)
            logger.info(f"Saved: {output_dir}/02_signals_and_indicators.png")
            plt.close()
            
            # Figure 3: HMM Regimes
            fig, ax = plt.subplots(figsize=(14, 6))
            
            regime_colors = {0: 'green', 1: 'yellow', 2: 'red'}
            regime_labels = {0: 'Bull', 1: 'Correction', 2: 'Crisis'}
            
            for regime_id in [0, 1, 2]:
                regime_mask = self.indicators['HMM_Regime'] == regime_id
                ax.scatter(
                    self.indicators.index[regime_mask],
                    self.data.loc[regime_mask, 'Close'],
                    color=regime_colors[regime_id],
                    alpha=0.5,
                    s=20,
                    label=regime_labels[regime_id]
                )
            
            ax.set_title('HMM Regime Detection', fontsize=12, fontweight='bold')
            ax.set_ylabel('Close Price', fontsize=10)
            ax.set_xlabel('Date', fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/03_hmm_regimes.png', dpi=150)
            logger.info(f"Saved: {output_dir}/03_hmm_regimes.png")
            plt.close()
            
            logger.info("All visualizations saved")
            
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
    
    def generate_report(self, output_file: str = 'results/backtest_report.json') -> None:
        """
        Generate comprehensive report.
        
        Args:
            output_file: Output JSON file path
        """
        logger.info("="*60)
        logger.info("STAGE 5: Report Generation")
        logger.info("="*60)
        
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'ticker': self.ticker,
                'period': {
                    'start': self.start_date,
                    'end': self.end_date
                },
                'backtest_results': {
                    k: float(v) if isinstance(v, (np.floating, float)) else str(v)
                    for k, v in self.backtest_results.items()
                    if k not in ['cumulative_returns', 'benchmark_returns', 'strategy_returns']
                },
                'data_summary': {
                    'total_rows': len(self.data),
                    'date_range': f"{self.data.index[0]} to {self.data.index[-1]}",
                    'close_price_range': f"{self.data['Close'].min():.2f} ~ {self.data['Close'].max():.2f}"
                },
                'signal_summary': {
                    'total_signals': len(self.signals),
                    'strong_sell': int((self.signals['signal'] == SignalType.STRONG_SELL.value).sum()),
                    'sell': int((self.signals['signal'] == SignalType.SELL.value).sum()),
                    'neutral': int((self.signals['signal'] == SignalType.NEUTRAL.value).sum()),
                    'buy': int((self.signals['signal'] == SignalType.BUY.value).sum()),
                    'strong_buy': int((self.signals['signal'] == SignalType.STRONG_BUY.value).sum()),
                    'signal_explanation': {
                        'neutral_reason': 'All signals suppressed by ADX filter (weak trend) OR no sell conditions met',
                        'adx_threshold': '< 20 = Weak Trend (suppress signals)',
                        'crisis_threshold': 'HMM State = 2 AND RSI < 45',
                        'normal_threshold': 'Condition A (RSI spike down) AND Condition C (VIX > 30)'
                    }
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Report saved to {output_file}")
            print("\n" + "="*60)
            print("BACKTEST REPORT")
            print("="*60)
            print(json.dumps(report, indent=2))
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    def run_full_pipeline(self) -> Dict:
        """
        실행: 전체 파이프라인 통합
        
        Process:
        1. Data fetch & prepare
        2. Signal processing (Kalman, HMM, indicators)
        3. Backtest
        4. Visualize
        5. Report
        
        Returns:
            Complete results dictionary
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING CRASH DETECTION PIPELINE")
        logger.info("="*70)
        
        try:
            # Stage 1
            self.fetch_and_prepare_data()
            
            # Stage 2
            self.process_signals()
            
            # Stage 3
            self.backtest()
            
            # Stage 4
            self.visualize_results()
            
            # Stage 5
            self.generate_report()
            
            logger.info("\n" + "="*70)
            logger.info("PIPELINE EXECUTION COMPLETE [OK]")
            logger.info("="*70)
            
            return {
                'status': 'SUCCESS',
                'data': self.data,
                'indicators': self.indicators,
                'signals': self.signals,
                'backtest_results': self.backtest_results
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise


if __name__ == "__main__":
    # 메인 실행
    pipeline = CrashDetectionPipeline(
        ticker='SPY',
        start_date='2015-01-01',
        end_date=None  # Use current date
    )
    
    results = pipeline.run_full_pipeline()
