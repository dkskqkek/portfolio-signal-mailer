"""
Signal Processor Module - Implements Kalman Filter and HMM for regime detection
파일 목적: 노이즈 필터링 (Kalman), 상태 감지 (HMM), ADX 기반 트렌드 필터
주요 기능: 깨끗한 신호 추출 및 현재 시장 레짐 판정
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional
# pykalman and hmmlearn will be imported lazily in methods that use them
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    신호 처리 및 노이즈 필터링을 담당하는 클래스
    
    Responsibility:
    - Kalman Filter를 통한 가격 데이터 노이즈 제거
    - GaussianHMM을 통한 시장 레짐 감지 (Bull, Correction, Crisis)
    - ADX 기반 트렌드 강도 필터링
    - RSI, VIX 등 기술 지표 계산
    """
    
    def __init__(self):
        """Initialize SignalProcessor with default parameters."""
        self.kalman_filter: Optional[any] = None  # Will be KalmanFilter when loaded
        self.hmm_model: Optional[any] = None  # Will be GaussianHMM when loaded
        self.is_trained = False
        logger.info("SignalProcessor initialized")
    
    def apply_kalman_filter(
        self,
        price_series: pd.Series,
        process_variance: float = 1e-4,
        measurement_variance: float = 1e-1
    ) -> pd.Series:
        """
        Apply Kalman Filter to smooth price data and reduce microstructure noise.
        
        수학적 배경 (Mathematical Background):
        상태 방정식: x_t = F_t * x_{t-1} + w_t
        관측 방정식: z_t = H_t * x_t + v_t
        
        여기서:
        - x_t: 실제 가격 (true underlying price)
        - z_t: 관측된 가격 (observed market price with noise)
        - w_t: 프로세스 노이즈 ~ N(0, Q)
        - v_t: 관측 노이즈 ~ N(0, R)
        
        Args:
            price_series: 원본 가격 시리즈
            process_variance: Q (프로세스 노이즈 분산)
            measurement_variance: R (관측 노이즈 분산)
        
        Returns:
            smoothed_prices: Kalman smoothing을 거친 가격
        
        Decision Record (선택 이유):
        - 1차 모델 (constant velocity): 단순하면서도 효과적
        - 큰 Q (process variance)는 원본 가격을 더 따라감
        - 큰 R (measurement variance)는 더 강하게 평활화
        """
        try:
            # Lazy import
            from pykalman import KalmanFilter
            
            logger.info(f"Applying Kalman Filter with Q={process_variance}, R={measurement_variance}")
            
            # 1차 모델: x_t = x_{t-1} (상수 속도 모델)
            # State: [price], Transition: [1], Measurement: [1]
            # pykalman API: transition_covariance (Q), observation_covariance (R)
            kf = KalmanFilter(
                transition_matrices=[[1.0]],
                observation_matrices=[[1.0]],
                initial_state_mean=[price_series.iloc[0]],
                initial_state_covariance=[[1.0]],
                transition_covariance=[[process_variance]],
                observation_covariance=[[measurement_variance]]
            )
            
            # Smoothing 수행 (forward-backward pass)
            smoothed_state, _ = kf.smooth(price_series.values)
            
            smoothed_prices = pd.Series(
                smoothed_state.flatten(),
                index=price_series.index,
                name='Kalman_Smoothed_Price'
            )
            
            logger.info(f"Kalman smoothing complete. Output shape: {smoothed_prices.shape}")
            
            return smoothed_prices
            
        except Exception as e:
            logger.error(f"Error in Kalman Filter: {e}")
            # Fallback: return original series
            logger.warning("Falling back to original price series due to Kalman error")
            return price_series
    
    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Average Directional Index (ADX) to measure trend strength.
        
        ADX < 20: Weak trend (Noise) - Suppress signals
        ADX > 25: Strong trend - Accept trend signals
        
        Args:
            high, low, close: OHLC data
            period: ADX period (default 14)
        
        Returns:
            ADX Series (0-100)
        
        Decision Record:
        - ADX는 추세 강도를 측정하는 표준 기술 지표
        - 값이 낮으면 노이즈가 많은 시장 (시그널 억제)
        - 값이 높으면 명확한 추세 (시그널 수용)
        """
        try:
            logger.info(f"Calculating ADX with period={period}")
            
            # True Range 계산
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # +DM, -DM 계산
            up = high - high.shift(1)
            down = low.shift(1) - low
            
            plus_dm = np.where((up > down) & (up > 0), up, 0)
            minus_dm = np.where((down > up) & (down > 0), down, 0)
            
            # ATR 계산 (더 안정적인 방법)
            atr = tr.rolling(period).mean()
            
            # +DI, -DI 계산 (분모 0 방지)
            plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(period).mean() / (atr + 1e-10)
            minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(period).mean() / (atr + 1e-10)
            
            # DX 계산 (분모 0 방지)
            di_diff = abs(plus_di - minus_di)
            di_sum = plus_di + minus_di + 1e-10
            dx = 100 * di_diff / di_sum
            
            # ADX (exponential smoothing for better stability)
            adx = dx.ewm(span=period, adjust=False).mean()
            
            # NaN 값을 평균값으로 채우기
            adx = adx.fillna(adx.mean())
            if adx.isna().all():
                adx = pd.Series(20.0, index=close.index)  # Default weak trend
            
            logger.info(f"ADX calculation complete. Shape: {adx.shape}")
            logger.info(f"ADX mean: {adx.mean():.2f}, std: {adx.std():.2f}")
            
            return adx
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            # Fallback: 기본값 (약한 추세)
            return pd.Series(20.0, index=close.index)
    
    def train_hmm_regime_detector(
        self,
        returns: pd.Series,
        realized_volatility: pd.Series,
        n_states: int = 3,
        n_iter: int = 1000
    ) -> None:
        """
        Train Gaussian HMM for regime detection.
        
        3가지 히든 상태 (Hidden States):
        - State 0: Bull (낮은 변동성, 양수 수익률)
        - State 1: Correction (중간 변동성, 혼합 수익률)
        - State 2: Crisis (높은 변동성, 음수 수익률)
        
        Args:
            returns: Daily log returns
            realized_volatility: Realized volatility (e.g., rolling std)
            n_states: Number of hidden states (default 3)
            n_iter: Number of EM iterations
        
        Decision Record:
        - GaussianHMM은 금융 레짐 전환을 잘 포착합니다
        - 2개 특징 사용: 수익률 + 변동성 (충분하고 정보력 높음)
        - 3개 상태: 이론적으로 시장의 주요 패턴 (bull, correction, crisis)를 표현
        """
        try:
            # Lazy import
            from hmmlearn.hmm import GaussianHMM
            
            logger.info(f"Training HMM with {n_states} states")
            
            # 데이터 준비 (NaN 제거)
            valid_idx = (returns.notna()) & (realized_volatility.notna())
            returns_clean = returns[valid_idx].values.reshape(-1, 1)
            vol_clean = realized_volatility[valid_idx].values.reshape(-1, 1)
            
            # 2D feature matrix
            X = np.column_stack([returns_clean, vol_clean])
            
            logger.info(f"Training data shape: {X.shape}")
            
            # HMM 모델 생성 및 학습
            self.hmm_model = GaussianHMM(
                n_components=n_states,
                covariance_type='full',
                n_iter=n_iter,
                random_state=42
            )
            
            self.hmm_model.fit(X)
            self.is_trained = True
            
            # 로그 기록: 학습된 파라미터
            logger.info(f"HMM training complete")
            logger.info(f"Transition matrix:\n{self.hmm_model.transmat_}")
            logger.info(f"Means:\n{self.hmm_model.means_}")
            
        except Exception as e:
            logger.error(f"Error training HMM: {e}")
            raise
    
    def predict_regime(
        self,
        returns: pd.Series,
        realized_volatility: pd.Series
    ) -> pd.Series:
        """
        Predict hidden states (regimes) using trained HMM.
        
        Args:
            returns: Daily log returns
            realized_volatility: Realized volatility
        
        Returns:
            Series of regime labels (0=Bull, 1=Correction, 2=Crisis)
        """
        if not self.is_trained:
            logger.error("HMM model not trained. Call train_hmm_regime_detector first.")
            raise ValueError("HMM model not trained")
        
        try:
            logger.info("Predicting regimes using HMM")
            
            # 데이터 준비
            valid_idx = (returns.notna()) & (realized_volatility.notna())
            returns_clean = returns[valid_idx].values.reshape(-1, 1)
            vol_clean = realized_volatility[valid_idx].values.reshape(-1, 1)
            
            X = np.column_stack([returns_clean, vol_clean])
            
            # 예측 (반환값은 정수 배열)
            hidden_states = self.hmm_model.predict(X)
            hidden_states = np.asarray(hidden_states, dtype=int)
            
            # 시리즈로 변환 (원본 인덱스 유지)
            # numpy int array를 pandas Series로 변환할 때 일반 int 사용
            regime = pd.Series(
                np.nan,
                index=returns.index,
                dtype='float64'  # 처음엔 float로 생성 (NaN 호환)
            )
            
            # 유효한 인덱스 위치에만 정수값 할당
            regime_values = hidden_states.astype(int)  # numpy int
            regime.loc[valid_idx] = regime_values
            
            # 마지막으로 Int64 타입으로 변환 (필요시)
            regime = regime.astype('Int64')
            regime.name = 'HMM_Regime'
            
            logger.info(f"Regime prediction complete. Unique states: {np.unique(hidden_states)}")
            logger.info(f"State distribution:\n{pd.Series(hidden_states).value_counts().sort_index()}")
            
            return regime
            
        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            raise
    
    def calculate_rsi(
        self,
        close: pd.Series,
        period: int = 14,
        smoothing: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI = 100 * (1 - 1 / (1 + RS))
        where RS = avg(gains) / avg(losses)
        
        Args:
            close: Close price series
            period: RSI period (default 14)
            smoothing: Optional pre-smoothed price series (from Kalman Filter)
        
        Returns:
            RSI Series (0-100)
        """
        try:
            price_to_use = smoothing if smoothing is not None else close
            
            logger.info(f"Calculating RSI with period={period}")
            
            # 가격 변화
            delta = price_to_use.diff()
            
            # 상승/하강 분리
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # 평균 계산 (EMA)
            avg_gains = gains.ewm(span=period, adjust=False).mean()
            avg_losses = losses.ewm(span=period, adjust=False).mean()
            
            # RS와 RSI
            rs = avg_gains / avg_losses
            rsi = 100 * (1 - 1 / (1 + rs))
            rsi = rsi.fillna(50)  # NaN to neutral
            
            logger.info(f"RSI calculation complete. Mean: {rsi.mean():.2f}, Std: {rsi.std():.2f}")
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(50, index=close.index)  # Neutral fallback
    
    def calculate_realized_volatility(
        self,
        returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate realized volatility (rolling standard deviation of returns).
        
        Args:
            returns: Log returns
            window: Rolling window (default 20 days)
        
        Returns:
            Realized volatility series
        """
        try:
            logger.info(f"Calculating realized volatility with window={window}")
            
            rv = returns.rolling(window=window).std() * np.sqrt(252)
            
            logger.info(f"Realized volatility computed. Shape: {rv.shape}")
            
            return rv
            
        except Exception as e:
            logger.error(f"Error calculating realized volatility: {e}")
            return pd.Series(np.nan, index=returns.index)


if __name__ == "__main__":
    # Test SignalProcessor
    logging.basicConfig(level=logging.INFO)
    
    processor = SignalProcessor()
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n = 252 * 5  # 5 years
    dates = pd.date_range('2019-01-01', periods=n, freq='D')
    
    # 합성 가격 데이터
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    price_series = pd.Series(prices, index=dates)
    
    # Kalman Filter 테스트
    smoothed = processor.apply_kalman_filter(price_series)
    print(f"Original price mean: {price_series.mean():.2f}, std: {price_series.std():.2f}")
    print(f"Smoothed price mean: {smoothed.mean():.2f}, std: {smoothed.std():.2f}")
    
    # RSI 테스트
    rsi = processor.calculate_rsi(price_series, smoothing=smoothed)
    print(f"\nRSI mean: {rsi.mean():.2f}")
    
    # 수익률 및 변동성
    returns = np.log(price_series / price_series.shift(1))
    realized_vol = processor.calculate_realized_volatility(returns)
    
    # HMM 학습 및 예측
    processor.train_hmm_regime_detector(returns, realized_vol)
    regimes = processor.predict_regime(returns, realized_vol)
    print(f"\nRegime prediction sample:\n{regimes.tail()}")
