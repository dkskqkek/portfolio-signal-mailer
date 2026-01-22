"""
Strategy Module - Hybrid signal generation combining multiple factors
파일 목적: Condition A/B/C 로직을 통한 신호 생성 및 포지션 관리
주요 기능: 충돌회피 기반 (Crash Detection) 신호 생성
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """신호 타입 정의"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class Strategy:
    """
    하이브리드 충돌회피 전략 (Hybrid Crash Avoidance Strategy)
    
    Responsibility:
    - 멀티팩터 신호 생성 (Breadth, Regime, Volatility)
    - Condition A/B/C 평가
    - 최종 포지션 결정
    
    설계 철학 (Design Philosophy):
    - Precision > Recall: False Positive 최소화
    - 리스크 관리 우선
    - 명확한 진입/탈출 로직
    """
    
    def __init__(self):
        """Initialize Strategy with parameters."""
        # Signal parameters
        self.rsi_crash_threshold = 40      # RSI momentum crash threshold
        self.rsi_overbought = 70           # RSI overbought threshold
        self.vix_high_threshold = 30       # VIX 경보 수준
        self.adx_strong_trend = 25         # ADX 강한 추세 기준
        self.adx_weak_trend = 20           # ADX 약한 추세 기준
        
        logger.info("Strategy initialized with default parameters")
    
    def _evaluate_condition_a(
        self,
        rsi: pd.Series,
        rsi_prev: pd.Series
    ) -> pd.Series:
        """
        Condition A: Breadth/Tech momentum crash
        
        Logic:
        - RSI(smoothed) < 40 AND RSI previously was > 70
        - Indicates momentum reversal from overbought to oversold
        
        Returns:
            Boolean Series (True = Condition A met)
        
        설명:
        - 기술주의 모멘텀이 극도의 과매수에서 급락하는 상황 포착
        - 나스닥/테크 섹터의 급격한 조정 신호
        """
        try:
            condition_a = (rsi < self.rsi_crash_threshold) & (rsi_prev > self.rsi_overbought)
            
            return condition_a
            
        except Exception as e:
            logger.error(f"Error in Condition A: {e}")
            return pd.Series(False, index=rsi.index)
    
    def _evaluate_condition_b(
        self,
        hmm_regime: pd.Series
    ) -> pd.Series:
        """
        Condition B: HMM Regime = Crisis
        
        Logic:
        - HMM Current State == 2 (Crisis)
        - Indicates market in heightened stress/volatility state
        
        Returns:
            Boolean Series (True = Condition B met)
        
        설명:
        - 머신러닝 기반 시장 상태 인식
        - 위기 상태로 전환된 것을 감지하면 높은 민감도로 신호
        """
        try:
            condition_b = (hmm_regime == 2)
            
            return condition_b
            
        except Exception as e:
            logger.error(f"Error in Condition B: {e}")
            return pd.Series(False, index=hmm_regime.index)
    
    def _evaluate_condition_c(
        self,
        vix: pd.Series,
        vix_term_structure_inverted: pd.Series
    ) -> pd.Series:
        """
        Condition C: Volatility spike or inversion
        
        Logic:
        - VIX Term Structure Inverted (Backwardation) OR VIX > 30
        - Indicates near-term volatility expectations exceed longer-term
        
        Returns:
            Boolean Series (True = Condition C met)
        
        설명:
        - VIX가 높거나 (공포 상태)
        - 변동성 기간구조가 역전 (근시적 불안정)
        - 둘 다 시장 스트레스의 강한 신호
        """
        try:
            condition_c = (
                (vix > self.vix_high_threshold) |
                vix_term_structure_inverted
            )
            
            return condition_c
            
        except Exception as e:
            logger.error(f"Error in Condition C: {e}")
            return pd.Series(False, index=vix.index)
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        kalman_smoothed_price: pd.Series,
        rsi: pd.Series,
        adx: pd.Series,
        hmm_regime: pd.Series,
        vix: pd.Series,
        vvix: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate signals using hybrid logic.
        
        Signal Logic:
        1. IF HMM State == 2 (Crisis):
               SELL if RSI < 45 (High Sensitivity)
        
        2. IF HMM State != 2 (Normal/Correction):
               SELL only if (Condition A AND Condition C) (High Specificity)
        
        3. IF ADX < 20:
               Suppress all SELL signals (Noise filtering)
        
        Args:
            df: Base dataframe with OHLCV
            kalman_smoothed_price: Kalman-filtered close price
            rsi: RSI indicator
            adx: ADX indicator
            hmm_regime: HMM regime predictions (0=Bull, 1=Correction, 2=Crisis)
            vix: VIX level
            vvix: Optional VVIX for term structure
        
        Returns:
            DataFrame with columns:
            - signal: {-2: STRONG_SELL, -1: SELL, 0: NEUTRAL, 1: BUY, 2: STRONG_BUY}
            - signal_strength: confidence score (0-1)
            - signal_reason: human-readable reason
        
        Decision Record:
        - ADX < 20일 때는 노이즈가 많으므로 신호 억제
        - 위기 상태에서는 더 높은 민감도 (RSI < 45)
        - 정상 상태에서는 더 높은 특이성 (Condition A AND C 모두)
        """
        logger.info("Generating hybrid signals")
        
        try:
            # 전날 RSI 값 (momentum 역전 감지용)
            rsi_prev = rsi.shift(1)
            
            # VIX 기간구조 (VVIX 또는 대체값)
            if vvix is not None and not vvix.isna().all():
                vix_term_inverted = vvix > vix  # VIX가 VVIX보다 높으면 역전
            else:
                vix_term_inverted = pd.Series(False, index=df.index)
                logger.warning("VVIX not available, using fallback for term structure")
            
            # Conditions 평가
            condition_a = self._evaluate_condition_a(rsi, rsi_prev)
            condition_b = self._evaluate_condition_b(hmm_regime)
            condition_c = self._evaluate_condition_c(vix, vix_term_inverted)
            
            # 신호 생성 (초기화)
            signals = pd.Series(SignalType.NEUTRAL.value, index=df.index, name='signal')
            signal_strength = pd.Series(0.0, index=df.index, name='signal_strength')
            signal_reason = pd.Series('', index=df.index, name='signal_reason')
            
            # ADX 필터 (추세 강도)
            adx_valid = (adx > self.adx_weak_trend)
            
            # 신호 로직
            for idx in df.index:
                if pd.isna(hmm_regime.loc[idx]) or pd.isna(rsi.loc[idx]) or pd.isna(adx.loc[idx]):
                    continue
                
                current_regime = int(hmm_regime.loc[idx])
                current_rsi = rsi.loc[idx]
                current_adx = adx.loc[idx]
                current_vix = vix.loc[idx] if not pd.isna(vix.loc[idx]) else 0
                
                # CASE 1: Crisis Regime (State == 2)
                if current_regime == 2:
                    if current_rsi < 45:
                        signals.loc[idx] = SignalType.STRONG_SELL.value
                        signal_strength.loc[idx] = 0.95
                        signal_reason.loc[idx] = f"STRONG_SELL: Crisis regime (RSI={current_rsi:.1f})"
                    else:
                        signals.loc[idx] = SignalType.SELL.value
                        signal_strength.loc[idx] = 0.70
                        signal_reason.loc[idx] = f"SELL: Crisis regime (RSI={current_rsi:.1f})"
                
                # CASE 2: Normal/Correction Regime (State != 2)
                else:
                    if condition_a.loc[idx] and condition_c.loc[idx]:
                        # High specificity: both conditions must be true
                        signals.loc[idx] = SignalType.SELL.value
                        signal_strength.loc[idx] = 0.75
                        signal_reason.loc[idx] = f"SELL: Condition A+C met (RSI={current_rsi:.1f}, VIX={current_vix:.1f})"
                    elif condition_b.loc[idx] and current_adx > self.adx_strong_trend:
                        # Secondary signal: regime shift with strong trend
                        signals.loc[idx] = SignalType.SELL.value
                        signal_strength.loc[idx] = 0.60
                        signal_reason.loc[idx] = f"SELL: Correction regime detected (ADX={current_adx:.1f})"
                
                # ADX Filter: Suppress if weak trend
                if current_adx < self.adx_weak_trend:
                    if signals.loc[idx] != SignalType.NEUTRAL.value:
                        logger.debug(f"Suppressing signal at {idx} due to weak ADX={current_adx:.1f}")
                        signals.loc[idx] = SignalType.NEUTRAL.value
                        signal_strength.loc[idx] = 0.0
                        signal_reason.loc[idx] = f"SUPPRESSED: Weak trend (ADX={current_adx:.1f})"
            
            # 결과 통합
            result = pd.DataFrame({
                'signal': signals,
                'signal_strength': signal_strength,
                'signal_reason': signal_reason
            })
            
            # 로깅
            sell_count = (result['signal'] == SignalType.SELL.value).sum()
            strong_sell_count = (result['signal'] == SignalType.STRONG_SELL.value).sum()
            logger.info(f"Signals generated: {strong_sell_count} STRONG_SELL, {sell_count} SELL")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            raise
    
    def calculate_position_sizing(
        self,
        signal: float,
        signal_strength: float,
        current_volatility: float,
        max_position_size: float = 1.0
    ) -> float:
        """
        Calculate position size based on signal strength and volatility.
        
        Args:
            signal: Signal value (-2 to 2)
            signal_strength: Confidence score (0-1)
            current_volatility: Current realized volatility
            max_position_size: Maximum position (0-1)
        
        Returns:
            Position sizing multiplier (0-1)
        
        결정 근거:
        - 신호 강도와 변동성에 따라 포지션 크기 조정
        - 변동성이 높을 때는 더 작은 포지션
        - 신호 강도가 약할 때도 더 작은 포지션
        """
        if signal == SignalType.NEUTRAL.value:
            return 0.0
        
        # 변동성 정규화 (연간화된 변동성, 기준: 20%)
        volatility_adjustment = 1.0 - (current_volatility / 0.40)  # Cap at 40%
        volatility_adjustment = max(0.3, min(1.0, volatility_adjustment))
        
        # 포지션 크기 = 신호 강도 * 변동성 조정
        position_size = signal_strength * volatility_adjustment * max_position_size
        
        return max(0.0, min(1.0, position_size))


if __name__ == "__main__":
    # Test Strategy
    logging.basicConfig(level=logging.INFO)
    
    strategy = Strategy()
    
    # 테스트 데이터
    n = 252 * 2  # 2 years
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    
    df_test = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(n) * 0.3),
        'High': 102 + np.cumsum(np.random.randn(n) * 0.3),
        'Low': 98 + np.cumsum(np.random.randn(n) * 0.3),
        'Close': 100 + np.cumsum(np.random.randn(n) * 0.3),
        'Volume': np.random.randint(1000000, 5000000, n)
    }, index=dates)
    
    # 합성 지표
    rsi_test = pd.Series(50 + np.random.randn(n) * 15, index=dates).clip(0, 100)
    adx_test = pd.Series(25 + np.random.randn(n) * 5, index=dates).clip(0, 100)
    hmm_regime_test = pd.Series(np.random.choice([0, 1, 2], n), index=dates)
    vix_test = pd.Series(15 + np.random.randn(n) * 5, index=dates).clip(0, 100)
    vvix_test = pd.Series(20 + np.random.randn(n) * 5, index=dates).clip(0, 100)
    
    # 신호 생성
    signals = strategy.generate_signals(
        df_test,
        df_test['Close'],
        rsi_test,
        adx_test,
        hmm_regime_test,
        vix_test,
        vvix_test
    )
    
    print(f"Signal distribution:\n{signals['signal'].value_counts().sort_index()}")
    print(f"\nSample signals:\n{signals.head(10)}")
