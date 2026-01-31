# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict


class AlphaLib:
    """
    Vectorized Technical Indicator Library.
    Optimized for calculating indicators on large DataFrames (Tickers x Date).
    """

    @staticmethod
    def sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
        return df.rolling(window=window).mean()

    @staticmethod
    def ema(df: pd.DataFrame, window: int) -> pd.DataFrame:
        return df.ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(close: pd.DataFrame, window: int = 20, num_std: float = 2.0):
        rolling_mean = close.rolling(window=window).mean()
        rolling_std = close.rolling(window=window).std()
        upper = rolling_mean + (rolling_std * num_std)
        lower = rolling_mean - (rolling_std * num_std)
        return upper, lower

    @staticmethod
    def atr(
        high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14
    ) -> pd.DataFrame:
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        ranges = pd.concat(
            [high_low, high_close, low_close], axis=1
        )  # This structure is tricky for 3D/Multiple Tickers
        # For multi-column DF (tickers), we must do element-wise max

        tr = pd.DataFrame(index=close.index, columns=close.columns)
        # Vectorized True Range
        tr = np.maximum(high - low, np.abs(high - close.shift()))
        tr = np.maximum(tr, np.abs(low - close.shift()))

        return (
            pd.DataFrame(tr, index=close.index, columns=close.columns)
            .rolling(window=window)
            .mean()
        )

    @staticmethod
    def donchian(high: pd.DataFrame, low: pd.DataFrame, window: int = 20):
        upper = high.rolling(window=window).max()
        lower = low.rolling(window=window).min()
        return upper, lower
