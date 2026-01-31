# -*- coding: utf-8 -*-
"""
Index Sniper V8.2 - Python Implementation
==========================================
Converted from Pine Script for integration with Antigravity signal_mailer.

Components:
- VIX Fix (Fear Detection)
- Momentum (Linear Regression)
- SuperTrend
- EMA Long Term Trend
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SniperSignal:
    """Index Sniper signal result"""

    current_state: str  # "HOLD" or "EXIT"
    is_buy: bool
    is_sell: bool
    fear_zone: bool
    buy_window: bool
    momentum_status: str
    ema_status: str
    supertrend_value: float
    ema_value: float
    bars_since_fear: int
    fear_window: int


class IndexSniper:
    """
    Index Sniper V8.2 (Weekly Optimized)

    원본: TradingView Pine Script
    용도: 주봉 스윙 트레이딩 신호
    """

    def __init__(
        self,
        # VIX Fix Settings
        pd_lookback: int = 22,
        bbl: int = 20,
        mult: float = 2.0,
        lb: int = 50,
        fear_window: int = 12,
        # Trend Settings
        st_factor: float = 2.0,
        st_period: int = 10,
        mom_len: int = 10,
        ema_len: int = 40,
        # Filters
        use_vol: bool = False,
        vol_mult: float = 1.5,
        use_accel: bool = False,
        strict_fear: bool = False,
        use_ema_exit: bool = True,
        cooldown_bars: int = 5,
    ):
        self.pd_lookback = pd_lookback
        self.bbl = bbl
        self.mult = mult
        self.lb = lb
        self.fear_window = fear_window

        self.st_factor = st_factor
        self.st_period = st_period
        self.mom_len = mom_len
        self.ema_len = ema_len

        self.use_vol = use_vol
        self.vol_mult = vol_mult
        self.use_accel = use_accel
        self.strict_fear = strict_fear
        self.use_ema_exit = use_ema_exit
        self.cooldown_bars = cooldown_bars

        # State tracking
        self.last_buy_idx = -100

    def _calc_vix_fix(self, close: pd.Series, low: pd.Series) -> pd.Series:
        """Williams VIX Fix calculation"""
        highest_close = close.rolling(self.pd_lookback).max()
        wvf = ((highest_close - low) / highest_close) * 100
        return wvf

    def _calc_bollinger(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Bollinger Bands for VIX Fix"""
        midline = series.rolling(self.bbl).mean()
        std = series.rolling(self.bbl).std()
        upper = midline + (self.mult * std)
        return midline, upper

    def _calc_momentum(
        self, close: pd.Series, high: pd.Series, low: pd.Series
    ) -> pd.Series:
        """Linear regression momentum"""
        hl_avg = (
            high.rolling(self.mom_len).max() + low.rolling(self.mom_len).min()
        ) / 2
        sma = close.rolling(self.mom_len).mean()
        final_avg = (hl_avg + sma) / 2

        # Linear regression of (close - final_avg)
        diff = close - final_avg

        # Simple approximation of linreg value
        # In Pine: linreg(source, length, offset)
        # We'll use a simple rolling linear regression slope
        def linreg_val(x):
            if len(x) < self.mom_len:
                return np.nan
            y = x.values
            n = len(y)
            x_vals = np.arange(n)
            denom = n * np.sum(x_vals**2) - np.sum(x_vals) ** 2
            if denom == 0:
                return np.nan
            slope = (n * np.sum(x_vals * y) - np.sum(x_vals) * np.sum(y)) / denom
            intercept = (np.sum(y) - slope * np.sum(x_vals)) / n
            return intercept + slope * (n - 1)  # Value at last point

        return diff.rolling(self.mom_len).apply(linreg_val, raw=False)  # type: ignore

    def _calc_supertrend(
        self, close: pd.Series, high: pd.Series, low: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """SuperTrend indicator"""
        atr_period = self.st_period
        multiplier = self.st_factor

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        # Basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        # SuperTrend calculation
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)

        # Initialize
        # direction.iloc[:atr_period] = 1  # Default bearish
        # supertrend.iloc[:atr_period] = upper_band.iloc[:atr_period]

        # Safer initialization
        for i in range(min(atr_period, len(close))):
            direction.iloc[i] = 1
            supertrend.iloc[i] = upper_band.iloc[i]

        for i in range(atr_period, len(close)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                supertrend.iloc[i] = np.nan
                direction.iloc[i] = 0
                continue

            # Previous values
            prev_st = (
                supertrend.iloc[i - 1]
                if not pd.isna(supertrend.iloc[i - 1])
                else upper_band.iloc[i]
            )
            prev_dir = direction.iloc[i - 1]

            # Current close
            curr_close = close.iloc[i]
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]

            # Direction logic
            if prev_dir == -1:  # Was bullish
                if curr_close < prev_st:
                    direction.iloc[i] = 1  # Flip to bearish
                    supertrend.iloc[i] = curr_upper
                else:
                    direction.iloc[i] = -1
                    supertrend.iloc[i] = max(curr_lower, prev_st)
            else:  # Was bearish
                if curr_close > prev_st:
                    direction.iloc[i] = -1  # Flip to bullish
                    supertrend.iloc[i] = curr_lower
                else:
                    direction.iloc[i] = 1
                    supertrend.iloc[i] = min(curr_upper, prev_st)

        return supertrend, direction

    def _bars_since(self, condition: pd.Series) -> pd.Series:
        """Calculate bars since condition was True"""
        result = pd.Series(
            index=condition.index, dtype=float
        )  # Use float for NaN support
        count = 0

        for i in range(len(condition)):
            if pd.isna(condition.iloc[i]):
                result.iloc[i] = np.nan
                continue

            if condition.iloc[i]:
                count = 0
            else:
                count += 1
            result.iloc[i] = count

        return result

    def analyze(self, df: pd.DataFrame, ticker: str = "Unknown") -> SniperSignal:
        """
        Analyze ticker and generate signal

        Args:
            df: DataFrame with columns: Open, High, Low, Close, Volume
            ticker: Ticker symbol for logging

        Returns:
            SniperSignal object
        """
        min_len = (
            max(self.pd_lookback, self.bbl, self.lb, self.ema_len, self.st_period) + 20
        )
        if len(df) < min_len:
            return SniperSignal(
                current_state="DATA_SHORT",
                is_buy=False,
                is_sell=False,
                fear_zone=False,
                buy_window=False,
                momentum_status="N/A",
                ema_status="N/A",
                supertrend_value=0,
                ema_value=0,
                bars_since_fear=999,
                fear_window=self.fear_window,
            )

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = (
            df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)
        )

        # 1. VIX Fix (Fear Detection)
        wvf = self._calc_vix_fix(close, low)
        midline, upper_band = self._calc_bollinger(wvf)
        range_high = wvf.rolling(self.lb).max() * 0.85

        cond_band = wvf >= upper_band
        cond_range = wvf >= range_high

        if self.strict_fear:
            is_fear = cond_band & cond_range
        else:
            is_fear = cond_band | cond_range

        # 2. Momentum
        val = self._calc_momentum(close, high, low)
        is_bull_mom = val > 0
        is_accel = val > val.shift(1)
        mom_cross_up = (val > 0) & (val.shift(1) <= 0)

        # 3. Volume
        vol_sma = volume.rolling(20).mean()
        vol_cond = volume > (vol_sma * self.vol_mult)

        # 4. Trend
        supertrend, direction = self._calc_supertrend(close, high, low)
        ema_long = close.ewm(span=self.ema_len, adjust=False).mean()

        # Trend flips
        trend_flip_bull = (direction == -1) & (direction.shift(1) == 1)
        trend_flip_bear = (direction == 1) & (direction.shift(1) == -1)

        # 5. Bars since fear
        bars_since_fear = self._bars_since(is_fear)
        was_fear_recently = bars_since_fear < self.fear_window

        # 6. Filter conditions - explicitly handle as Series
        use_vol_s = pd.Series(self.use_vol, index=df.index)
        use_accel_s = pd.Series(self.use_accel, index=df.index)

        pass_vol = (~use_vol_s) | vol_cond
        pass_accel = (~use_accel_s) | is_accel

        # 7. Buy conditions - safe boolean logic
        buy_cond_1 = (
            was_fear_recently & is_bull_mom & trend_flip_bull & pass_vol & pass_accel
        )

        # buy_cond_2 components
        c1 = was_fear_recently
        c2 = is_bull_mom
        c3 = direction == -1
        c4 = close > ema_long
        c5 = mom_cross_up
        c6 = pass_vol
        c7 = pass_accel

        buy_cond_2 = c1 & c2 & c3 & c4 & c5 & c6 & c7

        raw_buy = buy_cond_1 | buy_cond_2

        # 8. Sell conditions
        exit_st = trend_flip_bear

        use_ema_exit_s = pd.Series(self.use_ema_exit, index=df.index)
        ema_condition = (close < ema_long) & (close.shift(1) >= ema_long.shift(1))
        exit_ema = use_ema_exit_s & ema_condition

        raw_sell = exit_st | exit_ema

        # Get latest values safely
        latest_idx = len(df) - 1

        def safe_bool(val):
            if pd.isna(val):
                return False
            return bool(val)

        # Check cooldown for buy
        is_buy = False
        latest_raw_buy = safe_bool(raw_buy.iloc[-1])

        if latest_raw_buy:
            if latest_idx - self.last_buy_idx > self.cooldown_bars:
                is_buy = True
                self.last_buy_idx = latest_idx

        is_sell = safe_bool(raw_sell.iloc[-1])

        # Build result components
        latest_direction = direction.iloc[-1]
        current_state = "HOLD" if latest_direction == -1 else "EXIT"

        fear_zone = safe_bool(is_fear.iloc[-1])
        latest_wfr = safe_bool(was_fear_recently.iloc[-1])
        buy_window = latest_wfr and not fear_zone

        mom_val = val.iloc[-1]

        if pd.isna(mom_val):
            momentum_status = "N/A"
        elif mom_val > 0:
            latest_accel = safe_bool(is_accel.iloc[-1])
            if latest_accel:
                momentum_status = "상승 가속 📈"
            else:
                momentum_status = "상승 중 📈"
        else:
            momentum_status = "하락 중 📉"

        latest_close = close.iloc[-1]
        latest_ema = ema_long.iloc[-1]
        ema_status = "EMA 위 ✅" if latest_close > latest_ema else "EMA 아래 ⚠️"

        latest_bars_fear = bars_since_fear.iloc[-1]
        bars_fear_val = int(latest_bars_fear) if not pd.isna(latest_bars_fear) else 999

        latest_supertrend = supertrend.iloc[-1]
        st_val = float(latest_supertrend) if not pd.isna(latest_supertrend) else 0.0

        ema_val = float(latest_ema) if not pd.isna(latest_ema) else 0.0

        return SniperSignal(
            current_state=current_state,
            is_buy=is_buy,
            is_sell=is_sell,
            fear_zone=fear_zone,
            buy_window=buy_window,
            momentum_status=momentum_status,
            ema_status=ema_status,
            supertrend_value=st_val,
            ema_value=ema_val,
            bars_since_fear=bars_fear_val,
            fear_window=self.fear_window,
        )


def analyze_ticker(
    ticker: str, period: str = "2y", interval: str = "1wk"
) -> Optional[SniperSignal]:
    """
    Analyze a single ticker

    Args:
        ticker: Stock/ETF ticker symbol
        period: yfinance period (e.g., "2y", "5y", "max")
        interval: yfinance interval (e.g., "1wk" for weekly)

    Returns:
        SniperSignal or None if error
    """
    import yfinance as yf
    import traceback

    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            group_by="ticker",
            auto_adjust=True,
        )
        if data.empty:
            print(f"No data found for {ticker}")
            return None

        # print(f"DEBUG: {ticker} Columns: {data.columns}")

        # Flatten MultiIndex columns if present
        # yf 0.2.x+ with group_by="ticker" creates Price, Ticker hierarchy? No, actually Ticker, Price.
        # But we asked for auto_adjust=True.
        # Let's inspect structure dynamically.

        df = data.copy()

        # Case 1: Ticker is top level (Standard multi-ticker download structure)
        if isinstance(df.columns, pd.MultiIndex):
            # If (Ticker, Price)
            if ticker in df.columns.levels[0]:
                df = df[ticker]
            # If (Price, Ticker) - rare but possible
            elif ticker in df.columns.levels[1]:
                df = df.xs(ticker, axis=1, level=1)

        # Case 2: Single ticker, columns might be just Price types
        # But sometimes they retain MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            # Try to drop levels if Close is found
            df.columns = df.columns.droplevel(0)  # Hope it exposes Close

        # Ensure 'Close' exists
        if "Close" not in df.columns:
            # Try flattening completely if single ticker
            # print(f"DEBUG: Columns after flattening: {df.columns}")
            pass

        sniper = IndexSniper()
        return sniper.analyze(df, ticker)
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        # traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test with VGT (Vanguard IT ETF)
    tickers = ["VGT", "QQQ", "SPY"]

    for ticker in tickers:
        result = analyze_ticker(ticker)
        if result:
            print(f"\n📊 {ticker} - Index Sniper V8.2")
            print(f"   State: {result.current_state}")
            print(f"   BUY Signal: {'🟢 YES' if result.is_buy else 'No'}")
            print(f"   SELL Signal: {'🔴 YES' if result.is_sell else 'No'}")
            print(f"   Fear Zone: {'🔴 Active' if result.fear_zone else 'Clear'}")
            print(f"   Buy Window: {'🟠 Open' if result.buy_window else 'Closed'}")
            print(f"   Momentum: {result.momentum_status}")
            print(f"   Trend: {result.ema_status}")
