import pandas as pd
import numpy as np


def calculate_mmi(series, length=20):
    """
    Market Meanness Index (MMI) by JCL.
    Measures 'Mean Reversion'.
    MMI > 0.75 : Mean Reverting (Choppy)
    MMI < 0.75 : Trending

    Formula:
    Count pairs where price moves towards the median.
    """
    # 1. Calculate Median
    median = series.rolling(length).median()

    # 2. Identify Reversion
    # Py < Median AND Pt > Py  (Below median, moving up)
    # Py > Median AND Pt < Py  (Above median, moving down)

    prev_p = series.shift(1)
    # Note: median uses a rolling window including current,
    # but strictly we should compare against the median of the *window* context.
    # Standard implementation often just compares Pt and Pt-1 relative to Current Rolling Median.

    cond1 = (prev_p < median) & (series > prev_p)
    cond2 = (prev_p > median) & (series < prev_p)

    reverting = (cond1 | cond2).astype(int)

    # 3. Sum over window and normalize
    mmi = reverting.rolling(length).sum() / length

    return mmi


def calculate_t3(series, length=50, vfactor=0.4):
    """
    T3 Moving Average (Tim Tillson).
    T3 = GD(GD(GD(x)))
    GD = EMA(x)*(1+v) - EMA(EMA(x))*v
    """

    def gd(src, length, v):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        return ema1 * (1 + v) - ema2 * v

    t1 = gd(series, length, vfactor)
    t2 = gd(t1, length, vfactor)
    t3 = gd(t2, length, vfactor)

    return t3


def calculate_supertrend(df, period=10, factor=3.0):
    """
    SuperTrend Indicator.
    Returns: DataFrame with 'SuperTrend' (Line) and 'Direction' (1=Up, -1=Down)
    """
    # ATR Calculation
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    # Basic Bands
    hl2 = (high + low) / 2
    basic_upper = hl2 + (factor * atr)
    basic_lower = hl2 - (factor * atr)

    # Final Bands
    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)
    direction = pd.Series(1, index=df.index)  # 1: Bull, -1: Bear

    # Iterative calculation required for Trailing Logic
    # Using numpy for speed

    c_vals = close.values
    bu_vals = basic_upper.values
    bl_vals = basic_lower.values
    fu_vals = np.zeros(len(df))
    fl_vals = np.zeros(len(df))
    st_vals = np.zeros(len(df))
    dir_vals = np.zeros(len(df))

    # Initialize
    fu_vals[0] = bu_vals[0]
    fl_vals[0] = bl_vals[0]
    dir_vals[0] = 1

    for i in range(1, len(df)):
        # Final Upper
        if (bu_vals[i] < fu_vals[i - 1]) or (c_vals[i - 1] > fu_vals[i - 1]):
            fu_vals[i] = bu_vals[i]
        else:
            fu_vals[i] = fu_vals[i - 1]

        # Final Lower
        if (bl_vals[i] > fl_vals[i - 1]) or (c_vals[i - 1] < fl_vals[i - 1]):
            fl_vals[i] = bl_vals[i]
        else:
            fl_vals[i] = fl_vals[i - 1]

        # Direction
        prev_dir = dir_vals[i - 1]

        if prev_dir == 1:  # Bullish
            if c_vals[i] < fl_vals[i]:
                dir_vals[i] = -1
            else:
                dir_vals[i] = 1
        else:  # Bearish
            if c_vals[i] > fu_vals[i]:
                dir_vals[i] = 1
            else:
                dir_vals[i] = -1

        # SuperTrend Value
        if dir_vals[i] == 1:
            st_vals[i] = fl_vals[i]
        else:
            st_vals[i] = fu_vals[i]

    return pd.DataFrame({"SuperTrend": st_vals, "Direction": dir_vals}, index=df.index)
