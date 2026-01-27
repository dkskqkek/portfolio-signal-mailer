# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MAEngine:
    @staticmethod
    def SMA(series, period):
        return series.rolling(window=period).mean()

    @staticmethod
    def EMA(series, period):
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def WMA(series, period):
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def HMA(series, period):
        """Hull Moving Average"""
        half_len = int(period / 2)
        sqrt_len = int(np.sqrt(period))
        wma_half = MAEngine.WMA(series, half_len)
        wma_full = MAEngine.WMA(series, period)
        diff = 2 * wma_half - wma_full
        return MAEngine.WMA(diff, sqrt_len)

    @staticmethod
    def ZLEMA(series, period):
        """Zero Lag Exponential Moving Average"""
        lag = (period - 1) // 2
        data_lag = series + (series - series.shift(lag))
        return MAEngine.EMA(data_lag, period)

    @staticmethod
    def KAMA(series, period, fast_span=2, slow_span=30):
        """Kaufman's Adaptive Moving Average"""
        change = (series - series.shift(period)).abs()
        volatility = (series - series.shift(1)).abs().rolling(window=period).sum()
        er = change / volatility
        sc = (er * (2/(fast_span+1) - 2/(slow_span+1)) + 2/(slow_span+1))**2
        kama = pd.Series(index=series.index, dtype=float)
        for i in range(period, len(series)):
            if i == period:
                kama.iloc[i] = series.iloc[i]
            else:
                kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
        return kama

def backtest_ma_logic(df, ticker, ma_type='SMA', period=200, safe_ticker='BIL'):
    close = df[ticker]
    safe = df[safe_ticker]
    
    # Calculate MA
    if ma_type == 'SMA': ma_line = MAEngine.SMA(close, period)
    elif ma_type == 'EMA': ma_line = MAEngine.EMA(close, period)
    elif ma_type == 'WMA': ma_line = MAEngine.WMA(close, period)
    elif ma_type == 'HMA': ma_line = MAEngine.HMA(close, period)
    elif ma_type == 'ZLEMA': ma_line = MAEngine.ZLEMA(close, period)
    elif ma_type == 'KAMA': ma_line = MAEngine.KAMA(close, period)
    else: ma_line = MAEngine.SMA(close, period)
    
    # Signals: 0 = Risky, 1 = Safe (Danger)
    # Applied to next day's return
    signal = (close < ma_line).astype(int).shift(1).fillna(0)
    
    risky_ret = close.pct_change().fillna(0)
    safe_ret = safe.pct_change().fillna(0)
    
    # Strategy Return
    strat_ret = np.where(signal == 1, safe_ret, risky_ret)
    equity = pd.Series((1 + strat_ret).cumprod() * 100, index=close.index)
    
    # Metrics (post 250 burn-in)
    eval_equity = equity.iloc[250:]
    if len(eval_equity) < 10: return {'cagr': 0, 'mdd': 0, 'sharpe': 0}
    
    cagr = (eval_equity.iloc[-1] / eval_equity.iloc[0]) ** (252 / len(eval_equity)) - 1
    mdd = (eval_equity / eval_equity.cummax() - 1).min()
    daily_ret = eval_equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else 0
    
    return {
        'ma_type': ma_type,
        'period': period,
        'cagr': cagr * 100,
        'mdd': mdd * 100,
        'sharpe': sharpe,
        'final_value': eval_equity.iloc[-1]
    }
