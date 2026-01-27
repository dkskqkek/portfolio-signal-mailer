# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add root to path to import signal_mailer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research.data_loader import fetch_validated_data
from signal_mailer.index_sniper import IndexSniper


def calc_performance(equity_curve):
    """Calculate CAGR, MDD, Sharpe"""
    if equity_curve.empty:
        return {}

    # CAGR
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    tr = equity_curve.iloc[-1] / equity_curve.iloc[0]
    cagr = (tr ** (365 / days)) - 1

    # MDD
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    mdd = dd.min()

    # Sharpe (RF=4%)
    daily_ret = equity_curve.pct_change().dropna()
    excess_ret = daily_ret - (0.04 / 252)
    sharpe = np.sqrt(252) * (excess_ret.mean() / daily_ret.std())

    return {"CAGR": cagr * 100, "MDD": mdd * 100, "Sharpe": sharpe}


def run_backtest():
    print(">>> Backtesting Index Sniper vs Existing Logic...")

    # 1. Fetch Data
    tickers = ["QQQ", "QLD", "SHY"]  # SHY as defensive proxy (Cash/Bond)
    data = fetch_validated_data(tickers, start_date="2008-01-01")

    # Extract Daily Close
    close_daily = data.xs("Close", level=1, axis=1)
    qqq_daily = close_daily["QQQ"]
    qld_daily = close_daily["QLD"]
    shy_daily = close_daily["SHY"]

    # 2. Prepare Weekly Data for Sniper
    # Resample to Weekly (Friday)
    logic_data_weekly = (
        data["QQQ"]
        .resample("W-FRI")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )

    # 3. Generate Sniper Signals (Vectorized)
    print(">>> Calculating Sniper Signals (Weekly)...")
    sniper = IndexSniper()

    # We need to expose internal calc methods of IndexSniper or modify it.
    # Luckily, helper methods _calc_supertrend etc take Series, so we can use them!

    # Calculate components
    st, direction = sniper._calc_supertrend(
        logic_data_weekly["Close"], logic_data_weekly["High"], logic_data_weekly["Low"]
    )

    # VIX Fix
    wvf = sniper._calc_vix_fix(logic_data_weekly["Close"], logic_data_weekly["Low"])
    midline, upper_band = sniper._calc_bollinger(wvf)
    range_high = wvf.rolling(sniper.lb).max() * 0.85
    cond_band = wvf >= upper_band
    cond_range = wvf >= range_high
    is_fear = cond_band | cond_range

    # Momentum
    mom_val = sniper._calc_momentum(
        logic_data_weekly["Close"], logic_data_weekly["High"], logic_data_weekly["Low"]
    )
    is_bull_mom = mom_val > 0
    is_accel = mom_val > mom_val.shift(1)

    # EMA
    ema_long = logic_data_weekly["Close"].ewm(span=sniper.ema_len, adjust=False).mean()

    # Buy/Sell Logic Recreation (Vectorized is hard for stateful 'was_fear_recently')
    # Use loop for precise state reproduction

    signals = pd.Series(0, index=logic_data_weekly.index)  # 1=Buy/Hold, -1=Defensive

    # State
    last_buy_idx = -100
    curr_state = -1  # Start defensive

    # For loop to simulate real-time decision
    # logic_data_weekly index i corresponds to the END of that week.
    # The decision applies to the NEXT week.

    signal_history = []

    for i in range(len(logic_data_weekly)):
        if i < 50:
            signal_history.append(0)  # Neutral/Defensive
            continue

        # Context at end of week i
        c_dir = direction.iloc[i]
        c_close = logic_data_weekly["Close"].iloc[i]
        c_ema = ema_long.iloc[i]
        c_mom = mom_val.iloc[i]
        c_fear = is_fear.iloc[i]

        # Check fear history (recent window)
        # Lookback 12 bars exclude current
        start_win = max(0, i - sniper.fear_window)
        recent_fear_window = is_fear.iloc[
            start_win:i
        ]  # python slice excludes i? no we want up to i-1?
        # was_fear_recently = any true in last 12 bars?
        # Pine ta.barssince(is_fear) < 12
        # means is_fear was true 0..11 bars ago.
        was_fear_recently = recent_fear_window.any()

        # Buy Logic
        # 1. Fear + Bull Mom + Trend Flip
        trend_flip_bull = (c_dir == -1) and (direction.iloc[i - 1] == 1)
        # pass_vol, pass_accel assumed true for simplicity or if disabled

        buy_cond_1 = was_fear_recently and (c_mom > 0) and trend_flip_bull

        # 2. Golden Cross style
        mom_cross_up = (c_mom > 0) and (mom_val.iloc[i - 1] <= 0)
        buy_cond_2 = (
            was_fear_recently
            and (c_mom > 0)
            and (c_dir == -1)
            and (c_close > c_ema)
            and mom_cross_up
        )

        raw_buy = buy_cond_1 or buy_cond_2

        # Sell Logic
        trend_flip_bear = (c_dir == 1) and (direction.iloc[i - 1] == -1)
        ema_cross_under = (c_close < c_ema) and (
            logic_data_weekly["Close"].iloc[i - 1] >= ema_long.iloc[i - 1]
        )

        raw_sell = trend_flip_bear or ema_cross_under

        # State Machine
        next_state = curr_state

        if raw_buy and (i - last_buy_idx > sniper.cooldown_bars):
            next_state = 1  # HOLD
            last_buy_idx = i
        elif raw_sell:
            next_state = -1  # EXIT

        # Additional: If already HOLD, stay HOLD unless SELL
        if curr_state == 1 and not raw_sell:
            next_state = 1

        # Additional: If already EXIT, stay EXIT unless BUY
        if curr_state == -1 and not raw_buy:
            next_state = -1

        curr_state = next_state
        signal_history.append(curr_state)

    logic_data_weekly["Signal"] = signal_history

    # 4. Align Weekly Signal to Daily
    # Signal at Friday Close determines allocation for NEXT Monday onwards
    # Shift Weekly signal by 1 week then reindex?
    # Better: Reindex to Daily (ffill) then shift 1 day?
    # Correct Way:
    #   Weekly Date (Fri) -> Signal calculated at Fri close.
    #   Daily Series reindexed from Weekly.
    #   Fill with previous week's signal.

    daily_signal_sniper = (
        logic_data_weekly["Signal"]
        .reindex(close_daily.index)
        .ffill()
        .shift(1)
        .fillna(0)
    )

    # 5. Calculate Dual SMA Signal (Daily) for Comparison
    # MA 110/250 on Daily QQQ
    ma110 = qqq_daily.rolling(110).mean()
    ma250 = qqq_daily.rolling(250).mean()

    # Bull if QQQ > 250 (Basic Logic) or use Dual logic
    # Simplified Logic we use: QLD if QQQ > MA250, else Defensive
    daily_signal_dual = (
        (qqq_daily > ma250).astype(int).replace(0, -1).shift(1).fillna(0)
    )

    # 6. Run Backtest Loop
    def run_strategy(signal_series, name):
        cash = 10000.0
        shares = 0
        equity = []

        for date, sig in signal_series.items():
            price_qld = qld_daily.loc[date]
            price_def = shy_daily.loc[date]

            # 1 = Bull (QLD), -1 = Bear (SHY)
            if sig == 1:
                # Buy QLD
                val = (
                    (cash + shares * price_def) if shares == 0 else (shares * price_qld)
                )  # Wait, simple rebal
                # Let's use simple vector math for speed
                pass

        # Vectorized Backtest
        # Returns
        ret_qld = qld_daily.pct_change().fillna(0)
        ret_shy = shy_daily.pct_change().fillna(0)

        # Signal is for start of day.
        # Strategy Return = Signal * Asset_Ret
        # Signal 1 -> QLD, Signal -1 -> SHY
        # Map signal: 1 -> QLD Return, -1 -> SHY Return

        strat_ret = pd.Series(0.0, index=signal_series.index)
        strat_ret[signal_series == 1] = ret_qld[signal_series == 1]
        strat_ret[signal_series == -1] = ret_shy[signal_series == -1]
        # 0 or NaN -> Cash (0 return)

        equity = (1 + strat_ret).cumprod()
        return equity

    print(">>> Running Simulations...")
    eq_sniper = run_strategy(daily_signal_sniper, "Sniper")
    eq_dual = run_strategy(daily_signal_dual, "Dual SMA")
    eq_buyhold = (1 + qld_daily.pct_change().fillna(0)).cumprod()

    # 7. Results
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS (2008 - 2026)")
    print("=" * 50)

    res_bh = calc_performance(eq_buyhold)
    res_dual = calc_performance(eq_dual)
    res_sniper = calc_performance(eq_sniper)

    # Stats
    days_in_mkt = daily_signal_sniper[daily_signal_sniper == 1].count()
    total_days = len(daily_signal_sniper)
    mkt_exposure = (days_in_mkt / total_days) * 100

    print(
        f"{'Strategy':<15} | {'CAGR':<8} | {'MDD':<8} | {'Sharpe':<6} | {'Exposure':<8}"
    )
    print("-" * 65)
    print(
        f"{'QLD Buy&Hold':<15} | {res_bh['CAGR']:<7.2f}% | {res_bh['MDD']:<7.2f}% | {res_bh['Sharpe']:.2f}   | 100%    "
    )
    print(
        f"{'Dual SMA (Live)':<15} | {res_dual['CAGR']:<7.2f}% | {res_dual['MDD']:<7.2f}% | {res_dual['Sharpe']:.2f}   | {(pd.Series(daily_signal_dual) == 1).sum() / len(daily_signal_dual) * 100:.1f}%"
    )
    print(
        f"{'Index Sniper':<15} | {res_sniper['CAGR']:<7.2f}% | {res_sniper['MDD']:<7.2f}% | {res_sniper['Sharpe']:.2f}   | {mkt_exposure:.1f}%"
    )
    print("=" * 65)

    print("\n[Diagnosis]")
    print(f"Sniper Days in Market: {days_in_mkt} / {total_days}")
    if mkt_exposure < 50:
        print(
            "⚠️  Warning: Strategy is out of market >50% of time. 'Fear Condition' might be too strict."
        )

    print("\nCombined Strategy Idea:")
    print(
        "If we use Dual SMA as CORE, and Sniper only for 'Dip Buying' during Bear markets?"
    )

    print(
        "\nNote: 'Index Sniper' uses QLD when Bullish (Weekly Signal), SHY when Bearish."
    )
    print(
        "Comparison: Does checking Weekly SuperTrend/VIX beat simple Daily MA250 Trend?"
    )


if __name__ == "__main__":
    run_backtest()
