import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Data Download
# ---------------------------------------------------------
target_asset = "SPY"
canary_assets = ["VWO", "BND"]
defensive_assets = ["SHY"]
risk_indicator = "^VIX"

start_date = "2008-01-01"
end_date = "2024-12-31"

tickers = [target_asset] + canary_assets + defensive_assets + [risk_indicator]

print(f"Downloading data for V3.2 (MA60 Blend)...")
df = yf.download(tickers, start=start_date, end=end_date, progress=False)

# Robust Column Handling
if isinstance(df.columns, pd.MultiIndex):
    if "Adj Close" in df.columns.get_level_values(0):
        df = df["Adj Close"]
    elif "Close" in df.columns.get_level_values(0):
        df = df["Close"]
elif "Adj Close" in df.columns:
    df = df["Adj Close"]
elif "Close" in df.columns:
    df = df["Close"]

df = df.ffill()

# Normalize Timezone
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# ---------------------------------------------------------
# 2. Indicator Calculation
# ---------------------------------------------------------
# A. Macro Canary (Momentum 12M)
df["mom_vwo"] = df["VWO"].pct_change(252)
df["mom_bnd"] = df["BND"].pct_change(252)

# B. Trends
df["ma185"] = df[target_asset].rolling(185).mean()
df["ma60"] = df[target_asset].rolling(60).mean()

# C. VIX
df["vix"] = df["^VIX"]

# ---------------------------------------------------------
# 3. Double-Key V3.2 (MA60 Blend) Logic
# ---------------------------------------------------------
# Logic:
# Core Port (80%): Follows V3 (MA185 + Canary)
# Sat Port (20%): Follows MA60 (Simple Trend)
# Global Override: If VIX > 30, Force Max 50% Equity (Panic Defense)

df["strategy_return"] = 0.0
equity_curve = [1.0]

# Arrays
price_arr = df[target_asset].values
ma185_arr = df["ma185"].values
ma60_arr = df["ma60"].values
vwo_mom_arr = df["mom_vwo"].values
bnd_mom_arr = df["mom_bnd"].values
vix_arr = df["vix"].values
shy_ret_arr = df["SHY"].pct_change().fillna(0).values
spy_ret_arr = df[target_asset].pct_change().fillna(0).values

# State Management
state_185 = 1  # 1=Bull, 0=Bear
days_above_185 = 0

state_60 = 1  # 1=Bull, 0=Bear
days_above_60 = 0

capital = 1.0

for i in range(1, len(df)):
    if np.isnan(ma185_arr[i]) or np.isnan(ma60_arr[i]):
        equity_curve.append(capital)
        continue

    p = price_arr[i - 1]

    # --- Logic A: MA185 (Core 80%) ---
    ma_long = ma185_arr[i - 1]
    if state_185 == 1:
        if p < ma_long * 0.97:
            state_185 = 0
            days_above_185 = 0
    else:
        if p > ma_long:
            days_above_185 += 1
        else:
            days_above_185 = 0
        if days_above_185 >= 3:
            state_185 = 1

    # --- Logic B: MA60 (Sat 20%) ---
    ma_med = ma60_arr[i - 1]
    # Simple logic for MA60: Just Price < MA vs Price > MA (maybe 2 day confirm?)
    # Let's use 2 day confirm like User suggested for MA20, or V3 style 3 day?
    # Let's use V3 style 3-day logic but without the 3% buffer (MA60 is tighter)
    if state_60 == 1:
        if p < ma_med * 0.99:  # 1% Buffer for MA60
            state_60 = 0
            days_above_60 = 0
    else:
        if p > ma_med:
            days_above_60 += 1
        else:
            days_above_60 = 0
        if days_above_60 >= 2:  # Faster reentry
            state_60 = 1

    # --- Canary & Risk ---
    vwo_bad = vwo_mom_arr[i - 1] <= 0
    bnd_bad = bnd_mom_arr[i - 1] <= 0
    risk_level = 0
    if vwo_bad and bnd_bad:
        risk_level = 2
    elif vwo_bad or bnd_bad:
        risk_level = 1

    is_panic = vix_arr[i - 1] >= 30

    # --- Allocation Calculation ---

    # 1. Bucket A (MA185, 80% Weight)
    # Inside this bucket, we apply V3 Cash rules
    # If Bear (state=0): 50% Cash (Relative to bucket) -> So 40% Equity / 40% Cash?
    # Wait, V3 says "Target Cash %".
    # Let's calculate Equity % for Bucket A.

    equity_A = 1.0  # Default
    if state_185 == 0:
        equity_A = 0.50  # Bear
    elif risk_level == 2:
        equity_A = 0.75  # Macro Bad
    elif risk_level == 1:
        equity_A = 0.90  # Macro Mixed

    # 2. Bucket B (MA60, 20% Weight)
    equity_B = 1.0
    if state_60 == 0:
        equity_B = 0.0  # Full Cash in this bucket if MA60 broken

    # 3. Blending
    # Total Equity = (0.8 * equity_A) + (0.2 * equity_B)
    final_equity_pct = (0.8 * equity_A) + (0.2 * equity_B)

    # 4. Global Panic Override
    if is_panic:
        # VIX > 30 implies MAX 50% Equity.
        if final_equity_pct > 0.50:
            final_equity_pct = 0.50

    # Execution
    final_cash_pct = 1.0 - final_equity_pct

    daily_ret = (final_equity_pct * spy_ret_arr[i]) + (final_cash_pct * shy_ret_arr[i])
    capital *= 1 + daily_ret
    equity_curve.append(capital)

df["Equity"] = equity_curve


# Metrics
def calc_metrics(series, name):
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    days = (series.index[-1] - series.index[0]).days
    if days == 0:
        return 0, 0, 0
    cagr = (1 + total_ret) ** (365 / days) - 1

    peak = series.cummax()
    dd = (series - peak) / peak
    mdd = dd.min()

    daily_rets = series.pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)

    print(
        f"[{name}] CAGR: {cagr * 100:.2f}% | MDD: {mdd * 100:.2f}% | Sharpe: {sharpe:.2f}"
    )
    return cagr, mdd, sharpe


print("\n--- Strategy Performance (2008-2024) ---")
c, m, s = calc_metrics(df["Equity"], "V3.2 (MA60 Blend)")

with open("v3_ma60_metrics.txt", "w") as f:
    f.write(f"V3.2,{c * 100:.2f},{m * 100:.2f},{s:.2f}")
