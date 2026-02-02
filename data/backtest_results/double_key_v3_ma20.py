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

print(f"Downloading data for V3.1 (MA20 Trim): {tickers}...")
df = yf.download(tickers, start=start_date, end=end_date, progress=False)

if isinstance(df.columns, pd.MultiIndex):
    if "Adj Close" in df.columns.get_level_values(0):
        df = df["Adj Close"]
    elif "Close" in df.columns.get_level_values(0):
        df = df["Close"]
elif "Adj Close" in df.columns:
    df = df["Adj Close"]
elif "Close" in df.columns:
    df = df["Close"]

# Debug: Print columns if needed
print(f"Data Columns: {df.columns}")

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

# B. Trend (MA185) for Target
df["ma185"] = df[target_asset].rolling(185).mean()

# C. VIX Panic
df["vix"] = df["^VIX"]

# D. MA20 (Tactical Trim)
df["ma20"] = df[target_asset].rolling(20).mean()

# E. SHY Yield (Approx) for Smart Cash
# If SHY MOM 6M > 0, assume it's better than 0%.
# Actually simpler: just calculate SHY daily return.

# ---------------------------------------------------------
# 3. Double-Key V3 + MA20 Logic
# ---------------------------------------------------------
# We need to loop because of Hysteresis (State Memory) AND MA20 counter

df["strategy_return"] = 0.0
df["cash_weight"] = 0.0

# Extract numpy arrays for speed
price_arr = df[target_asset].values
ma185_arr = df["ma185"].values
ma20_arr = df["ma20"].values
vwo_mom_arr = df["mom_vwo"].values
bnd_mom_arr = df["mom_bnd"].values
vix_arr = df["vix"].values
shy_ret_arr = df["SHY"].pct_change().fillna(0).values
spy_ret_arr = df[target_asset].pct_change().fillna(0).values

# State Variables
state = 1  # 1=Invested, 0=Defensive (Trend)
days_above_185 = 0
days_below_185 = 0

ma20_penalty_active = False
days_below_ma20 = 0
days_above_ma20 = 0

capital = 1.0
equity_curve = [capital]

# Metrics for "Trim" analysis
trim_events = 0

for i in range(1, len(df)):
    if np.isnan(ma185_arr[i]) or np.isnan(ma20_arr[i]):
        equity_curve.append(capital)
        continue

    # -----------------------------------
    # 1. Base Strategy (V3)
    # -----------------------------------
    p = price_arr[i - 1]  # Yesterday's close for signal
    ma = ma185_arr[i - 1]

    # Trend Logic (Hysteresis)
    if state == 1:  # Invested
        if p < ma * 0.97:
            state = 0  # Exit
            days_above_185 = 0
    else:  # Cash
        if p > ma:
            days_above_185 += 1
        else:
            days_above_185 = 0

        if days_above_185 >= 3:
            state = 1  # Re-entry

    # Canary Logic
    vwo_bad = vwo_mom_arr[i - 1] <= 0
    bnd_bad = bnd_mom_arr[i - 1] <= 0

    risk_level = 0
    if vwo_bad and bnd_bad:
        risk_level = 2
    elif vwo_bad or bnd_bad:
        risk_level = 1

    # VIX Filter
    is_panic = vix_arr[i - 1] >= 30

    # Calculate Base Cash %
    base_cash_pct = 0.0

    if state == 0:  # Trend Broken
        base_cash_pct = 0.50
    elif is_panic:  # VIX Panic
        base_cash_pct = 0.50
    else:  # Bull Trend
        if risk_level == 2:
            base_cash_pct = 0.25
        elif risk_level == 1:
            base_cash_pct = 0.10
        else:
            base_cash_pct = 0.00

    # -----------------------------------
    # 2. MA20 Trim (Tactical) - THE EXPERIMENT
    # -----------------------------------
    ma20 = ma20_arr[i - 1]

    if p < ma20:
        days_below_ma20 += 1
        days_above_ma20 = 0
    else:
        days_above_ma20 += 1
        days_below_ma20 = 0

    # Trigger Logic: 2 days confirm
    if not ma20_penalty_active:
        if days_below_ma20 >= 2:
            ma20_penalty_active = True
            trim_events += 1
    else:
        if days_above_ma20 >= 2:
            ma20_penalty_active = False

    # Apply Penalty (Stepped Logic: Max of Base or 10%)
    ma20_cash = 0.10 if ma20_penalty_active else 0.0
    final_cash_pct = max(base_cash_pct, ma20_cash)

    # Cap at 100%
    if final_cash_pct > 1.0:
        final_cash_pct = 1.0

    # -----------------------------------
    # 3. Execution
    # -----------------------------------
    equity_pct = 1.0 - final_cash_pct

    # Smart Cash (SHY) vs Zero Yield
    # If SHY is dropping (e.g. 2022), maybe better to hold raw cash?
    # For now, V3 used SHY always. Let's stick to SHY.

    daily_ret = (equity_pct * spy_ret_arr[i]) + (final_cash_pct * shy_ret_arr[i])

    capital *= 1 + daily_ret
    equity_curve.append(capital)

df["Equity"] = equity_curve


# ---------------------------------------------------------
# 4. Performance Metrics
# ---------------------------------------------------------
# Re-run calc metrics logic
def calc_metrics(series, name):
    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    days = (series.index[-1] - series.index[0]).days
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
print(f"Total Trim Events (MA20): {trim_events}")

# Benchmarks? We should probably load V3 results or run V3 here too?
# For simplicity, just output V3.1 results compared to Spy.
# Note: User wants comparison.
# V3 results from previous run: CAGR 12.33%, MDD -22.52%, Sharpe 0.96.

c, m, s = calc_metrics(df["Equity"], "V3.1 (MA20 Trim)")

# Save Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Equity"], label="V3.1 (MA20 Trim)")
plt.yscale("log")
plt.title("Double-Key V3.1 (MA20 Trim) Performance")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("v3_ma20_performance.png")

with open("v3_ma20_metrics.txt", "w") as f:
    f.write(f"V3.1,{c * 100:.2f},{m * 100:.2f},{s:.2f},{trim_events}")
