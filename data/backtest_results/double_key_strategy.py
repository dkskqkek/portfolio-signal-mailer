import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Data Download
# ---------------------------------------------------------
target_asset = "SPY"
canary_assets = ["VWO", "BND"]
all_symbols = [target_asset] + canary_assets

print(f"Downloading data for: {all_symbols}")
data = yf.download(all_symbols, start="2010-01-01", progress=False)

# Robust Data Access
if "Adj Close" in data.columns:
    data = data["Adj Close"]
elif "Close" in data.columns:
    print("'Adj Close' not found, using 'Close'")
    data = data["Close"]
else:
    # Handle MultiIndex
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                data = data.xs("Adj Close", level=0, axis=1)
            elif "Close" in data.columns.get_level_values(0):
                data = data.xs("Close", level=0, axis=1)
    except:
        pass

data = data.ffill()

# ---------------------------------------------------------
# 2. Indicator Calculation
# ---------------------------------------------------------

# A. Key 1: Canary Momentum (12-month)
momentum_window = 252
canary_mom = data[canary_assets].pct_change(momentum_window)

# B. Key 2: Price Tunnel components
ma_window = 185
ma185 = data[target_asset].rolling(window=ma_window).mean()
buffer_exit_line = ma185 * 0.97
# Re-entry line is just ma185
price = data[target_asset]

# ---------------------------------------------------------
# 3. Strategy Logic (Double-Key with 3-Day Re-entry)
# ---------------------------------------------------------

df = pd.DataFrame(
    {
        "Price": price,
        "MA185": ma185,
        "ExitLine": buffer_exit_line,
        "VWO_Mom": canary_mom["VWO"],
        "BND_Mom": canary_mom["BND"],
    }
).dropna()

# Pre-calculate simple boolean conditions
cond_exit_zone = df["Price"] < df["ExitLine"]
cond_above_ma = df["Price"] > df["MA185"]

# Vectorized Canary Status
vwo_bad = df["VWO_Mom"] <= 0
bnd_bad = df["BND_Mom"] <= 0
any_canary_bad = vwo_bad | bnd_bad
all_canary_bad = vwo_bad & bnd_bad

# State Iteration for Price Tunnel
# State 0: Normal/Bullish (Price key allows 100%)
# State 1: Crisis (Price key forces 50% Cash)
price_tunnel_crisis = np.zeros(len(df), dtype=int)
current_crisis_state = 0  # Start as Normal
days_above_ma = 0

price_vals = df["Price"].values
ma_vals = df["MA185"].values
exit_vals = df["ExitLine"].values

# Standard loop for hysteresis with counter
for i in range(len(df)):
    p = price_vals[i]
    ma = ma_vals[i]
    exit_l = exit_vals[i]

    # 1. Check Exit Trigger
    if current_crisis_state == 0:
        if p < exit_l:
            current_crisis_state = 1  # Enter Crisis
            days_above_ma = 0  # Reset counter

    # 2. Check Re-entry Trigger
    else:  # current_crisis_state == 1
        if p > ma:
            days_above_ma += 1
        else:
            days_above_ma = 0  # Reset if falls below MA

        if days_above_ma >= 3:
            current_crisis_state = 0  # Back to Normal

    price_tunnel_crisis[i] = current_crisis_state

# Combine Rules
# 1. Canary Logic (Base Cash)
#    - Any Bad: 10%
#    - All Bad: 25%
# 2. Price Tunnel Logic (Overlay)
#    - Crisis: Minimum 50% (Overrides Canary)
#    - Normal: Follow Canary (0%, 10%, 25%)

cash_ratio = pd.Series(0.0, index=df.index)

# Base Canary
cash_ratio[any_canary_bad] = 0.10
cash_ratio[all_canary_bad] = 0.25

# Overlay Price Tunnel Crisis
# If Crisis (State 1), Force 50% Cash.
# Unless Canary wants MORE cash? (e.g. if Canary max was 70%, we'd take max).
# But here Canary max is 25%. So 50% is dominant.
is_crisis = price_tunnel_crisis == 1
cash_ratio[is_crisis] = 0.50

# Calculate Metrics
target_ret = df["Price"].pct_change().fillna(0)
pos_size = 1 - cash_ratio.shift(1).fillna(0)  # Assume fully invested at start or 0 cash
strat_ret = target_ret * pos_size
strat_cum = (1 + strat_ret).cumprod()
target_cum = (1 + target_ret).cumprod()

# Rebase
strat_cum = strat_cum / strat_cum.iloc[0]
target_cum = target_cum / target_cum.iloc[0]

# ---------------------------------------------------------
# 4. Visualization & Stats
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))

# Top: Equity Curve
plt.subplot(2, 1, 1)
plt.plot(target_cum, label="Buy & Hold (SPY)", color="gray", alpha=0.5)
plt.plot(strat_cum, label="Double-Key V2 (3-Day Confirm)", color="purple", linewidth=2)
plt.title("Double-Key V2 Performance")
plt.legend()
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.3)

# Bottom: Cash Ratio
plt.subplot(2, 1, 2)
plt.plot(cash_ratio * 100, label="Cash Ratio (%)", color="orange")
plt.fill_between(cash_ratio.index, 0, cash_ratio * 100, color="orange", alpha=0.3)
plt.title("Cash Ratio (Risk Management)")
plt.ylabel("Cash %")
plt.ylim(0, 55)
plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("d:/gg/data/backtest_results/double_key_v2_result.png")


# Stats
def calc_mdd(series):
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max
    return dd.min()


days = len(strat_cum)
years = days / 252.0
cagr_strat = (strat_cum.iloc[-1] / strat_cum.iloc[0]) ** (1 / years) - 1
mdd_strat = calc_mdd(strat_cum)

cagr_bh = (target_cum.iloc[-1] / target_cum.iloc[0]) ** (1 / years) - 1
mdd_bh = calc_mdd(target_cum)

print("\n--- Performance Summary (V2: 3-Day Confirm) ---")
print(f"Strategy CAGR: {cagr_strat * 100:.2f}% | MDD: {mdd_strat * 100:.2f}%")
print(f"B & H    CAGR: {cagr_bh * 100:.2f}% | MDD: {mdd_bh * 100:.2f}%")
print(f"Defense Improvement: MDD reduced by {abs(mdd_bh - mdd_strat) * 100:.2f}%p")
print(f"Latest Cash Ratio: {cash_ratio.iloc[-1] * 100:.0f}%")

plt.show()
