import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Data Download
# ---------------------------------------------------------
target_asset = "SPY"
canary_assets = ["VWO", "BND"]
defensive_asset = "SHY"
risk_index = "^VIX"

all_symbols = [target_asset] + canary_assets + [defensive_asset, risk_index]

print(f"Downloading data for V4: {all_symbols}")
start_date = "2008-01-01"
data = yf.download(all_symbols, start=start_date, progress=False)

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
# 2. Indicator Calculation & Scoring Components
# ---------------------------------------------------------

# A. Trend Score (Max 40)
# Condition: Price > MA185 (with 3% buffer logic implicit? Let's use clean MA185 for score simplicity, or V3 buffer?)
# Plan said: "SPY Price > 185 MA (with 3% buffer applied)."
# Let's reuse V3 Hysteresis Logic for the "Trend Status".
ma_window = 185
ma185 = data[target_asset].rolling(window=ma_window).mean()
buffer_exit_line = ma185 * 0.97
price = data[target_asset]

# B. Macro Score (Max 20)
# VWO Mom > 0 (+10), BND Mom > 0 (+10)
momentum_window = 252
canary_mom = data[canary_assets].pct_change(momentum_window)

# C. VIX Score (Max 40)
vix = data[risk_index]

# ---------------------------------------------------------
# 3. Strategy Logic
# ---------------------------------------------------------

df = pd.DataFrame(
    {
        "Price": price,
        "MA185": ma185,
        "ExitLine": buffer_exit_line,
        "VWO_Mom": canary_mom["VWO"],
        "BND_Mom": canary_mom["BND"],
        "VIX": vix,
        "Target_Ret": data[target_asset].pct_change(),
        "Cash_Ret": data[defensive_asset].pct_change(),
    }
).dropna()

# --- Component 1: Trend Status (Hysteresis) ---
# Same as V3: 0 = Bull (Safe), 1 = Bear (Crisis)
# If 0 (Bull) -> +40 points. If 1 (Bear) -> 0 points.
trend_crisis_state = np.zeros(len(df), dtype=int)
current_state = 0
days_above_ma = 0
price_vals = df["Price"].values
ma_vals = df["MA185"].values
exit_vals = df["ExitLine"].values

for i in range(len(df)):
    p = price_vals[i]
    ma = ma_vals[i]
    exit_l = exit_vals[i]

    if current_state == 0:  # Bull
        if p < exit_l:
            current_state = 1
            days_above_ma = 0
    else:  # Bear
        if p > ma:
            days_above_ma += 1
        else:
            days_above_ma = 0

        if days_above_ma >= 3:
            current_state = 0

    trend_crisis_state[i] = current_state

# Score A: Trend
score_trend = pd.Series(0, index=df.index)
score_trend[trend_crisis_state == 0] = 40  # Bullish

# Score B: Macro
score_macro = pd.Series(0, index=df.index)
score_macro += np.where(df["VWO_Mom"] > 0, 10, 0)
score_macro += np.where(df["BND_Mom"] > 0, 10, 0)

# Score C: VIX (Gradual)
# < 15: 40
# 15-20: 30
# 20-25: 20
# 25-30: 10
# >= 30: 0
score_vix = pd.Series(0, index=df.index)
vix_vals = df["VIX"].values
vix_points = np.zeros(len(df))

# Bucketing using numpy select or conditions
# Defaults to 0 (>= 30)
conditions = [
    (vix_vals < 15),
    (vix_vals >= 15) & (vix_vals < 20),
    (vix_vals >= 20) & (vix_vals < 25),
    (vix_vals >= 25) & (vix_vals < 30),
]
choices = [40, 30, 20, 10]
score_vix = np.select(conditions, choices, default=0)  # >= 30 gets 0

# Total Score
total_score = score_trend + score_macro + score_vix

# Cash Ladder Mapping
# 80-100: 0% Cash
# 60-79: 20%
# 40-59: 50%
# 20-39: 70%
# 0-19: 100%

cash_weight = np.zeros(len(df))
# Using numpy select again for speed/clarity
cond_cash = [
    (total_score >= 80),
    (total_score >= 60) & (total_score < 80),
    (total_score >= 40) & (total_score < 60),
    (total_score >= 20) & (total_score < 40),
    (total_score < 20),
]
choice_cash = [0.0, 0.2, 0.5, 0.7, 1.0]
cash_weight = np.select(cond_cash, choice_cash, default=1.0)

cash_ratio_v4 = pd.Series(cash_weight, index=df.index)

# ---------------------------------------------------------
# 4. Performance Calculation (V3 vs V4)
# ---------------------------------------------------------
# V3 Logic (Re-implemented for direct comparison in this script)
# V3: (TrendState==1 -> 50%) OR (VIX>30 -> 50%). Else Canary (0/10/25).
# Base Canary V3
base_cash_v3 = pd.Series(0.0, index=df.index)
any_bad = (df["VWO_Mom"] <= 0) | (df["BND_Mom"] <= 0)
all_bad = (df["VWO_Mom"] <= 0) & (df["BND_Mom"] <= 0)
base_cash_v3[any_bad] = 0.10
base_cash_v3[all_bad] = 0.25

cash_ratio_v3 = base_cash_v3.copy()
is_tunnel_crisis = trend_crisis_state == 1
is_vix_panic = df["VIX"] > 30
cash_ratio_v3[is_tunnel_crisis] = 0.50
cash_ratio_v3[is_vix_panic] = 0.50

# Calculate Returns
# V3
pos_v3 = 1 - cash_ratio_v3.shift(1).fillna(0)
cash_v3 = cash_ratio_v3.shift(1).fillna(0)
ret_v3 = df["Target_Ret"] * pos_v3 + df["Cash_Ret"] * cash_v3
cum_v3 = (1 + ret_v3).cumprod()

# V4
pos_v4 = 1 - cash_ratio_v4.shift(1).fillna(0)
cash_v4 = cash_ratio_v4.shift(1).fillna(0)
ret_v4 = df["Target_Ret"] * pos_v4 + df["Cash_Ret"] * cash_v4
cum_v4 = (1 + ret_v4).cumprod()

# Buy & Hold
cum_bh = (1 + df["Target_Ret"]).cumprod()

# Rebase
cum_v3 /= cum_v3.iloc[0]
cum_v4 /= cum_v4.iloc[0]
cum_bh /= cum_bh.iloc[0]


# Metrics Function
def calc_metrics(series, name):
    total_ret = series.iloc[-1] - 1
    days = (series.index[-1] - series.index[0]).days
    cagr = (series.iloc[-1]) ** (365.25 / days) - 1
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max
    mdd = dd.min()
    daily_rets = series.pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    print(
        f"[{name}] CAGR: {cagr * 100:.2f}% | MDD: {mdd * 100:.2f}% | Sharpe: {sharpe:.2f}"
    )
    return cagr, mdd, sharpe


print("\n--- Strategy Performance (2008-2024) ---")
with open("v4_metrics.txt", "w") as f:
    c, m, s = calc_metrics(cum_bh, "Buy & Hold")
    f.write(f"Buy & Hold,{c * 100:.2f},{m * 100:.2f},{s:.2f}\n")
    c, m, s = calc_metrics(cum_v3, "V3 (Binary Logic)")
    f.write(f"V3 (Binary),{c * 100:.2f},{m * 100:.2f},{s:.2f}\n")
    c, m, s = calc_metrics(cum_v4, "V4 (Score Logic)")
    f.write(f"V4 (Score),{c * 100:.2f},{m * 100:.2f},{s:.2f}\n")

# ---------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------
plt.figure(figsize=(14, 10))

# Equity Curve
plt.subplot(3, 1, 1)
plt.plot(cum_bh, label="Buy & Hold", color="gray", alpha=0.3)
plt.plot(cum_v3, label="V3 (Binary)", color="blue", linewidth=1.5, alpha=0.7)
plt.plot(cum_v4, label="V4 (Score)", color="green", linewidth=2.0)
plt.title("V3 vs V4: Performance Comparison")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)

# Cash Weights Comparison
plt.subplot(3, 1, 2)
plt.plot(cash_ratio_v3 * 100, label="V3 Cash %", color="blue", alpha=0.4)
plt.plot(cash_ratio_v4 * 100, label="V4 Cash %", color="green", alpha=0.6)
plt.ylabel("Cash %")
plt.title("Cash Allocation Dynamics")
plt.legend()
plt.grid(True, alpha=0.3)

# Score Components (Just total score for V4)
plt.subplot(3, 1, 3)
combo_score = pd.Series(total_score, index=df.index)
plt.plot(combo_score, label="V4 Safety Score (0-100)", color="purple")
plt.axhline(60, color="orange", linestyle="--", label="Start Defensive (<60)")
plt.axhline(20, color="red", linestyle="--", label="Max Defensive (<20)")
plt.title("V4 Risk Score History")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("v3_vs_v4_comparison.png")
print("Comparison chart saved to v3_vs_v4_comparison.png")
