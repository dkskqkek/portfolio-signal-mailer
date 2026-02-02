import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Data Download
# ---------------------------------------------------------
# Assets for all strategies
# Buffer: SPY
# PAA: SPY, VEA, VWO, BND (Nasdaq/Advanced/Emerging/Bond)
# LAA: IWD, IEF, GLD, SHY (LargeVal/Treasury/Gold/ShortTerm) + SPY(Signal)
all_symbols = list(set(["SPY", "VEA", "VWO", "BND", "IWD", "IEF", "GLD", "SHY"]))
print(f"Downloading data for: {all_symbols}")
data = yf.download(all_symbols, start="2010-01-01", progress=False)

# Fill NA for safe backtesting (forward fill)
# Robust Data Access
if "Adj Close" in data.columns:
    data = data["Adj Close"]
elif "Close" in data.columns:
    print("'Adj Close' not found, using 'Close'")
    data = data["Close"]
else:
    # Handle MultiIndex
    try:
        # Check if levels exist
        if isinstance(data.columns, pd.MultiIndex):
            # Try level 0 for Price Type
            if "Adj Close" in data.columns.get_level_values(0):
                data = data.xs("Adj Close", level=0, axis=1)
            elif "Close" in data.columns.get_level_values(0):
                data = data.xs("Close", level=0, axis=1)
    except:
        pass

data = data.ffill()


# ---------------------------------------------------------
# 2. Strategy Logic
# ---------------------------------------------------------

# Common helper: Daily Returns
returns = data.pct_change()


# --- A. Buffer Strategy (SPY Timing) ---
# Rule: Hold SPY if Price > 185 MA * 1.03 (simplified hysteresis)
# To be robust: Using the exact logic from previous `paa_buffer_check.py`
def run_buffer(price_series):
    ma = price_series.rolling(window=185).mean()
    buffer = 0.03
    signal = np.zeros(len(price_series))
    current_state = 1

    # Loop for hysteresis
    price_val = price_series.values
    ma_val = ma.values

    for i in range(len(price_series)):
        if np.isnan(ma_val[i]):
            signal[i] = (
                1  # Default to invested at start? Or 0. Let's say 1 for benchmark.
            )
            continue

        upper = ma_val[i] * (1 + buffer)
        lower = ma_val[i] * (1 - buffer)

        if price_val[i] > upper:
            current_state = 1
        elif price_val[i] < lower:
            current_state = 0

        signal[i] = current_state

    return pd.Series(signal, index=price_series.index)


buffer_signal = run_buffer(data["SPY"])
# Buffer Returns: Invest in SPY when Signal=1, else Cash (SHY or 0%)
# Assuming Cash = SHY for fair comparison with LAA
buffer_equity = (
    returns["SPY"] * buffer_signal.shift(1)
    + returns["SHY"] * (1 - buffer_signal.shift(1))
).fillna(0)
buffer_cum = (1 + buffer_equity).cumprod()

# --- B. PAA Strategy (Protective Asset Allocation) ---
# Universe: SPY, VEA, VWO, BND
# Rule: Calculate 12m momentum. Breadth = 'Good' assets count.
# Capital Allocation: (Breadth / 4) to Risk Assets, rest to Safe (IEF/SHY).
# Risk Assets Selection: Simple PAA invests in ALL with positive momentum? Or Top N?
# Original PAA: "Vigilant" style?
# Impl: Equal weight to 'Good' assets?
# Let's use the USER's PAA logic: "Breadth < 2 -> Cash, else Invest" (Binary)
# USER Logic re-use: `paa_signal = np.where(breadth < (total_assets / 2), 0, 1)`
# This treats PAA as a "Timing Signal" for the portfolio.
# We will apply this signal to an Equal Weight Portfolio of [SPY, VEA, VWO, BND] OR just SPY?
# Given User's previous code compared PAA signal vs Buffer Signal on SPY chart,
# likely they treat PAA as a regime filter for their main portfolio.
# BUT LAA is definitely a Portfolio strategy.
# DECISION: Implement "Portfolio PAA".
# If Breadth is good, hold [SPY, VEA, VWO, BND]. If bad, hold [IEF/SHY].
paa_assets = ["SPY", "VEA", "VWO", "BND"]
paa_ma = data[paa_assets].pct_change(252)  # 12-month momentum approximation
breadth = (paa_ma > 0).sum(axis=1)
pct_risk = (
    breadth / 4.0
)  # Standard PAA: Weight to Risk Assets depends on breadth (High breadth -> High risk)
# Wait, user's logic was Binary. Let's stick to "Standard PAA" which is better (Generalized).
# If Breadth=4 -> 100% Risk. Breadth=0 -> 100% Cash.
# Risk Allocation: Equal weight among the 4 assets? Or just the positive ones?
# Standard PAA (Keller): Average momentum of universe.
# Let's implement a simple "Generalized PAA":
# Valid Risk Assets = those with mom > 0.
# Weight per asset = 1/4 * Momentum_Protection_Factor?
# Let's do:
# Risk Portion = (Breadth / 4). Distributed equally to [SPY, VEA, VWO, BND].
# Safe Portion = 1 - Risk Portion. Invested in IEF (or SHY).
# This is "PAA-GW" (Generalized Width).
r_paa = returns[paa_assets].mean(axis=1)  # Return of EW Risk Basket
r_safe = returns["IEF"]  # Safe asset
# dynamic weight
w_risk = breadth / 4.0
w_safe = 1 - w_risk
paa_equity = (r_paa * w_risk.shift(1) + r_safe * w_safe.shift(1)).fillna(0)
paa_cum = (1 + paa_equity).cumprod()

# --- C. LAA Strategy (Lethargic Asset Allocation) ---
# Permanent: 25% IWD, 25% IEF, 25% GLD, 25% SHY.
# Timing: Checks Unemployment Rate (we use SPY < 200MA proxy).
# If SPY < 200MA, Swap IWD -> SHY.
spy_ma200 = data["SPY"].rolling(200).mean()
laa_condition = data["SPY"] > spy_ma200
# Weights matrix
laa_w_iwd = np.where(laa_condition, 0.25, 0.0)
laa_w_shy = np.where(laa_condition, 0.25, 0.50)  # 25% fixed + 25% from IWD if Bear
laa_w_ief = 0.25
laa_w_gld = 0.25

laa_daily_ret = (
    returns["IWD"] * pd.Series(laa_w_iwd, index=data.index).shift(1)
    + returns["SHY"] * pd.Series(laa_w_shy, index=data.index).shift(1)
    + returns["IEF"] * laa_w_ief
    + returns["GLD"] * laa_w_gld
).fillna(0)
laa_cum = (1 + laa_daily_ret).cumprod()


# ---------------------------------------------------------
# 3. Visualization & Stats
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
# Normalize to 100
(buffer_cum * 100).plot(label="Buffer (SPY Timing)", color="red", alpha=0.7)
(paa_cum * 100).plot(label="PAA (Dynamic AA)", color="blue", alpha=0.7)
(laa_cum * 100).plot(label="LAA (Static AA + Timing)", color="green", linewidth=2)
plt.title("Strategy Comparison: LAA vs PAA vs Buffer (2010 ~ )")
plt.ylabel("Normalized Return (Start=100)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("d:/gg/data/backtest_results/strategy_comparison.png")


# Stats Table
def get_stats(equity):
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25
    cagr = (1 + total_ret) ** (1 / years) - 1

    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    mdd = dd.min()

    daily_ret = equity.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (cagr - 0.03) / vol  # Risk free 3%

    return cagr, mdd, sharpe


stats = pd.DataFrame(columns=["CAGR", "MDD", "Sharpe"])
for name, ser in [("Buffer", buffer_cum), ("PAA", paa_cum), ("LAA", laa_cum)]:
    c, m, s = get_stats(ser)
    stats.loc[name] = [f"{c * 100:.2f}%", f"{m * 100:.2f}%", f"{s:.2f}"]

print("\n--- Strategy Performance Comparison ---")
print(stats)
print("\n[Note] LAA uses SPY < 200MA as a proxy for Unemployment Rate signal.")
