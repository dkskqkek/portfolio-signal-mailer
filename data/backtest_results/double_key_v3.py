import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Data Download
# ---------------------------------------------------------
target_asset = "SPY"
canary_assets = ["VWO", "BND"]
defensive_asset = "SHY"  # Smart Cash
risk_index = "^VIX"

all_symbols = [target_asset] + canary_assets + [defensive_asset, risk_index]

print(f"Downloading data for V3: {all_symbols}")
# 2008 for VIX/Crisis testing
# SHY inception was 2002, VIX history long enough.
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
# 2. Indicator Calculation
# ---------------------------------------------------------

# A. Key 1: Canary Momentum (12-month)
momentum_window = 252
canary_mom = data[canary_assets].pct_change(momentum_window)

# B. Key 2: Price Tunnel components
ma_window = 185
ma185 = data[target_asset].rolling(window=ma_window).mean()
buffer_exit_line = ma185 * 0.97
price = data[target_asset]

# C. VIX Filter (V3 Exclusive)
vix = data[risk_index]

# ---------------------------------------------------------
# 3. Strategy Logic (Double-Key V2 -> V3)
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
        "Cash_Ret": data[defensive_asset].pct_change(),  # Smart Cash Return
    }
).dropna()

# Vectorized Canary Status
vwo_bad = df["VWO_Mom"] <= 0
bnd_bad = df["BND_Mom"] <= 0
any_canary_bad = vwo_bad | bnd_bad
all_canary_bad = vwo_bad & bnd_bad

# State Iteration for Price Tunnel (Hysteresis)
# State 0: Normal/Bullish (Invest)
# State 1: Crisis (Exit/Defensive)
price_tunnel_crisis = np.zeros(len(df), dtype=int)
current_crisis_state = 0
days_above_ma = 0

price_vals = df["Price"].values
ma_vals = df["MA185"].values
exit_vals = df["ExitLine"].values

for i in range(len(df)):
    p = price_vals[i]
    ma = ma_vals[i]
    exit_l = exit_vals[i]

    # 1. Check Exit Trigger
    if current_crisis_state == 0:
        if p < exit_l:
            current_crisis_state = 1  # Enter Crisis
            days_above_ma = 0

    # 2. Check Re-entry Trigger
    else:  # current_crisis_state == 1
        if p > ma:
            days_above_ma += 1
        else:
            days_above_ma = 0

        if days_above_ma >= 3:
            current_crisis_state = 0  # Back to Normal

    price_tunnel_crisis[i] = current_crisis_state

# Cash Ratio Calculation
# 1. Base Canary

# Wait! current V3 logic in script combined VIX logic into cash_ratio.
# So 'pos_size' above ALREADY includes VIX logic if I didn't separate them?
# Let's check lines 117-128.
# v2 logic ended at line 128 (Tunnel Crisis).
# VIX logic was added at 136.
# I need to separate `cash_ratio_v2` and `cash_ratio_v3`.

# Re-calculating correctly for comparison
# 1. Base Canary
base_cash = pd.Series(0.0, index=df.index)
base_cash[any_canary_bad] = 0.10
base_cash[all_canary_bad] = 0.25

# 2. V2 Final (Tunnel Only)
cash_ratio_v2 = base_cash.copy()
is_tunnel_crisis = price_tunnel_crisis == 1
cash_ratio_v2[is_tunnel_crisis] = 0.50

# 3. V3 Final (Tunnel + VIX)
cash_ratio_v3 = cash_ratio_v2.copy()
is_vix_panic = df["VIX"] > 30
cash_ratio_v3[is_vix_panic] = 0.50

# Positions
pos_v2 = 1 - cash_ratio_v2.shift(1).fillna(0)
cash_v2 = cash_ratio_v2.shift(1).fillna(0)

pos_v3 = 1 - cash_ratio_v3.shift(1).fillna(0)
cash_v3 = cash_ratio_v3.shift(1).fillna(0)

# Returns
# A. V2 Classic (Zero Cash)
v2_classic_ret = df["Target_Ret"] * pos_v2
v2_classic_cum = (1 + v2_classic_ret).cumprod()

# B. V2 Smart (Smart Cash) -> The "Interest" Effect
v2_smart_ret = df["Target_Ret"] * pos_v2 + df["Cash_Ret"] * cash_v2
v2_smart_cum = (1 + v2_smart_ret).cumprod()

# C. V3 Complete (Smart Cash + VIX) -> The "Timing" Effect added
v3_ret = df["Target_Ret"] * pos_v3 + df["Cash_Ret"] * cash_v3
v3_cum = (1 + v3_ret).cumprod()

# Buy & Hold
bh_cum = (1 + df["Target_Ret"]).cumprod()

# Normalize start
v2_classic_cum /= v2_classic_cum.iloc[0]
v2_smart_cum /= v2_smart_cum.iloc[0]
v3_cum /= v3_cum.iloc[0]
bh_cum /= bh_cum.iloc[0]


# Metrics
def calc_metrics(series, name):
    total_ret = series.iloc[-1] - 1
    days = (series.index[-1] - series.index[0]).days
    cagr = (series.iloc[-1]) ** (365.25 / days) - 1

    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max
    mdd = dd.min()

    # Sharpe (heuristic using daily returns)
    daily_rets = series.pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)

    metric_str = f"[{name}] CAGR: {cagr * 100:.2f}% | MDD: {mdd * 100:.2f}% | Sharpe: {sharpe:.2f}"
    print(metric_str)
    with open("metrics.txt", "a") as f:
        f.write(metric_str + "\n")
    return cagr, mdd, sharpe


# Clear file
with open("metrics.txt", "w") as f:
    f.write("--- Strategy Performance ---\n")

# D. Golden Cross (50/200) Comparison
ma50 = df["Price"].rolling(50).mean()
ma200 = df["Price"].rolling(200).mean()
# Signal: 50 > 200
gc_pos = (ma50 > ma200).astype(int).shift(1).fillna(0)
gc_ret = (
    df["Target_Ret"] * gc_pos
)  # Zero Cash return for fair comparison (or use Smart Cash?)
# Let's use Smart Cash for fair comparison with V3
gc_ret = df["Target_Ret"] * gc_pos + df["Cash_Ret"] * (1 - gc_pos)
gc_cum = (1 + gc_ret).cumprod()
gc_cum /= gc_cum.iloc[0]

print("\n--- Strategy Performance (2008-2024) ---")
calc_metrics(bh_cum, "Buy & Hold")
calc_metrics(v2_classic_cum, "V2 (Zero Cash)")
calc_metrics(v3_cum, "V3 (Smart Cash + VIX)")
calc_metrics(gc_cum, "Golden Club (50/200)")  # Classic GC

# ---------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------
plt.figure(figsize=(12, 10))

# Equity Curve
plt.subplot(3, 1, 1)
plt.plot(bh_cum, label="Buy & Hold", color="gray", alpha=0.3)
plt.plot(v2_classic_cum, label="V2 (Zero Cash)", color="purple", linestyle="--")
plt.plot(v3_cum, label="V3 (Smart Cash + VIX)", color="blue", linewidth=2)
plt.plot(gc_cum, label="Golden Cross (50/200)", color="orange", linewidth=1.5)
plt.title("Performance: V3 vs Golden Cross")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)

# Cash Ratio
plt.subplot(3, 1, 2)
plt.plot(cash_ratio_v3 * 100, label="Cash Allocation % (V3)", color="green", alpha=0.6)
plt.ylabel("Cash %")
plt.title("Dynamic Cash Allocation")
plt.grid(True, alpha=0.3)

# VIX & Triggers
plt.subplot(3, 1, 3)
plt.plot(df["VIX"], label="VIX", color="red", alpha=0.6)
plt.axhline(30, color="black", linestyle="--", label="Panic Threshold (30)")
plt.title("VIX Panic Monitor")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("double_key_v3_result.png")

# ---------------------------------------------------------
# 6. Signal Visualization (Requested by User)
# ---------------------------------------------------------
plt.figure(figsize=(14, 8))
plt.plot(df.index, df["Price"], label="SPY Price", color="black", alpha=0.5)

# Detect Sells (Cash Ratio Increasing)
# V2 Sells
v2_sell_mask = (cash_ratio_v2 > cash_ratio_v2.shift(1)) & (cash_ratio_v2 > 0)
v2_sells = df.loc[v2_sell_mask]

# V3 Sells
v3_sell_mask = (cash_ratio_v3 > cash_ratio_v3.shift(1)) & (cash_ratio_v3 > 0)
v3_sells = df.loc[v3_sell_mask]

# Plot
plt.scatter(
    v2_sells.index,
    v2_sells["Price"],
    color="purple",
    marker="v",
    s=100,
    label="V2 Sell (Tunnel)",
    zorder=5,
)
plt.scatter(
    v3_sells.index,
    v3_sells["Price"],
    color="blue",
    marker="x",
    s=100,
    label="V3 Sell (Tunnel+VIX)",
    zorder=6,
)

# Highlight VIX Triggers specifically
# VIX Sells are where V3 sold but V2 didn't (or V3 sold earlier/harder due to VIX)
# Or simply where VIX > 30 triggered the state.
vix_triggers = df.loc[is_vix_panic & (is_vix_panic != is_vix_panic.shift(1))]
plt.scatter(
    vix_triggers.index,
    vix_triggers["Price"],
    color="red",
    marker="*",
    s=200,
    label="VIX Panic Trigger (>30)",
    zorder=10,
)

plt.title("Sell Signal Comparison: V2 vs V3")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("signal_comparison.png")
print("Signal chart saved to signal_comparison.png")

# ---------------------------------------------------------
# 7. Zoomed Visualization (2023)
# ---------------------------------------------------------
plt.figure(figsize=(14, 8))
# Filter for 2023
zoom_start = "2023-01-01"
zoom_end = "2023-12-31"
df_zoom = df.loc[zoom_start:zoom_end]
v2_sells_zoom = v2_sells.loc[zoom_start:zoom_end]
v3_sells_zoom = v3_sells.loc[zoom_start:zoom_end]
vix_triggers_zoom = vix_triggers.loc[zoom_start:zoom_end]

plt.plot(df_zoom.index, df_zoom["Price"], label="SPY Price", color="black", alpha=0.5)

# Plot Zoomed Markers
if not v2_sells_zoom.empty:
    plt.scatter(
        v2_sells_zoom.index,
        v2_sells_zoom["Price"],
        color="purple",
        marker="v",
        s=150,
        label="V2 Sell",
        zorder=5,
    )
if not v3_sells_zoom.empty:
    plt.scatter(
        v3_sells_zoom.index,
        v3_sells_zoom["Price"],
        color="blue",
        marker="x",
        s=150,
        label="V3 Sell",
        zorder=6,
    )
if not vix_triggers_zoom.empty:
    plt.scatter(
        vix_triggers_zoom.index,
        vix_triggers_zoom["Price"],
        color="red",
        marker="*",
        s=250,
        label="VIX Panic",
        zorder=10,
    )

plt.title("Signal Comparison: 2023 Zoom")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("signal_comparison_2023.png")
# ---------------------------------------------------------
# 8. Divergence Analysis (Finding the "Alpha" moments)
# ---------------------------------------------------------
print("\n--- Divergence Analysis (Where V3 differed from V2) ---")
# Divergence = Cash Ratio Different
divergence_mask = cash_ratio_v3 != cash_ratio_v2
diff_dates = df.index[divergence_mask]

if not diff_dates.empty:
    print(f"Total Divergence Days: {len(diff_dates)}")
    # Group into periods
    # Simple trick: check if day diff > 1
    diff_days = diff_dates.to_series().diff().dt.days
    new_periods = diff_days > 1
    period_starts = diff_dates[new_periods | (diff_days.isna())]

    print("Top 5 Divergence Periods (VIX Triggered):")
    for start_date in period_starts[:5]:  # Show first 5 or relevant ones
        # Find end of this block
        # Start searching from start_date
        block = df.loc[start_date:][divergence_mask].head(1)
        # Actually easier to just iterate and print blocks
        pass

    # Just save the data to csv for manual verification if needed
    div_df = df.loc[divergence_mask, ["Price", "VIX", "MA185"]]
    div_df["Cash_V2"] = cash_ratio_v2.loc[divergence_mask]
    div_df["Cash_V3"] = cash_ratio_v3.loc[divergence_mask]
    print(div_df.head(10))
    print("...")
    print(div_df.tail(10))

    # Generate Chart for the MOST SIGNIFICANT Divergence (likely 2020)
    # Check 2020-02 to 2020-04
    zoom_dates = ["2020-01-01", "2020-06-30"]
    mask_2020 = (df.index >= zoom_dates[0]) & (df.index <= zoom_dates[1])

    if mask_2020.any():
        plt.figure(figsize=(14, 8))
        df_2020 = df.loc[mask_2020]
        c_v2_2020 = cash_ratio_v2.loc[mask_2020]
        c_v3_2020 = cash_ratio_v3.loc[mask_2020]

        plt.subplot(2, 1, 1)
        plt.plot(df_2020.index, df_2020["Price"], label="Price", color="black")
        plt.plot(
            df_2020.index, df_2020["MA185"], label="MA185", color="gray", linestyle="--"
        )

        # Highlight Divergence Zone
        div_2020 = c_v3_2020 > c_v2_2020
        plt.fill_between(
            df_2020.index,
            df_2020["Price"].min(),
            df_2020["Price"].max(),
            where=div_2020,
            color="red",
            alpha=0.1,
            label="VIX Defense Active",
        )

        plt.title("2020 Covid Crash: V2 vs V3 (VIX Trigger)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(df_2020.index, df_2020["VIX"], label="VIX", color="red")
        plt.axhline(30, color="black", linestyle="--")
        plt.title("VIX Level")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("signal_divergence_2020.png")
        # ---------------------------------------------------------
# 9. Comparison: V3 vs Pure MA185 (User Request)
# ---------------------------------------------------------
print("\n--- Generating V3 vs MA185 Comparison Charts (2020, 2024) ---")

# Pure MA185 Logic
# Use 1-day lag for signal to avoid lookahead, similar to V3
pure_ma_pos = (df["Price"] > df["MA185"]).astype(int)
pure_ma_pos = pure_ma_pos.shift(1).fillna(0)

# Detect Signals
# Buy: 0 -> 1, Sell: 1 -> 0
pure_ma_buys = df.loc[(pure_ma_pos == 1) & (pure_ma_pos.shift(1) == 0)]
pure_ma_sells = df.loc[(pure_ma_pos == 0) & (pure_ma_pos.shift(1) == 1)]

# V3 Signals (already have Sell masks, need Buys)
# V3 Buys: Cash Ratio Decreases (Investment Increases).
# Assuming "Full Equity" is optimal, any decrease in cash is a "Buy"
# Or specifically when transitioning from >0 cash to 0 cash (Full Entry)
# Let's simplify: "Buy" = Cash Ratio decreases. "Sell" = Cash Ratio increases.
v3_buy_mask = cash_ratio_v3 < cash_ratio_v3.shift(1)
# Filter trivial changes? No, catching all is fine.
v3_buys = df.loc[v3_buy_mask]
# v3_sells already defined


def plot_yearly_comparison(target_year, filename):
    start = f"{target_year}-01-01"
    end = f"{target_year}-12-31"
    sub_df = df.loc[start:end]

    if sub_df.empty:
        print(f"No data for {target_year}")
        return

    sub_ma_buys = pure_ma_buys.loc[start:end]
    sub_ma_sells = pure_ma_sells.loc[start:end]
    sub_v3_buys = v3_buys.loc[start:end]
    sub_v3_sells = v3_sells.loc[start:end]

    plt.figure(figsize=(14, 8))

    # Price & MA
    plt.plot(sub_df.index, sub_df["Price"], label="Price", color="black", alpha=0.6)
    plt.plot(
        sub_df.index, sub_df["MA185"], label="MA 185", color="orange", linestyle="--"
    )

    # Pure MA185 Signals (Small markers)
    if not sub_ma_buys.empty:
        plt.scatter(
            sub_ma_buys.index,
            sub_ma_buys["Price"],
            marker="^",
            color="green",
            s=80,
            label="MA185 Buy",
            alpha=0.6,
        )
    if not sub_ma_sells.empty:
        plt.scatter(
            sub_ma_sells.index,
            sub_ma_sells["Price"],
            marker="v",
            color="red",
            s=80,
            label="MA185 Sell",
            alpha=0.6,
        )

    # V3 Signals (Large markers, Bold colors)
    if not sub_v3_buys.empty:
        plt.scatter(
            sub_v3_buys.index,
            sub_v3_buys["Price"] * 1.02,
            marker="P",
            color="blue",
            s=150,
            label="V3 Buy (Smart)",
            zorder=10,
        )
    if not sub_v3_sells.empty:
        plt.scatter(
            sub_v3_sells.index,
            sub_v3_sells["Price"] * 1.02,
            marker="X",
            color="purple",
            s=150,
            label="V3 Sell (Defensive)",
            zorder=10,
        )

    # Highlight VIX Triggers if any
    sub_vix = vix_triggers.loc[start:end]
    if not sub_vix.empty:
        plt.scatter(
            sub_vix.index,
            sub_vix["Price"] * 1.04,
            marker="*",
            color="red",
            s=300,
            label="V3 Panic (VIX > 30)",
            zorder=15,
        )

    plt.title(f"{target_year} Comparison: Pure MA185 vs V3 (Double-Key)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")


plot_yearly_comparison(2020, "comparison_2020.png")
plot_yearly_comparison(2024, "comparison_2024.png")


# 7-Year Full Cycle Comparison (2018-2024)
def plot_period_comparison(start_year, end_year, filename):
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"
    sub_df = df.loc[start:end]

    if sub_df.empty:
        return

    sub_ma_buys = pure_ma_buys.loc[start:end]
    sub_ma_sells = pure_ma_sells.loc[start:end]
    sub_v3_buys = v3_buys.loc[start:end]
    sub_v3_sells = v3_sells.loc[start:end]
    sub_vix = vix_triggers.loc[start:end]

    plt.figure(figsize=(16, 9))

    # Price & MA
    plt.plot(
        sub_df.index,
        sub_df["Price"],
        label="Price",
        color="black",
        alpha=0.6,
        linewidth=1,
    )
    plt.plot(
        sub_df.index,
        sub_df["MA185"],
        label="MA 185",
        color="orange",
        linestyle="--",
        linewidth=1,
    )

    # Markers (Adjust size for longer timeframe)
    # Pure MA185
    if not sub_ma_buys.empty:
        plt.scatter(
            sub_ma_buys.index,
            sub_ma_buys["Price"],
            marker="^",
            color="green",
            s=50,
            label="MA185 Buy",
            alpha=0.5,
        )
    if not sub_ma_sells.empty:
        plt.scatter(
            sub_ma_sells.index,
            sub_ma_sells["Price"],
            marker="v",
            color="red",
            s=50,
            label="MA185 Sell",
            alpha=0.5,
        )

    # V3
    if not sub_v3_buys.empty:
        plt.scatter(
            sub_v3_buys.index,
            sub_v3_buys["Price"] * 1.02,
            marker="P",
            color="blue",
            s=100,
            label="V3 Buy",
            zorder=10,
        )
    if not sub_v3_sells.empty:
        plt.scatter(
            sub_v3_sells.index,
            sub_v3_sells["Price"] * 1.02,
            marker="X",
            color="purple",
            s=100,
            label="V3 Sell",
            zorder=10,
        )

    # VIX Triggers
    if not sub_vix.empty:
        plt.scatter(
            sub_vix.index,
            sub_vix["Price"] * 1.05,
            marker="*",
            color="red",
            s=200,
            label="V3 Panic (VIX > 30)",
            zorder=20,
        )

    plt.title(f"{start_year}-{end_year} Full Cycle Comparison: Pure MA185 vs V3")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")


plot_period_comparison(2018, 2024, "comparison_7yr.png")

# 10. VIX Trigger Analysis (User Concern: Too frequent?)
print("\n--- VIX > 30 Trigger Analysis ---")
vix_panic_days = df[df["VIX"] > 30].index
print(f"Total Trading Days: {len(df)}")
print(
    f"VIX > 30 Days: {len(vix_panic_days)} ({len(vix_panic_days) / len(df) * 100:.2f}%)"
)

# Group into distinct periods (consecutive days)
vix_series = (df["VIX"] > 30).astype(int)
vix_starts = vix_series[(vix_series == 1) & (vix_series.shift(1) == 0)].index

print(f"Distinct Panic Events: {len(vix_starts)}")
# 11. "Stupid Sell" Analysis (Selling above MA185)
print("\n--- Premature Sell Analysis (Selling while Price > MA185) ---")
# Identify days where V3 sold (Cash Ratio Increased) AND Price > MA185
# "Sells" are when cash_ratio_v3 increases
v3_is_selling = cash_ratio_v3 > cash_ratio_v3.shift(1)
premature_sells = v3_is_selling & (df["Price"] > df["MA185"])
premature_dates = df.index[premature_sells]

if not premature_dates.empty:
    print(f"Total Premature Sells: {len(premature_dates)} days")

    # Analyze Next 20 Days Return for each sell
    outcomes = []
    for date in premature_dates:
        try:
            # Find price 20 days later
            idx = df.index.get_loc(date)
            if idx + 20 < len(df):
                price_now = df["Price"].iloc[idx]
                price_future = df["Price"].iloc[idx + 20]
                ret = (price_future / price_now) - 1
                outcomes.append(ret)
            else:
                outcomes.append(0.0)
        except:
            pass

    outcomes = np.array(outcomes)
    # Good Sell = Price Dropped (obs < 0)
    # Bad Sell = Price Rose (obs > 0)
    good_sells = np.sum(outcomes < 0)
    bad_sells = np.sum(outcomes > 0)
    avg_avoided_loss = np.mean(outcomes[outcomes < 0]) * 100 if good_sells > 0 else 0
    avg_missed_gain = np.mean(outcomes[outcomes > 0]) * 100 if bad_sells > 0 else 0

    print(
        f"Success Rate (Price Dropped after sell): {good_sells}/{len(outcomes)} ({good_sells / len(outcomes) * 100:.1f}%)"
    )
    print(f"Avg Loss Avoided (Good Calls): {avg_avoided_loss:.2f}%")
    print(f"Avg Gain Missed (Bad Calls): {avg_missed_gain:.2f}%")
    print("Conclusion: Is it stupid? Depends if Avoided Loss > Missed Gain.")

    with open("premature_analysis.txt", "w") as f:
        f.write(f"Total Premature Sells: {len(premature_dates)}\n")
        f.write(f"Success Rate: {good_sells / len(outcomes) * 100:.1f}%\n")
        f.write(f"Avg Loss Avoided: {avg_avoided_loss:.2f}%\n")
        f.write(f"Avg Gain Missed: {avg_missed_gain:.2f}%\n")
else:
    print("No premature sells found.")
