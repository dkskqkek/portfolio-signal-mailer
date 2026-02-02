import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Data Download
# ---------------------------------------------------------
# Assets
target_asset = "SPY"
defensive_asset = "SHY"

# Pillar Data Sources
# Valuation: SPY Dividends (will fetch separately)
# Trend: SPY Price
# Risk: VIX
# Macro: 10Y Treasury (^TNX) - 13W Treasury (^IRX)
risk_index = "^VIX"
ten_year = "^TNX"
three_month = "^IRX"

all_symbols = [target_asset, defensive_asset, risk_index, ten_year, three_month]

print(f"Downloading data for V5: {all_symbols}")
start_date = "2000-01-01"  # Need longer history for rolling percentile
data = yf.download(all_symbols, start=start_date, progress=False)

# Robust Data Access
if "Adj Close" in data.columns:
    df_price = data["Adj Close"]
elif "Close" in data.columns:
    df_price = data["Close"]
else:
    # Handle MultiIndex
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                df_price = data.xs("Adj Close", level=0, axis=1)
            else:
                df_price = data.xs("Close", level=0, axis=1)
    except:
        df_price = data

df_price = df_price.ffill()

# Normalize Timezone (Fix Mismatch Error)
if df_price.index.tz is not None:
    df_price.index = df_price.index.tz_localize(None)

# Fetch Dividends for Valuation Pillar
print("Fetching Dividend Data for Valuation Pillar...")
spy_ticker = yf.Ticker("SPY")
div_history = spy_ticker.dividends
# Filter to start date
if div_history.index.tz is not None:
    div_history.index = div_history.index.tz_localize(None)

div_history = div_history[div_history.index >= start_date]

# ---------------------------------------------------------
# 2. Indicator Calculation (The 4 Pillars)
# ---------------------------------------------------------

# Align Dividend Data to Daily Price
# Create a series with dividends on ex-date, 0 otherwise
daily_divs = pd.Series(0.0, index=df_price.index)
# Reindex dividends to match daily dates (filling 0)
daily_divs = daily_divs.add(div_history, fill_value=0)
# Rolling Sum of Dividends (12 months)
rolling_annual_div = daily_divs.rolling(window=252).sum()
# Dividend Yield = Annual Div / Current Price
div_yield = rolling_annual_div / df_price[target_asset]

# A. Valuation Score: Dividend Yield
# Higher Yield = Better (Cheap)
val_indicator = div_yield

# B. Trend Score: 12-Month Momentum
# Price / Price_252_ago - 1
trend_indicator = df_price[target_asset].pct_change(252)

# C. Risk Score: Inverse VIX
# Lower VIX = Better. We rank VIX normally (High Rank = High VIX = Bad).
# Then Score = 1 - Rank.
risk_indicator = df_price[risk_index]

# D. Macro Score: Yield Curve Slope (10Y - 3M)
# Steep positive slope = Good. Inverted (Negative) = Bad.
# ^TNX and ^IRX are in %, e.g., 4.5
macro_indicator = df_price[ten_year] - df_price[three_month]

# ---------------------------------------------------------
# 3. Rolling Percentile Scoring
# ---------------------------------------------------------
# Lookback window for ranking: 5 Years (1260 days)
lookback_window = 1260


def rolling_percentile(series, window):
    # Calculates the percentile rank of the current value within the rolling window
    # Returns 0.0 to 1.0
    return series.rolling(window).apply(lambda x: (x < x[-1]).mean())


print("Calculating Rolling Percentile Scores (This may take a moment)...")

# 1. Valuation Score (Higher Yield is better)
# If Rank is 0.9, it means current yield is higher than 90% of history -> Good -> Score 0.9 (90)
score_val = rolling_percentile(val_indicator, lookback_window) * 100

# 2. Trend Score (Higher Mom is better)
# Rank 0.9 -> Mom higher than 90% of history -> Good -> Score 0.9 (90)
score_trend = rolling_percentile(trend_indicator, lookback_window) * 100

# 3. Risk Score (Lower VIX is better)
# We want Low VIX to have High Score.
# Rank of VIX: High Rank = High VIX.
# So Score = (1 - Rank) * 100
rank_vix = rolling_percentile(risk_indicator, lookback_window)
score_risk = (1 - rank_vix) * 100

# 4. Macro Score (Higher Slope is better)
score_macro = rolling_percentile(macro_indicator, lookback_window) * 100

# Composite Score
composite_score = (score_val + score_trend + score_risk + score_macro) / 4

# ---------------------------------------------------------
# 4. Calibration & Strategy Logic
# ---------------------------------------------------------
# Schnetzer Rule: Score determines Over/Underweight.
# We map Score directly to Equity Weight (Linear).
# Score 100 -> 100% Equity
# Score 0 -> 0% Equity (100% Cash)
# Score 50 -> 50% Equity
target_weight = composite_score / 100.0
target_weight = target_weight.fillna(0)  # Before enough data exists

# Create DataFrame for analysis from 2008
analysis_start = "2008-01-01"
df = pd.DataFrame(
    {
        "Price": df_price[target_asset],
        "Score_Val": score_val,
        "Score_Trend": score_trend,
        "Score_Risk": score_risk,
        "Score_Macro": score_macro,
        "Total_Score": composite_score,
        "Target_Weight": target_weight,
        "Target_Ret": df_price[target_asset].pct_change(),
        "Cash_Ret": df_price[defensive_asset].pct_change(),
    }
)
df = df.loc[analysis_start:]

# Calculate Returns
# Position logic: target_weight today determines exposure for tomorrow
pos_v5 = df["Target_Weight"].shift(1).fillna(0)
cash_v5 = 1 - pos_v5

ret_v5 = df["Target_Ret"] * pos_v5 + df["Cash_Ret"] * cash_v5
cum_v5 = (1 + ret_v5).cumprod()

# Comparison: V3 (Import or replicate simpler version)
# V3 Proxy for comparison
# We rely on saved metrics for precise V3 comparison, or just simpler Buy&Hold here.
cum_bh = (1 + df["Target_Ret"]).cumprod()

# Rebase
cum_v5 /= cum_v5.iloc[0]
cum_bh /= cum_bh.iloc[0]


# Metrics
def calc_metrics(series, name):
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


print("\n--- V5 Schnetzer Performance (2008-2024) ---")
calc_metrics(cum_bh, "Buy & Hold")
c, m, s = calc_metrics(cum_v5, "V5 (Schnetzer 4-Pillar)")

# Save Metrics
with open("v5_metrics.txt", "w") as f:
    f.write(f"V5 (Schnetzer),{c * 100:.2f},{m * 100:.2f},{s:.2f}\n")

# ---------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------
plt.figure(figsize=(14, 12))

# Equity Curve
plt.subplot(4, 1, 1)
plt.plot(cum_bh, label="Buy & Hold", color="gray", alpha=0.3)
plt.plot(cum_v5, label="V5 (Schnetzer)", color="purple", linewidth=2)
plt.title("V5 (Schnetzer 4-Pillar): Performance")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)

# Weight History
plt.subplot(4, 1, 2)
plt.plot(df["Target_Weight"] * 100, label="Equity Weight %", color="green", alpha=0.6)
plt.axhline(50, color="orange", linestyle="--")
plt.ylabel("Equity %")
plt.title("Dynamic Asset Allocation (Equity %)")
plt.legend()
plt.grid(True, alpha=0.3)

# Pillar Scores
plt.subplot(4, 1, 3)
plt.plot(df["Score_Val"], label="Valuation", alpha=0.3)
plt.plot(df["Score_Trend"], label="Trend", alpha=0.3)
plt.plot(df["Score_Risk"], label="Risk", alpha=0.3)
plt.plot(df["Score_Macro"], label="Macro", alpha=0.3)
plt.plot(df["Total_Score"], label="TOTAL SCORE", color="black", linewidth=2)
plt.title("Pillar Scores (0-100)")
plt.legend(ncol=5, fontsize="small")
plt.grid(True, alpha=0.3)

# Returns Distribution (Optional) or Drawdown
plt.subplot(4, 1, 4)
roll_max = cum_v5.cummax()
dd = (cum_v5 - roll_max) / roll_max
plt.plot(dd, label="Drawdown", color="red", alpha=0.5)
plt.fill_between(dd.index, dd, 0, color="red", alpha=0.1)
plt.title("Drawdown Profile")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("v5_schnetzer_analysis.png")
print("Analysis saved to v5_schnetzer_analysis.png")
