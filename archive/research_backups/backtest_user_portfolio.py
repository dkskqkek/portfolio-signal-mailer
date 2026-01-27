import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# ==========================================
# 1. Configuration & User Weights
# ==========================================

# 사용자 정의 분류 및 비중 (Normalized to 100%)
# Total Size Est: ~226M KRW
# Weights: Tactical ~50.6%, Core ~22.3%, Korea ~13.3%, Safe ~10.1%

# Logic:
# - NORMAL Regime: Hold "User Mix" (Alphabet, VGT, S&P, QQQ, SCHD, etc.) using current relative weights.
# - DANGER Regime:
#   - Tactical (Alphabet, VGT, SPY, QQQ) -> Switch to Top-3 Defensive Ensemble (GLD, BIL, etc.)
#   - Core/Korea/Safe -> HOLD

PORTFOLIO = {
    # --- Tactical (Active Switching) ---
    "Tactical": {"GOOGL": 0.131, "VGT": 0.084, "QQQ": 0.111, "SPY": 0.154},
    # --- Static Holds (Buy & Hold) ---
    "Core": {"SCHD": 0.082, "COWZ": 0.088, "XLV": 0.054},
    "Korea": {"VXUS": 0.048, "EWY": 0.085},
    "Safe": {"CVX": 0.050, "GLD": 0.051},
}

# Normalize Weights
all_weights = {}
for category in PORTFOLIO.values():
    all_weights.update(category)

total_w = sum(all_weights.values())
for k in all_weights:
    all_weights[k] /= total_w  # Normalize to sum 1.0

print("=== Normalized User Portfolio Weights ===")
for k, v in all_weights.items():
    print(f"{k}: {v:.1%}")

# Defensive Pool
DEFENSIVE_POOL = ["BIL", "IEF", "TLT", "DBC", "GLD", "UUP", "SPY"]


# ==========================================
# 2. Data Fetching (Robust)
# ==========================================
def fetch_data(tickers, days_back=365 * 12):  # 12 years to cover 2015
    print(f"\nFetching data for {len(tickers)} tickers (Sequential)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    data_dict = {}

    for t in tickers:
        try:
            # Download individually to avoid MultiIndex issues
            df = yf.download(
                t, start=start_date, end=end_date, progress=False, auto_adjust=False
            )

            if df.empty:
                print(f"Warning: {t} is empty.")
                continue

            # Robust Column Extraction
            if "Adj Close" in df.columns:
                data_dict[t] = df["Adj Close"]
            elif "Close" in df.columns:
                data_dict[t] = df["Close"]
            else:
                # Check if single column remaining
                if len(df.columns) == 1:
                    data_dict[t] = df.iloc[:, 0]
                else:
                    print(f"Warning: {t} has unexpected columns {df.columns}")

        except Exception as e:
            print(f"Error fetching {t}: {e}")

    data = pd.DataFrame(data_dict)
    data = data.ffill().dropna()
    print(f"Data fetched successfully: {data.shape}")
    return data


# Extract unique tickers
active_tickers = list(PORTFOLIO["Tactical"].keys())
cat_tickers = []
for c in ["Core", "Korea", "Safe"]:
    cat_tickers.extend(list(PORTFOLIO[c].keys()))

all_tickers = list(set(active_tickers + cat_tickers + DEFENSIVE_POOL + ["QQQ"]))

data = fetch_data(all_tickers)


# ==========================================
# 3. Logic: Dual SMA + Defensive
# ==========================================
def calculate_dual_sma_signal(price_series, short_w=110, long_w=250):
    ma_short = price_series.rolling(window=short_w).mean()
    ma_long = price_series.rolling(window=long_w).mean()

    status_list = []
    state = "NORMAL"

    for i in range(len(price_series)):
        p = price_series.iloc[i]
        s = ma_short.iloc[i]
        l = ma_long.iloc[i]

        if pd.isna(s) or pd.isna(l):
            status_list.append("NORMAL")
            continue

        if p < s and p < l:
            state = "DANGER"
        elif p > s and p > l:
            state = "NORMAL"
        # Hysteresis: keep previous state

        status_list.append(state)

    return status_list


print("\nGenerating Signals (QQQ Dual SMA)...")
signals = calculate_dual_sma_signal(data["QQQ"])
data["Signal"] = signals

# ==========================================
# 4. Backtest Loop
# ==========================================
portfolio_val = 1000.0
daily_values = [portfolio_val]
dates = data.index
returns = data.pct_change()

# Tactical Base Weight
tactical_total_w = sum(PORTFOLIO["Tactical"].values())  # ~0.48 normalized
# Sub-weights within Tactical (sum to 1.0 relative to tactical part)
tactical_sub_weights = {
    k: v / tactical_total_w for k, v in PORTFOLIO["Tactical"].items()
}


def get_defensive_assets(current_date, lookback=168, qt=3):
    try:
        idx = data.index.get_loc(current_date)
        if idx < lookback:
            return ["BIL"]

        subset = data[DEFENSIVE_POOL].iloc[idx - lookback : idx]
        mom = (subset.iloc[-1] / subset.iloc[0]) - 1
        positive = mom[mom > 0].sort_values(ascending=False)

        if len(positive) == 0:
            return ["BIL"]
        return positive.head(qt).index.tolist()
    except:
        return ["BIL"]


print("\nRunning Simulation...")

for i in range(1, len(data)):
    date = dates[i]
    prev_date = dates[i - 1]

    # 1. Static Portfolio Return
    static_ret = 0.0
    for cat in ["Core", "Korea", "Safe"]:
        for t, w in PORTFOLIO[cat].items():
            r = returns[t].iloc[i]
            if pd.isna(r):
                r = 0.0
            static_ret += r * all_weights[t]  # Use global normalized weight

    # 2. Tactical Portfolio Return
    signal = data["Signal"].iloc[i - 1]
    tactical_ret_component = 0.0

    if signal == "NORMAL":
        # Hold User Mix
        for t, sub_w in tactical_sub_weights.items():
            r = returns[t].iloc[i]
            if pd.isna(r):
                r = 0.0
            # Weight contribution to total portfolio = (Tactical Total) * (Sub Weight)
            tactical_ret_component += r * (tactical_total_w * sub_w)
    else:
        # Switch to Defensive
        def_assets = get_defensive_assets(prev_date)
        w_each = tactical_total_w / len(def_assets)
        for t in def_assets:
            r = returns[t].iloc[i]
            if pd.isna(r):
                r = 0.0
            tactical_ret_component += r * w_each

    total_day_ret = static_ret + tactical_ret_component
    portfolio_val *= 1 + total_day_ret
    daily_values.append(portfolio_val)

# ==========================================
# 5. Benchmark (SPY)
# ==========================================
spy_val = [1000.0]
for i in range(1, len(data)):
    r = returns["SPY"].iloc[i]
    if pd.isna(r):
        r = 0.0
    spy_val.append(spy_val[-1] * (1 + r))

# ==========================================
# 6. Reporting
# ==========================================
results = pd.DataFrame(
    {"User_Portfolio": daily_values, "SPY_Benchmark": spy_val}, index=data.index
)


def calc_metrics(s):
    try:
        total_ret = (s.iloc[-1] / s.iloc[0]) - 1
        days = (s.index[-1] - s.index[0]).days
        years = days / 365.25
        cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
        mdd = (s / s.cummax() - 1).min()
        vol = s.pct_change().std() * np.sqrt(252)
        sharpe = (cagr - 0.04) / vol
        return cagr, mdd, sharpe
    except:
        return 0, 0, 0


u_cagr, u_mdd, u_sharpe = calc_metrics(results["User_Portfolio"])
b_cagr, b_mdd, b_sharpe = calc_metrics(results["SPY_Benchmark"])

print("\n" + "=" * 50)
print("   BACKTEST RESULTS (User Portfolio vs SPY)")
print("=" * 50)
print(f"Start Date   : {dates[0].date()}")
print(f"End Date     : {dates[-1].date()}")
print("-" * 50)
print(f"Metric       | User Portfolio | SPY Benchmark")
print("-" * 50)
print(f"CAGR         | {u_cagr:13.2%} | {b_cagr:12.2%}")
print(f"MDD          | {u_mdd:13.2%} | {b_mdd:12.2%}")
print(f"Sharpe Ratio | {u_sharpe:13.2f} | {b_sharpe:12.2f}")
print("-" * 50)
print(f"Final Value  | {portfolio_val:13.2f} | {spy_val[-1]:12.2f}")
print("=" * 50)

# Save Plot
plt.figure(figsize=(12, 6))
plt.plot(
    results["User_Portfolio"], label="User Portfolio (Dynamic Defense)", linewidth=2
)
plt.plot(results["SPY_Benchmark"], label="SPY (Benchmark)", alpha=0.6)
plt.title(f"User Portfolio Backtest\n(Tactical Portion Swaps to Defensive in Danger)")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("research/user_portfolio_backtest.png")
print("Chart saved.")
