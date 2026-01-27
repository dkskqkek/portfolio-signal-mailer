import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# ==========================================
# 1. Robust Data Fetching
# ==========================================
def fetch_data(tickers, days_back=365 * 12):
    print(f"\nFetching data for {len(tickers)} tickers (Sequential)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    data_dict = {}

    for t in tickers:
        try:
            df = yf.download(
                t, start=start_date, end=end_date, progress=False, auto_adjust=False
            )
            if df.empty:
                continue

            if "Adj Close" in df.columns:
                data_dict[t] = df["Adj Close"]
            elif "Close" in df.columns:
                data_dict[t] = df["Close"]
            else:
                if len(df.columns) == 1:
                    data_dict[t] = df.iloc[:, 0]
        except:
            pass

    data = pd.DataFrame(data_dict)
    data = data.ffill().dropna()
    return data


# Tickers needed for both strategies
# User: QQQ, SPY, GOOGL, VGT, SCHD, COWZ, XLV, VXUS, EWY, CVX, GLD
# Optimized: QLD, SPY, EWY, GLD
# Defensive: BIL (Cash proxy)

tickers = [
    "QQQ",
    "SPY",
    "GOOGL",
    "VGT",
    "SCHD",
    "COWZ",
    "XLV",
    "VXUS",
    "EWY",
    "CVX",
    "GLD",
    "QLD",
    "BIL",
    "IEF",
    "TLT",
    "DBC",  # Defensive pool
]
# Add user specific if missing
user_specific = ["GOOGL", "VGT", "SCHD", "COWZ", "XLV", "CVX"]
tickers = list(set(tickers + user_specific))

data = fetch_data(tickers)
returns = data.pct_change()
print(f"Data Loaded: {data.index[0].date()} ~ {data.index[-1].date()}")


# ==========================================
# 2. Logic: Dual SMA Signal (on QQQ)
# ==========================================
def calculate_signal(price_series):
    short_w = 110
    long_w = 250
    ma_s = price_series.rolling(short_w).mean()
    ma_l = price_series.rolling(long_w).mean()

    signals = []
    state = "NORMAL"

    for i in range(len(price_series)):
        p = price_series.iloc[i]
        s = ma_s.iloc[i]
        l = ma_l.iloc[i]

        if pd.isna(s) or pd.isna(l):
            signals.append("NORMAL")
            continue

        if p < s and p < l:
            state = "DANGER"
        elif p > s and p > l:
            state = "NORMAL"
        signals.append(state)
    return signals


signals = calculate_signal(data["QQQ"])
dates = data.index


# ==========================================
# 3. Defensive Selection (Top-3)
# ==========================================
def get_defensive(date):
    try:
        idx = data.index.get_loc(date)
        lookback = 168
        if idx < lookback:
            return ["BIL"]

        pool = ["BIL", "IEF", "TLT", "DBC", "GLD"]
        subset = data[pool].iloc[idx - lookback : idx]
        mom = (subset.iloc[-1] / subset.iloc[0]) - 1
        pos = mom[mom > 0].sort_values(ascending=False)
        if len(pos) == 0:
            return ["BIL"]
        return pos.head(3).index.tolist()
    except:
        return ["BIL"]


# ==========================================
# 4. Simulation Loop
# ==========================================
# User Portfolio Weights (Normalized)
# Tactical (~50%): GOOGL(14), VGT(9), QQQ(12), SPY(15) -> Total 0.50
# Static (~50%): SCHD(9), COWZ(9), XLV(6), VXUS(5), EWY(9), CVX(6), GLD(6)
w_user = {
    "Tactical": {"GOOGL": 0.14, "VGT": 0.09, "QQQ": 0.12, "SPY": 0.15},
    "Static": {
        "SCHD": 0.09,
        "COWZ": 0.09,
        "XLV": 0.06,
        "VXUS": 0.05,
        "EWY": 0.09,
        "CVX": 0.06,
        "GLD": 0.06,
    },
}

# Optimized Strategy Weights
# Tactical (45%): QLD
# Core (20%): SPY
# Korea (20%): EWY
# Safe (15%): GLD
w_opt = {"Tactical": {"QLD": 0.45}, "Static": {"SPY": 0.20, "EWY": 0.20, "GLD": 0.15}}

val_user = [100.0]
val_opt = [100.0]
val_spy = [100.0]

for i in range(1, len(data)):
    date = dates[i]
    prev_date = dates[i - 1]
    sig = signals[i - 1]

    # Defensive assets if needed
    def_assets = get_defensive(prev_date) if sig == "DANGER" else []

    # --- User Portfolio ---
    ret_u = 0.0
    # Static
    for t, w in w_user["Static"].items():
        r = returns[t].iloc[i] if not pd.isna(returns[t].iloc[i]) else 0
        ret_u += r * w
    # Tactical
    if sig == "NORMAL":
        for t, w in w_user["Tactical"].items():
            r = returns[t].iloc[i] if not pd.isna(returns[t].iloc[i]) else 0
            ret_u += r * w
    else:
        # Switch Tactical Total (0.50) to Defensive
        w_total = sum(w_user["Tactical"].values())
        w_each = w_total / len(def_assets)
        for t in def_assets:
            r = returns[t].iloc[i] if not pd.isna(returns[t].iloc[i]) else 0
            ret_u += r * w_each

    val_user.append(val_user[-1] * (1 + ret_u))

    # --- Optimized Strategy ---
    ret_o = 0.0
    # Static
    for t, w in w_opt["Static"].items():
        r = returns[t].iloc[i] if not pd.isna(returns[t].iloc[i]) else 0
        ret_o += r * w
    # Tactical
    if sig == "NORMAL":
        for t, w in w_opt["Tactical"].items():
            r = returns[t].iloc[i] if not pd.isna(returns[t].iloc[i]) else 0
            ret_o += r * w
    else:
        w_total = sum(w_opt["Tactical"].values())
        w_each = w_total / len(def_assets)
        for t in def_assets:
            r = returns[t].iloc[i] if not pd.isna(returns[t].iloc[i]) else 0
            ret_o += r * w_each

    val_opt.append(val_opt[-1] * (1 + ret_o))

    # --- SPY ---
    r_spy = returns["SPY"].iloc[i] if not pd.isna(returns["SPY"].iloc[i]) else 0
    val_spy.append(val_spy[-1] * (1 + r_spy))

# ==========================================
# 5. Metrics & Plot
# ==========================================
results = pd.DataFrame(
    {"User": val_user, "Optimized": val_opt, "SPY": val_spy}, index=data.index
)


def calc_metrics(s):
    try:
        total_ret = (s.iloc[-1] / s.iloc[0]) - 1
        days = (s.index[-1] - s.index[0]).days
        cagr = (s.iloc[-1] / s.iloc[0]) ** (365.25 / days) - 1
        mdd = (s / s.cummax() - 1).min()
        vol = s.pct_change().std() * np.sqrt(252)
        sharpe = (cagr - 0.04) / vol
        return cagr, mdd, sharpe
    except:
        return 0, 0, 0


m_u = calc_metrics(results["User"])
m_o = calc_metrics(results["Optimized"])
m_s = calc_metrics(results["SPY"])

print("\n" + "=" * 60)
print("             STRATEGY COMPARISON REPORT")
print("=" * 60)
print(f"Metrics      | User Portfolio | Optimized (Antigravity) | SPY")
print("-" * 60)
print(f"CAGR         | {m_u[0]:13.2%} | {m_o[0]:22.2%} | {m_s[0]:6.2%}")
print(f"MDD          | {m_u[1]:13.2%} | {m_o[1]:22.2%} | {m_s[1]:6.2%}")
print(f"Sharpe Ratio | {m_u[2]:13.2f} | {m_o[2]:22.2f} | {m_s[2]:6.2f}")
print("=" * 60)

plt.figure(figsize=(12, 7))
plt.plot(results["User"], label=f"User Portfolio (CAGR {m_u[0]:.1%})", linewidth=2)
plt.plot(
    results["Optimized"],
    label=f"Optimized Antigravity (CAGR {m_o[0]:.1%})",
    linewidth=2,
    linestyle="--",
)
plt.plot(
    results["SPY"], label=f"SPY Benchmark (CAGR {m_s[0]:.1%})", alpha=0.5, color="gray"
)

plt.title(
    "Portfolio Strategy Comparison (2015-2026)\nDynamic Defense (Dual SMA) applied to Tactical Portion"
)
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("research/full_strategy_comparison.png")
print("Chart saved to research/full_strategy_comparison.png")
