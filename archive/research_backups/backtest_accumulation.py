import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==========================================
# 1. Configuration
# ==========================================
INITIAL_CAPITAL = 200_000_000
MONTHLY_CONTRIBUTION = 1_000_000

TARGET_W = {"Tactical": 0.45, "Core": 0.20, "Korea": 0.20, "Safe": 0.15}


# ==========================================
# 2. Robust Data Fetching
# ==========================================
def fetch_ticker(symbol, days_back=365 * 12):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    try:
        # Request single ticker
        df = yf.download(
            symbol, start=start_date, end=end_date, progress=False, auto_adjust=False
        )
        if df.empty:
            return None

        # Flatten columns if MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            if "Adj Close" in df.columns:
                s = df["Adj Close"]
            elif "Close" in df.columns:
                s = df["Close"]
            else:
                s = df.iloc[:, 0]
        else:
            if "Adj Close" in df.columns:
                s = df["Adj Close"]
            elif "Close" in df.columns:
                s = df["Close"]
            else:
                s = df.iloc[:, 0]

        # Ensure it is a Series, not DataFrame (if duplicate columns exist)
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]  # Take first

        return s
    except Exception as e:
        print(f"Error {symbol}: {e}")
        return None


tickers = ["QQQ", "QLD", "SPY", "EWY", "GLD", "BIL"]
print(f"Fetching {tickers}...")

data_dict = {}
for t in tickers:
    s = fetch_ticker(t)
    if s is not None:
        s.name = t
        data_dict[t] = s

# Align
data = pd.concat(data_dict.values(), axis=1, keys=data_dict.keys())
data = data.ffill().dropna()
dates = data.index
print(f"Data Loaded: {data.shape}")


# ==========================================
# 3. Signal Logic
# ==========================================
def calc_signal(prices):
    # Ensure Series
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    s = prices.rolling(110).mean()
    l = prices.rolling(250).mean()

    sigs = []
    # Use numpy iteration for safety
    np_p = prices.values
    np_s = s.values
    np_l = l.values

    for i in range(len(np_p)):
        try:
            p = np_p[i]
            ms = np_s[i]
            ml = np_l[i]

            if np.isnan(ms) or np.isnan(ml):
                sigs.append("NORMAL")
            elif p < ms and p < ml:
                sigs.append("DANGER")
            elif p > ms and p > ml:
                sigs.append("NORMAL")
            else:
                sigs.append("NORMAL")
        except:
            sigs.append("NORMAL")
    return sigs


signals = calc_signal(data["QQQ"])


# ==========================================
# 4. Simulation
# ==========================================
def run_sim(strategy):
    holdings = {
        "QLD": INITIAL_CAPITAL * TARGET_W["Tactical"],
        "SPY": INITIAL_CAPITAL * TARGET_W["Core"],
        "EWY": INITIAL_CAPITAL * TARGET_W["Korea"],
        "GLD": INITIAL_CAPITAL * TARGET_W["Safe"],
    }
    cash_pile = 0
    tvs = []
    prev_m = -1

    pct = data.pct_change()

    # Pre-calculate Numpy arrays for loop speed/safety
    np_pct = pct.values  # shape (N, cols)
    cols = list(pct.columns)
    col_idx = {c: i for i, c in enumerate(cols)}

    asset_keys = list(holdings.keys())

    for i in range(1, len(data)):
        date = dates[i]
        curr_m = date.month

        # 1. Returns
        row_rets = np_pct[i]  # array of returns

        for asset in asset_keys:
            if asset in col_idx:
                r = row_rets[col_idx[asset]]
                if np.isnan(r):
                    r = 0
                holdings[asset] *= 1 + r

        # Cash returns
        if "BIL" in col_idx:
            r_c = row_rets[col_idx["BIL"]]
            if np.isnan(r_c):
                r_c = 0
            cash_pile *= 1 + r_c

        # 2. Monthly Contribution
        if curr_m != prev_m:
            sig = signals[i - 1]
            amt = MONTHLY_CONTRIBUTION

            if strategy == "Simple":
                holdings["QLD"] += amt * TARGET_W["Tactical"]
                holdings["SPY"] += amt * TARGET_W["Core"]
                holdings["EWY"] += amt * TARGET_W["Korea"]
                holdings["GLD"] += amt * TARGET_W["Safe"]

            elif strategy == "Smart":
                if sig == "NORMAL":
                    holdings["QLD"] += amt
                else:
                    holdings["GLD"] += amt

            elif strategy == "Sniper":
                if sig == "NORMAL":
                    buy = amt + cash_pile
                    holdings["QLD"] += buy
                    cash_pile = 0
                else:
                    cash_pile += amt

            prev_m = curr_m

        tvs.append(sum(holdings.values()) + cash_pile)

    return pd.Series(tvs, index=dates[1:])


print("Running Simulations...")
s1 = run_sim("Simple")
s2 = run_sim("Smart")
s3 = run_sim("Sniper")

# Final check
f1, f2, f3 = s1.iloc[-1], s2.iloc[-1], s3.iloc[-1]
print(f"End Values: Simple={f1:,.0f}, Smart={f2:,.0f}, Sniper={f3:,.0f}")

plt.figure(figsize=(10, 6))
plt.plot(s1, label="Simple DCA", linestyle="--")
plt.plot(s2, label="Smart Regime DCA")
plt.plot(s3, label="Cash Sniper")
plt.title("Monthly Accumulation Strategy Comparison")
plt.legend()
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.savefig("research/accumulation_comparison.png")
print("Chart saved.")
