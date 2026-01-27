import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. Configuration
# ==========================================
PORTFOLIO = {
    "Tactical": {"GOOGL": 0.131, "VGT": 0.084, "QQQ": 0.111, "SPY": 0.154},
    "Static": {
        "SCHD": 0.082,
        "COWZ": 0.088,
        "XLV": 0.054,
        "VXUS": 0.048,
        "EWY": 0.085,
        "CVX": 0.050,
        "GLD": 0.051,
    },
}
DEFENSIVE_POOL = ["BIL", "IEF", "TLT", "DBC", "GLD", "UUP", "SPY"]

# Normalize
all_weights = {}
for cat in PORTFOLIO:
    for t, w in PORTFOLIO[cat].items():
        all_weights[t] = w
total_w = sum(all_weights.values())
for k in all_weights:
    all_weights[k] /= total_w

tactical_total_w = sum(PORTFOLIO["Tactical"].values()) / total_w


# ==========================================
# 2. Robust Data Fetching
# ==========================================
def fetch_data(tickers, days_back=365 * 12):
    print(f"Fetching data for {len(tickers)} tickers...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    series_list = []

    for t in tickers:
        try:
            df = yf.download(
                t, start=start_date, end=end_date, progress=False, auto_adjust=False
            )
            if df.empty:
                continue

            # Extract Series
            s = None
            if "Adj Close" in df.columns:
                s = df["Adj Close"]
            elif "Close" in df.columns:
                s = df["Close"]
            else:
                s = df.iloc[:, 0]

            if s is not None:
                s.name = t
                series_list.append(s)
        except:
            pass

    print(f"Collected {len(series_list)} series.")
    # Use concat which aligns indexes correctly
    data = pd.concat(series_list, axis=1)
    data = data.ffill().dropna()
    print(f"Data shape: {data.shape}")
    return data


active = list(PORTFOLIO["Tactical"].keys())
static = list(PORTFOLIO["Static"].keys())
all_t = list(set(active + static + DEFENSIVE_POOL + ["QQQ"]))

data = fetch_data(all_t)


# ==========================================
# 3. Weekly History Construction
# ==========================================
def calculate_signal(prices):
    s = prices.rolling(110).mean()
    l = prices.rolling(250).mean()
    state = "NORMAL"
    sigs = []
    for i in range(len(prices)):
        p = prices.iloc[i]
        ms = s.iloc[i]
        ml = l.iloc[i]

        if pd.isna(ms) or pd.isna(ml):
            sigs.append("NORMAL")
            continue

        if p < ms and p < ml:
            state = "DANGER"
        elif p > ms and p > ml:
            state = "NORMAL"
        sigs.append(state)
    return sigs


signals = calculate_signal(data["QQQ"])
dates = data.index

history_rows = []
prev_week = -1


def get_defensive(date):
    try:
        idx = data.index.get_loc(date)
        if idx < 168:
            return ["BIL"]
        sub = data[DEFENSIVE_POOL].iloc[idx - 168 : idx]
        mom = (sub.iloc[-1] / sub.iloc[0]) - 1
        top = mom[mom > 0].sort_values(ascending=False).head(3).index.tolist()
        return top if len(top) > 0 else ["BIL"]
    except:
        return ["BIL"]


print("Generating history...")
for i in range(1, len(data)):
    date = dates[i]
    if i >= len(signals):
        break

    # Check if Friday or last day of data
    is_friday = date.weekday() == 4
    if is_friday:
        sig = signals[i - 1]  # Use signal from prev day close for action? Or today?
        # Usually signal is calculated effectively on Close.

        holdings_str = ""
        # Static
        holdings_str += f"[Static {1 - tactical_total_w:.0%}] "

        # Tactical
        if sig == "NORMAL":
            holdings_str += f"[Tactical {tactical_total_w:.0%}: GOOGL,VGT,..."
        else:
            def_list = get_defensive(dates[i - 1])
            def_names = ",".join(def_list)
            holdings_str += f"[Defensive {tactical_total_w:.0%}: {def_names}]"

        row = {
            "Date": date.strftime("%Y-%m-%d"),
            "Signal": sig,
            "Holdings": holdings_str,
            "QQQ_Price": f"{data['QQQ'].iloc[i]:.2f}",
        }
        history_rows.append(row)

df_hist = pd.DataFrame(history_rows)
# Save
out_path = "research/user_portfolio_weekly_history.csv"
df_hist.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"Saved {out_path}")
print(df_hist.tail(10).to_markdown(index=False))
