import yfinance as yf
import pandas as pd
import numpy as np

# 1. Portfolios to Compare
current_weights = {
    "SPY": 0.163,
    "GOOGL": 0.135,
    "QQQ": 0.118,
    "VGT": 0.095,
    "SCHD": 0.087,
    "COWZ": 0.064,
    "XLV": 0.053,
    "CVX": 0.052,
    "VXUS": 0.049,
    "EWY": 0.100,
    "GLD": 0.073,
    "BIL": 0.002,
    "NVR": 0.008,
}

user_strategic_weights = {
    "GLD": 0.15,
    "GOOGL": 0.20,
    "CVX": 0.10,
    "VXUS": 0.10,
    "SCHD": 0.30,
    "NVR": 0.05,
    "TIP": 0.10,  # 물가연동채 (TIPS)
}

# Tickers needed for historical analysis
all_tickers = sorted(
    list(set(current_weights.keys()) | set(user_strategic_weights.keys()))
)


def analyze_comparison():
    print(f"Downloading data for: {all_tickers}")
    data = yf.download(all_tickers, period="5y")["Close"].dropna()
    returns = data.pct_change().dropna()

    mu = returns.mean() * 252
    cov = returns.cov() * 252

    def get_stats(w_dict):
        # Align weights with columns
        w = np.array([w_dict.get(t, 0) for t in returns.columns])
        w = w / np.sum(w)  # Normalize
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe = (ret - 0.04) / vol
        return ret, vol, sharpe

    c_ret, c_vol, c_sharpe = get_stats(current_weights)
    u_ret, u_vol, u_sharpe = get_stats(user_strategic_weights)

    print("\n--- Portfolio Comparison Report ---")
    print(f"{'Metric':<15} | {'Current Portfolio':<18} | {'User Strategic':<18}")
    print("-" * 60)
    print(f"{'Exp. Return':<15} | {c_ret * 100:17.2f}% | {u_ret * 100:17.2f}%")
    print(f"{'Volatility':<15} | {c_vol * 100:17.2f}% | {u_vol * 100:17.2f}%")
    print(f"{'Sharpe Ratio':<15} | {c_sharpe:17.3f} | {u_sharpe:17.3f}")

    print("\n[User Strategic Composition]")
    for t, w in sorted(
        user_strategic_weights.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {t:<10}: {w * 100:>5.1f}%")


if __name__ == "__main__":
    analyze_comparison()
