import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# 1. Asset Mapping to Tickers (Proxies for non-US assets)
current_portfolio = {
    "SPY": 0.163,  # S&P 500
    "GOOGL": 0.135,  # GOOGL Individual
    "QQQ": 0.118,  # Nasdaq 100
    "VGT": 0.095,  # Technology
    "SCHD": 0.087,  # Dividend (Dow Jones)
    "COWZ": 0.064,  # Cash Cow
    "XLV": 0.053,  # Healthcare
    "CVX": 0.052,  # Energy
    "VXUS": 0.049,  # International
    "EWY": 0.100,  # South Korea
    "GLD": 0.073,  # Gold
    "BIL": 0.002,  # Cash/Bills
    "NVR": 0.008,  # Homebuilder
}

tickers = list(current_portfolio.keys())


def run_optimization():
    print(f"Downloading historical data for: {tickers}")
    try:
        # Get 5 years of data
        data = yf.download(tickers, period="5y")["Close"]
    except Exception as e:
        print(f"Error: {e}")
        return

    # Filter out tickers with missing data if any
    data = data.dropna(axis=1, how="all").dropna()
    valid_tickers = data.columns.tolist()

    returns = data.pct_change().dropna()

    # Annualized stats
    mu = returns.mean() * 252
    cov = returns.cov() * 252

    # Map current weights to filtered tickers
    curr_w_raw = np.array([current_portfolio[t] for t in valid_tickers])
    curr_w = curr_w_raw / np.sum(curr_w_raw)  # Normalize

    def get_stats(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        sharpe = (ret - 0.04) / vol
        return ret, vol, sharpe

    # Optimization: Max Sharpe
    def objective(w):
        _, _, sharpe = get_stats(w)
        return -sharpe

    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    # Bounds: Diversification limit (Min 1%, Max 20%)
    bounds = tuple((0.01, 0.20) for _ in range(len(valid_tickers)))

    res = minimize(objective, curr_w, method="SLSQP", bounds=bounds, constraints=cons)

    if not res.success:
        print("Optimization failed:", res.message)
        return

    opt_w = res.x

    c_ret, c_vol, c_sharpe = get_stats(curr_w)
    o_ret, o_vol, o_sharpe = get_stats(opt_w)

    print("\n--- Optimization Report (Max Sharpe Ratio Strategy) ---")
    print(f"Metrics         Current    Optimized  Diff")
    print(
        f"Exp. Return:    {c_ret * 100:7.2f}%   {o_ret * 100:7.2f}%   {(o_ret - c_ret) * 100:+.2f}%"
    )
    print(
        f"Volatility:     {c_vol * 100:7.2f}%   {o_vol * 100:7.2f}%   {(o_vol - c_vol) * 100:+.2f}%"
    )
    print(
        f"Sharpe Ratio:   {c_sharpe:7.3f}   {o_sharpe:7.3f}   {(o_sharpe - c_sharpe):+.3f}"
    )

    report = pd.DataFrame(
        {
            "Ticker": valid_tickers,
            "Current (%)": curr_w * 100,
            "Optimized (%)": opt_w * 100,
            "Delta (%)": (opt_w - curr_w) * 100,
        }
    )

    print("\n[Proposed Allocation Changes]")
    print(
        report.sort_values(by="Delta (%)", ascending=False).to_string(
            index=False,
            formatters={
                "Current (%)": "{:,.2f}".format,
                "Optimized (%)": "{:,.2f}".format,
                "Delta (%)": "{:+.2f}".format,
            },
        )
    )


if __name__ == "__main__":
    run_optimization()
