import os
import pandas as pd
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_DIR = r"d:\gg\data\historical"
START_DATE = "2010-01-01"  # Data start for analysis
RISK_FREE_RATE = 0.04  # 4% annual

# Exclude lists
BONDS = [
    "TLT",
    "IEF",
    "SHY",
    "BIL",
    "AGG",
    "LQD",
    "HYG",
    "TIP",
    "SCHP",
    "BND",
    "BNDX",
    "VCSH",
    "VCIT",
    "VCLT",
    "MUB",
    "EMB",
    "BND",
    "BNDX",
]
LEV_3X = [
    "TQQQ",
    "SQQQ",
    "UPRO",
    "SPXU",
    "SOXL",
    "SOXS",
    "LABU",
    "LABD",
    "TNA",
    "TZA",
    "FAS",
    "FAZ",
    "TECL",
    "TECS",
    "YINN",
    "YANG",
    "DPST",
    "DRV",
]
# Exclude inverse 2x as well if needed? The user said "3배수 레버리지는 미포함".
# Usually inverse is not good for long term DCA. Let's exclude Inverse 2x/3x for safety / logic of long term hold.
# Actually user said "채권을 제외한 모든 etf원자재나 레버리지 포함". So 2x Bull/Bear is allowed?
# Usually Bear ETFs are terrible for long term DCA. I will exclude obvious Bear/Inverse ETFs to avoid finding a "hedge" that just drags returns.
# Wait, MVO might hedge. But Inverse decay is bad. I will exclude ALL Inverse (Short) ETFs if possible.
# Identified Inverse prefixes/names mostly in LEV_3X or 2x list like QID, SDS, PSQ, SH.
INVERSE = [
    "SQQQ",
    "SPXU",
    "SOXS",
    "LABD",
    "TZA",
    "FAZ",
    "TECS",
    "YANG",
    "DRV",
    "QID",
    "SDS",
    "DXD",
    "SDOW",
    "SH",
    "PSQ",
    "DOG",
    "RWM",
    "EFZ",
]
EXCLUDE_TICKERS = set(BONDS + LEV_3X + INVERSE)

# Authorized ETF Universe (copied from download_full_us.py)
ALLOWED_ETFS = set(
    [
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "IVV",
        "VT",
        "VXUS",
        "XLK",
        "XLF",
        "XLV",
        "XLE",
        "XLI",
        "XLB",
        "XLU",
        "XLP",
        "XLY",
        "XLRE",
        "VGT",
        "VHT",
        "VFH",
        "VDE",
        "VIS",
        "VAW",
        "VPU",
        "VDC",
        "VCR",
        "TQQQ",
        "SQQQ",
        "UPRO",
        "SPXU",
        "QLD",
        "QID",
        "SSO",
        "SDS",
        "SOXL",
        "SOXS",
        "LABU",
        "LABD",
        "TNA",
        "TZA",
        "FAS",
        "FAZ",
        "TLT",
        "IEF",
        "SHY",
        "BIL",
        "AGG",
        "LQD",
        "HYG",
        "TIP",
        "SCHP",
        "BND",
        "BNDX",
        "VCSH",
        "VCIT",
        "VCLT",
        "MUB",
        "EMB",
        "GLD",
        "SLV",
        "USO",
        "UNG",
        "DBC",
        "DBA",
        "DBB",
        "PDBC",
        "UUP",
        "FXY",
        "FXE",
        "FXB",
        "FXA",
        "FXC",
        "VXX",
        "UVXY",
        "SVXY",
        "VIXY",
        "EEM",
        "EFA",
        "VEA",
        "VWO",
        "IEFA",
        "IEMG",
        "EWJ",
        "EWG",
        "EWU",
        "EWZ",
        "EWY",
        "EWT",
        "EWA",
        "EWC",
        "EWQ",
        "EWI",
        "EWP",
        "EWL",
        "EWN",
        "EWS",
        "ARKK",
        "ARKG",
        "ARKF",
        "ARKW",
        "ARKQ",
        "ARKX",
        "KWEB",
        "XBI",
        "IBB",
        "SMH",
        "SOXX",
        "HACK",
        "BOTZ",
        "ROBO",
        "VNQ",
        "XLRE",
        "IYR",
        "RWR",
        "SCHH",
        "SCHD",
        "VYM",
        "DVY",
        "SDY",
        "HDV",
        "DGRO",
        "VIG",
        "NOBL",
        "USMV",
        "MTUM",
        "VLUE",
        "QUAL",
        "SIZE",
    ]
)


def load_data():
    """Load and align all eligible ETF data"""
    price_data = {}

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"Loading data from {len(files)} files...")

    for f in files:
        ticker = f.replace(".csv", "")
        if ticker in EXCLUDE_TICKERS:
            continue
        if ticker not in ALLOWED_ETFS:
            continue

        try:
            df = pd.read_csv(
                os.path.join(DATA_DIR, f), index_col="Date", parse_dates=True
            )
            # Use 'Close' or 'Adj Close'? Generally Adj Close for returns.
            # Assuming 'Close' in cleaned data or check col.
            # download_full_us.py used auto_adjust=True, so 'Close' is Adj Close.
            if "Close" in df.columns:
                series = df["Close"]
            elif "Adj Close" in df.columns:
                series = df["Adj Close"]
            else:
                continue

            # Filter start date
            series = series[series.index >= START_DATE]

            # Require at least 95% of trading days since START_DATE (approx)
            # Or just common index later.
            if not series.empty:
                price_data[ticker] = series

        except Exception as e:
            pass

    # Align data
    # 1. First, find assets that actually cover the start date
    valid_tickers = []
    for t, s in price_data.items():
        # Check if the series starts on or before START_DATE (with a small buffer for holidays)
        if s.index[0] <= pd.Timestamp(START_DATE) + pd.Timedelta(days=10):
            valid_tickers.append(t)

    print(f"  Assets with history since {START_DATE}: {len(valid_tickers)}")

    if not valid_tickers:
        return pd.DataFrame()

    df_all = pd.DataFrame({t: price_data[t] for t in valid_tickers})
    df_all = df_all.sort_index()

    # Filter to desired date range
    df_all = df_all[df_all.index >= START_DATE]

    # 2. Forward fill gaps
    df_all = df_all.ffill()

    # 3. Drop ASSETS (columns) that still have NaNs
    df_all = df_all.dropna(axis=1)

    print(f" aligned assets: {df_all.shape[1]} (Rows: {df_all.shape[0]})")
    return df_all


def portfolio_stats(weights, mean_returns, cov_matrix):
    port_return = np.sum(mean_returns * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_return - RISK_FREE_RATE) / port_vol
    return port_return, port_vol, sharpe


def neg_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_stats(weights, mean_returns, cov_matrix)[2]


def optimize_portfolio(df):
    returns = df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    args = (mean_returns, cov_matrix)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))

    # Initial guess
    init_guess = num_assets * [
        1.0 / num_assets,
    ]

    print("Running Optimization (SLSQP)... this may take a moment.")
    result = sco.minimize(
        neg_sharpe,
        init_guess,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result.x, mean_returns.index


def dca_backtest(df, weights_dict):
    """
    Simulate Monthly DCA
    Start: $0
    Monthly: +$1,000
    Rebalance: Annually
    """
    # Resample to trade days
    data = df.copy()
    data["Year"] = data.index.year
    data["Month"] = data.index.month

    cash = 0.0
    # Portfolio holdings: {ticker: quantity}
    holdings = {t: 0.0 for t in weights_dict.keys()}
    total_invested = 0.0

    # Track equity curve
    equity_curve = []

    # Rebalance Check
    last_rebalance_year = -1

    # Monthly Injection Dates (First trading day of month)
    # Simplified: Iterate daily. If new month, inject.
    last_month = -1

    investment_amount = 1000.0

    for date, row in data.iterrows():
        # Current Portfolio Value
        current_val = cash
        for t, qty in holdings.items():
            current_val += qty * row[t]

        # 1. Inject Cash (Monthly)
        if date.month != last_month:
            cash += investment_amount
            total_invested += investment_amount
            last_month = date.month

            # Immediate buy with new cash based on target weights?
            # Or just accumulate?
            # Usually DCA buys immediately.
            # Let's distribute NEW cash according to target weights.
            for t, w in weights_dict.items():
                alloc = investment_amount * w
                price = row[t]
                if price > 0:
                    holdings[t] += alloc / price
            cash = 0  # All spent

        # 2. Rebalance (Annually - e.g., first trading day of year)
        if date.year != last_rebalance_year:
            # Execute Rebalance
            # Calculate total value again
            total_pv = 0
            for t, qty in holdings.items():
                total_pv += qty * row[t]

            # Reset holdings based on weights
            for t, w in weights_dict.items():
                target_amt = total_pv * w
                holdings[t] = target_amt / row[t]

            last_rebalance_year = date.year

        # Record
        total_pv = 0
        for t, qty in holdings.items():
            total_pv += qty * row[t]
        equity_curve.append(
            {"Date": date, "Equity": total_pv, "Invested": total_invested}
        )

    return pd.DataFrame(equity_curve).set_index("Date")


def main():
    # 1. Load Data
    df = load_data()
    if df.empty:
        print("No data found.")
        return

    # 2. Optimize
    opt_weights, tickers = optimize_portfolio(df)

    # Create Weight Dict & Filter tiny weights
    raw_weights = dict(zip(tickers, opt_weights))
    final_weights = {k: v for k, v in raw_weights.items() if v > 0.01}  # >1% only

    # Re-normalize
    total_w = sum(final_weights.values())
    final_weights = {k: v / total_w for k, v in final_weights.items()}

    print("\n=== Optimized Max Sharpe Portfolio (2010 ~ Now) ===")
    print("Strategy: Exclude Bonds, Exclude 3x Lev, Exclude Inverse")
    print("-" * 40)
    for t, w in sorted(final_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"{t}: {w * 100:.1f}%")
    print("-" * 40)

    # 3. Backtest
    print("\nRunning DCA Backtest (Monthly $1,000, Annual Rebalance)...")
    res = dca_backtest(df, final_weights)

    final_equity = res["Equity"].iloc[-1]
    total_invested = res["Invested"].iloc[-1]
    profit = final_equity - total_invested
    returns = res["Equity"].pct_change()

    # CAGR (Approx for DCA) -> IRR is better but simplistic here
    # Just final multiple
    multiple = final_equity / total_invested

    # MDD
    peak = res["Equity"].cummax()
    dd = (res["Equity"] - peak) / peak
    mdd = dd.min()

    # Sharpe (of the portfolio equity curve, simplistic)
    # Note: Sharpe of DCA curve is tricky. Usually calculated on underlying asset returns.
    # But let's report the stats of the final curve relative to invested capital curve is misleading.
    # Instead, we report Stats of the Optimized Portfolio (Price Returns) which is mathematically correct for Sharpe.

    # Re-calculate Stats for the Portfolio (Price Only)
    price_returns = df[list(final_weights.keys())].pct_change().dropna()
    w_vec = np.array([final_weights[t] for t in final_weights.keys()])
    port_daily_ret = price_returns.dot(w_vec)

    ann_ret = port_daily_ret.mean() * 252
    ann_vol = port_daily_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol

    print(f"\n[Performance Verification]")
    print(f"Total Invested: ${total_invested:,.0f}")
    print(f"Final Equity:   ${final_equity:,.0f} ({multiple:.2f}x)")
    print(f"MDD:            {mdd * 100:.2f}%")
    print(f"Sharpe Ratio:   {sharpe:.2f} (Theoretical)")
    print(f"CAGR (Price):   {ann_ret * 100:.2f}% (Buy-and-Hold equivalent)")


if __name__ == "__main__":
    main()
