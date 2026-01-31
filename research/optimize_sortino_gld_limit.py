import os
import pandas as pd
import numpy as np
import scipy.optimize as sco

# --- CONFIGURATION ---
DATA_DIR = r"d:\gg\data\historical"
START_DATE = "2020-01-01"
RISK_FREE_RATE = 0.04

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
        "DBMF",
        "JEPI",
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
    """Load and align all eligible ETF data (Prioritize Time Continuity)"""
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
            if "Close" in df.columns:
                series = df["Close"]
            elif "Adj Close" in df.columns:
                series = df["Adj Close"]
            else:
                continue

            # Filter start date (pre-check)
            if not series.empty:
                # Check if starts roughly around 2010
                if series.index[0] <= pd.Timestamp(START_DATE) + pd.Timedelta(days=10):
                    price_data[ticker] = series
        except:
            pass

    # Align data
    valid_tickers = list(price_data.keys())
    print(f"  Assets with history since {START_DATE}: {len(valid_tickers)}")

    if not valid_tickers:
        return pd.DataFrame()

    df_all = pd.DataFrame({t: price_data[t] for t in valid_tickers})
    df_all = df_all.sort_index()

    # Filter to desired date range
    df_all = df_all[df_all.index >= START_DATE]

    # Forward fill gaps
    df_all = df_all.ffill()

    # Drop assets (columns) that still have NaNs (started late or big gaps)
    df_all = df_all.dropna(axis=1)

    print(f" aligned assets: {df_all.shape[1]} (Rows: {df_all.shape[0]})")
    return df_all


def portfolio_sortino(weights, returns, risk_free_rate=0.04):
    """
    Calculate Sortino Ratio for a portfolio
    Sortino = (Mean Portfolio Return - Rf) / Downside Deviation
    """
    # Portfolio returns series
    port_returns = returns.dot(weights)

    # Annualized Return
    mean_return = port_returns.mean() * 252

    # Downside Deviation
    # Target return = 0 (or Rf/252, but usually 0 for Sortino is standard or MAR)
    # Let's use 0 as MAR for downside
    target = 0
    downside_returns = port_returns[port_returns < target]

    if len(downside_returns) == 0:
        return 0  # No downside? Infinite Sortino

    downside_std = np.sqrt((downside_returns**2).mean()) * np.sqrt(252)

    if downside_std == 0:
        return 0

    sortino = (mean_return - risk_free_rate) / downside_std
    return sortino


def neg_sortino(weights, returns):
    return -portfolio_sortino(weights, returns, RISK_FREE_RATE)


def optimize_portfolio(df):
    returns = df.pct_change().dropna()
    num_assets = len(df.columns)
    tickers = df.columns.tolist()

    args = (returns,)

    # Constraints
    constraints = []
    # 1. Sum of weights = 1
    constraints.append({"type": "eq", "fun": lambda x: np.sum(x) - 1})

    # 2. GLD <= 15%
    if "GLD" in tickers:
        gld_idx = tickers.index("GLD")
        print(f"  Adding GLD <= 15% constraint (Index: {gld_idx})")
        # Ineq constraint: cons(x) >= 0
        # 0.15 - x[gld] >= 0  => x[gld] <= 0.15
        constraints.append({"type": "ineq", "fun": lambda x: 0.15 - x[gld_idx]})
    else:
        print("  WARNING: GLD not found in universe, constraint ignored.")

    bounds = tuple((0.0, 1.0) for asset in range(num_assets))

    # Initial guess
    init_guess = num_assets * [
        1.0 / num_assets,
    ]

    print("Running Optimization (Max Sortino)...")
    result = sco.minimize(
        neg_sortino,
        init_guess,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    return result.x, tickers


def dca_backtest(df, weights_dict):
    """Monthly DCA Backtest"""
    data = df.copy()
    cash = 0.0
    holdings = {t: 0.0 for t in weights_dict.keys()}
    total_invested = 0.0
    equity_curve = []

    last_rebalance_year = -1
    last_month = -1
    investment_amount = 1000.0

    for date, row in data.iterrows():
        # Valuation
        current_val = cash
        for t, qty in holdings.items():
            current_val += qty * row[t]

        # Monthly Inject
        if date.month != last_month:
            cash += investment_amount
            total_invested += investment_amount
            last_month = date.month

            # Buy immediately
            for t, w in weights_dict.items():
                alloc = investment_amount * w
                price = row[t]
                if price > 0:
                    holdings[t] += alloc / price
            cash = 0

        # Annual Rebalance
        if date.year != last_rebalance_year:
            total_pv = 0
            for t, qty in holdings.items():
                total_pv += qty * row[t]

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
    # 1. Load
    df = load_data()
    if df.empty:
        return

    # 2. Optimize
    opt_weights, tickers = optimize_portfolio(df)

    raw_weights = dict(zip(tickers, opt_weights))
    final_weights = {k: v for k, v in raw_weights.items() if v > 0.01}  # >1%
    total_w = sum(final_weights.values())
    final_weights = {k: v / total_w for k, v in final_weights.items()}

    print("\n=== Optimized Max Sortino Portfolio (GLD <= 15%) ===")
    print("-" * 40)
    for t, w in sorted(final_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"{t}: {w * 100:.1f}%")
    print("-" * 40)

    # 3. Backtest
    print("\nRunning DCA Backtest...")
    res = dca_backtest(df, final_weights)

    final_eq = res["Equity"].iloc[-1]
    invested = res["Invested"].iloc[-1]
    multiple = final_eq / invested

    peak = res["Equity"].cummax()
    mdd = ((res["Equity"] - peak) / peak).min()

    # Portfolio Stats (Price-based for theoretical correctness)
    price_ret = df[list(final_weights.keys())].pct_change().dropna()
    w_vec = np.array([final_weights[t] for t in final_weights.keys()])
    port_ret = price_ret.dot(w_vec)

    ann_ret = port_ret.mean() * 252
    sortino = portfolio_sortino(w_vec, price_ret, RISK_FREE_RATE)

    print(f"\n[Performance Verification]")
    print(f"Final Equity: ${final_eq:,.0f} ({multiple:.2f}x)")
    print(f"MDD:          {mdd * 100:.2f}%")
    print(f"Sortino:      {sortino:.2f}")
    print(f"CAGR:         {ann_ret * 100:.2f}%")


if __name__ == "__main__":
    main()
