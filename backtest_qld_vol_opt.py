import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def run_backtest():
    # 1. Tickers & Pools
    core_assets = {"QLD": 0.45, "SPY": 0.20, "^KS200": 0.20, "GLD": 0.15}

    # 1x Defensive Pool (Pure)
    pool_1x = [
        "BTAL",
        "XLP",
        "XLU",
        "GLD",
        "FXY",
        "UUP",
        "MNA",
        "QAI",
        "DBC",
        "USFR",
        "GSY",
        "PFIX",
        "DBMF",
        "TAIL",
        "IVOL",
        "KMLM",
        "CTA",
        "PDBC",
        "SCHP",
        "BIL",
        "TLT",
        "IEF",
    ]

    # Leveraged Defensive Pool (Including 2x)
    pool_2x = pool_1x + [
        "NTSX",
        "UBT",
        "UST",
    ]  # UBT: 2x 20Y+, UST: 2x 7-10Y, NTSX: 1.5x Stock/Bond

    all_tickers = list(core_assets.keys()) + list(set(pool_2x))

    # 2. Data Fetching
    print(f"Fetching data for {len(all_tickers)} tickers...")
    raw_data = yf.download(
        all_tickers, start="2018-01-01", end="2026-12-31", progress=False
    )

    # Handle multi-index columns robustly
    data_dict = {}
    if isinstance(raw_data.columns, pd.MultiIndex):
        # Default yfinance format is (Field, Ticker)
        fields = raw_data.columns.get_level_values(0).unique()
        available_tickers = raw_data.columns.get_level_values(1).unique()

        for ticker in all_tickers:
            if ticker in available_tickers:
                if "Adj Close" in fields:
                    data_dict[ticker] = raw_data["Adj Close"][ticker]
                elif "Close" in fields:
                    data_dict[ticker] = raw_data["Close"][ticker]
    else:
        # Flat columns
        for ticker in all_tickers:
            if ticker in raw_data.columns:
                data_dict[ticker] = raw_data[ticker]

    data = pd.DataFrame(data_dict)

    if data.empty:
        print("Error: Downloaded data is empty.")
        return

    required = ["QLD", "SPY", "BIL"]
    missing = [r for r in required if r not in data.columns]
    if missing:
        print(f"Error: Missing required columns: {missing}")
        print(f"Available columns: {data.columns.tolist()}")
        return

    data = data.ffill().dropna(subset=required)

    # 3. Strategy Logic (Dual SMA 110/250 Hysteresis)
    print("Calculating signals...")
    qqq_raw = yf.download("QQQ", start="2017-01-01", end="2026-12-31", progress=False)
    if isinstance(qqq_raw.columns, pd.MultiIndex):
        qqq = (
            qqq_raw["Adj Close"]["QQQ"]
            if "Adj Close" in qqq_raw.columns.get_level_values(0)
            else qqq_raw["Close"]["QQQ"]
        )
    else:
        qqq = (
            qqq_raw["Adj Close"] if "Adj Close" in qqq_raw.columns else qqq_raw["Close"]
        )

    ma110 = qqq.rolling(110).mean()
    ma250 = qqq.rolling(250).mean()

    signal = pd.Series(index=data.index)
    curr_status = "NORMAL"

    for date in data.index:
        if date not in qqq.index:
            continue
        price = qqq.loc[date]
        m1 = ma110.loc[date]
        m2 = ma250.loc[date]

        if price > m1 and price > m2:
            curr_status = "NORMAL"
        elif price < m1 and price < m2:
            curr_status = "DANGER"
        # else: keep previous status

        signal.loc[date] = curr_status

    # 4. Defensive Selection Logic
    def get_ensemble_returns(date_idx, pool, lookback=168):
        if date_idx < lookback:
            return ["BIL"]

        # Look-back window to calculate momentum
        subset = data.iloc[date_idx - lookback : date_idx]
        returns = (subset.iloc[-1] / subset.iloc[0]) - 1

        valid_pool = [t for t in pool if t in returns.index]
        mom_rank = returns[valid_pool].dropna().sort_values(ascending=False)

        # Absolute filter (positive momentum)
        top_assets = mom_rank[mom_rank > 0].head(3).index.tolist()

        if not top_assets:
            return ["BIL"]
        return top_assets

    # 5. Backtest Loop
    results = {}

    for scenario_name, pool in [("Pure_1x", pool_1x), ("Leveraged_2x", pool_2x)]:
        print(f"Running Scenario: {scenario_name}...")
        portfolio_returns = []

        for i in range(1, len(data)):
            date = data.index[i]
            sig = signal.iloc[i - 1]

            # Daily returns
            daily_ret = data.iloc[i] / data.iloc[i - 1] - 1

            # Allocation
            if sig == "NORMAL":
                # Core Allocation
                daily_p_ret = sum(daily_ret[t] * core_assets[t] for t in core_assets)
            else:
                # Defensive Switch (Tactical part only)
                def_assets = get_ensemble_returns(i - 1, pool)
                def_weight = 0.45 / len(def_assets)

                def_ret_sum = sum(daily_ret[t] * def_weight for t in def_assets)
                core_rem_ret = (
                    daily_ret["SPY"] * 0.20
                    + daily_ret["^KS200"] * 0.20
                    + daily_ret["GLD"] * 0.15
                )
                daily_p_ret = def_ret_sum + core_rem_ret

            portfolio_returns.append(daily_p_ret)

        # Metrics Calculation
        port_ret_series = pd.Series(portfolio_returns, index=data.index[1:])
        cum_ret = (1 + port_ret_series).cumprod()

        cagr = (cum_ret.iloc[-1] ** (252 / len(port_ret_series))) - 1
        mdd = (cum_ret / cum_ret.cummax() - 1).min()
        vol = port_ret_series.std() * np.sqrt(252)
        sharpe = (cagr - 0.03) / vol  # Rf = 3%

        results[scenario_name] = {
            "CAGR": cagr,
            "MDD": mdd,
            "Vol": vol,
            "Sharpe": sharpe,
            "EquityCurve": cum_ret,
        }

    # 6. Output & Visualization
    print("\n" + "=" * 40)
    print("BACKTEST RESULTS: 1x vs 2x Defensive Ensemble")
    print("=" * 40)
    for s in results:
        print(f"[{s}]")
        print(f" - CAGR  : {results[s]['CAGR'] * 100:.2f}%")
        print(f" - MDD   : {results[s]['MDD'] * 100:.2f}%")
        print(f" - Sharpe: {results[s]['Sharpe']:.3f}")
        print("-" * 20)

    # Plotting
    plt.figure(figsize=(12, 7))
    for s in results:
        plt.plot(
            results[s]["EquityCurve"], label=f"{s} (Sharpe: {results[s]['Sharpe']:.2f})"
        )

    plt.title("QLD Tactical Strategy: 1x vs 2x Defensive Ensemble")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("backtest_qld_vol_opt.png")

    # Save CSV
    summary_df = pd.DataFrame(
        {
            "Metric": ["CAGR", "MDD", "Vol", "Sharpe"],
            "Pure_1x": [
                results["Pure_1x"]["CAGR"],
                results["Pure_1x"]["MDD"],
                results["Pure_1x"]["Vol"],
                results["Pure_1x"]["Sharpe"],
            ],
            "Leveraged_2x": [
                results["Leveraged_2x"]["CAGR"],
                results["Leveraged_2x"]["MDD"],
                results["Leveraged_2x"]["Vol"],
                results["Leveraged_2x"]["Sharpe"],
            ],
        }
    )
    summary_df.to_csv("backtest_qld_vol_opt_results.csv", index=False)
    print(
        f"\nResults saved to 'backtest_qld_vol_opt_results.csv' and 'backtest_qld_vol_opt.png'"
    )


if __name__ == "__main__":
    run_backtest()
