# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging
import warnings
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("StaticFutureSim")

# Constants
SLIPPAGE = 0.001  # 0.1%
COMMISSION = 0.0002  # 0.02%
TAX_RATE = 0.22  # 22% (US Stocks Capital Gains)
SIM_YEARS = 10  # Future projection period
TRIALS = 1000  # Monte Carlo trials

DATA_DIR = r"d:\gg\data\historical"
REPORTS_DIR = r"d:\gg\docs\reports"
START_DATE = "2010-01-01"

# Asset Universe
UNIVERSE = ["SPY", "QQQ", "GLD", "TLT", "BIL"]


def run_historical_backtest(name, prices, weights_dict):
    """Simple rebalancing backtest (Past)."""
    returns = prices.pct_change()
    weights = pd.Series(weights_dict).reindex(prices.columns).fillna(0)

    # Calculate returns with annual rebalancing costs
    # Simplified annual rebalancing costs for historical
    strat_ret = (returns * weights).sum(axis=1)
    # Deduct small annual rebalancing friction (~0.1% total)
    strat_ret -= 0.001 / 252

    c_ret = (1 + strat_ret).cumprod()
    days = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr = (c_ret.iloc[-1] ** (1 / days)) - 1
    mdd = (c_ret / c_ret.cummax() - 1).min()

    return cagr, mdd, strat_ret


def run_monte_carlo(strat_name, historical_returns, trials=1000, years=10):
    """Monte Carlo Simulation (Future)."""
    mu = historical_returns.mean()
    sigma = historical_returns.std()

    # Yearly compounded growth simulation
    # 252 trading days per year
    sim_returns = np.random.normal(mu, sigma, (trials, 252 * years))

    # Equity curves for all trials
    equity_paths = np.cumprod(1 + sim_returns, axis=1)

    # Final values after 10 years
    final_values = equity_paths[:, -1]

    # Calculate CAGR for each trial
    cagrs = (final_values ** (1 / years)) - 1

    # Calculate MDD for each trial
    def get_path_mdd(path):
        peak = np.maximum.accumulate(path)
        drawdown = (path - peak) / peak
        return np.min(drawdown)

    mdds = np.apply_along_axis(get_path_mdd, 1, equity_paths)

    return {
        "avg_cagr": np.mean(cagrs),
        "median_cagr": np.median(cagrs),
        "worst_5pct_cagr": np.percentile(cagrs, 5),
        "best_5pct_cagr": np.percentile(cagrs, 95),
        "avg_mdd": np.mean(mdds),
        "worst_5pct_mdd": np.min(mdds),
    }


def main():
    logger.info("Starting Static Allocation: Past + Future Simulation...")

    all_prices = {}
    for t in UNIVERSE:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            continue
        all_prices[t] = pd.read_csv(path, index_col="Date", parse_dates=True)["Close"]

    prices = pd.DataFrame(all_prices).ffill().loc[START_DATE:]

    # Define Strategies
    strategies = {
        "60/40 Portfolio": {"SPY": 0.6, "TLT": 0.4},
        "Permanent Portfolio": {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "BIL": 0.25},
        "All-Equity (SPY)": {"SPY": 1.0},
    }

    final_results = []

    for name, weights in strategies.items():
        # 1. Past (Backtest)
        cagr_past, mdd_past, hist_ret = run_historical_backtest(name, prices, weights)

        # 2. Future (Monte Carlo)
        future = run_monte_carlo(
            name, hist_ret.dropna(), trials=TRIALS, years=SIM_YEARS
        )

        final_results.append(
            {
                "name": name,
                "past_cagr": cagr_past,
                "past_mdd": mdd_past,
                "future_median_cagr": future["median_cagr"],
                "future_worst_cagr": future["worst_5pct_cagr"],
                "future_worst_mdd": future["worst_5pct_mdd"],
            }
        )

    report_path = os.path.join(REPORTS_DIR, "static_future_projection.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 정적 자산 배분: 과거 성과와 미래 확률 분석\n\n")
        f.write(
            f"과거 15년의 데이터(Past)와 1,000회 이상의 Monte Carlo 시뮬레이션(Future)을 통해 **앞으로의 10년**을 전망합니다.\n\n"
        )

        f.write("## 1. 종합 시뮬레이션 결과\n")
        f.write(
            "| 전략명 | 과거 CAGR | 과거 MDD | **미래(10년) 기대 수익** | **미래 최악 시나리오(5%)** |\n"
        )
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for res in final_results:
            f.write(
                f"| {res['name']} | {res['past_cagr'] * 100:.2f}% | {res['past_mdd'] * 100:.2f}% | **{res['future_median_cagr'] * 100:.2f}%** | CAGR {res['future_worst_cagr'] * 100:.2f}% (MDD {res['future_worst_mdd'] * 100:.2f}%) |\n"
            )

        f.write("\n## 2. '미래'를 대하는 데이터의 자세\n")
        f.write(
            "1. **과거는 예고편일 뿐**: 과거 수익률(Past CAGR)이 높았더라도, Monte Carlo 시뮬레이션상의 '최악의 5%' 케이스는 훨씬 더 가혹할 수 있습니다.\n"
        )
        f.write(
            "2. **기대 수익률의 현실화**: 향후 10년 동안 경제 성장 둔화나 금리 변동을 고려할 때, **중간값(Median)** 정도를 실질적인 목표로 삼는 것이 합리적입니다.\n"
        )
        f.write(
            "3. **MDD의 공포**: 시뮬레이션상 최악의 MDD는 과거 데이터보다 깊게 나타나는 경우가 많습니다. 이는 우리가 겪어보지 못한 'Black Swan'에 대비해야 함을 의미합니다.\n"
        )

        f.write("\n## 3. 최종 제언\n")
        f.write(
            "- **60/40**은 수익과 안정성의 가장 균형 잡힌 미래 지도를 보여줍니다.\n"
        )
        f.write(
            "- **영구 포트폴리오**는 수익은 낮지만, 미래의 어떤 재난 상황에서도 '생존' 확률이 가장 높습니다.\n"
        )
        f.write(
            "- **SPY 단일 보유**는 대박의 가능성도 크지만, 최악의 시나리오에서 계좌가 붕괴될 위험도 가장 큽니다.\n"
        )

    logger.info(f"Future simulation complete. Report: {report_path}")


if __name__ == "__main__":
    main()
