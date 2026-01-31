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
logger = logging.getLogger("AnnualShowdown")

# Realistic Cost Settings
SLIPPAGE = 0.001  # 0.1%
COMMISSION = 0.0002  # 0.02%
CAPITAL_GAINS_TAX = 0.22

DATA_DIR = r"d:\gg\data\historical"
REPORTS_DIR = r"d:\gg\docs\reports"
START_DATE = "2010-01-01"

# Asset Groups
EQUITIES = ["SPY", "QQQ", "IWM", "XLK", "XLV", "XLP"]
DEFENSIVE = ["TLT", "GLD", "BIL"]
ALL_ASSETS = list(set(EQUITIES + DEFENSIVE))


def run_strategy(name, prices, weight_func):
    """
    Runs a strategy with annual rebalancing and returns metrics.
    """
    returns = prices.pct_change()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    current_weights = {}

    # Yearly Rebalancing
    for year, group in prices.groupby(prices.index.year):
        reb_date = group.index[0]
        # Get target weights for this year
        target_weights = weight_func(prices, reb_date)

        for date in group.index:
            for t, w in target_weights.items():
                weights.at[date, t] = w

    weights = weights.shift(1).fillna(0)
    weight_changes = weights.diff().abs().sum(axis=1)
    tx_costs = weight_changes * (SLIPPAGE + COMMISSION)

    daily_gross_ret = (returns * weights).sum(axis=1)
    daily_net_ret = daily_gross_ret - tx_costs

    # Tax Calculation
    current_val = 1.0
    for year, group in daily_net_ret.groupby(daily_net_ret.index.year):
        year_start = current_val
        for r in group.dropna():
            current_val *= 1 + r
        profit = current_val - year_start
        if profit > 0:
            current_val -= profit * CAPITAL_GAINS_TAX

    days = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr_final = (current_val ** (1 / days)) - 1
    mdd_net = (
        (1 + daily_net_ret).cumprod() / (1 + daily_net_ret).cumprod().cummax() - 1
    ).min()

    return {"name": name, "cagr": cagr_final, "mdd": mdd_net}


# --- Strategy Weight Functions ---


def weight_60_40(prices, date):
    return {"SPY": 0.6, "TLT": 0.4}


def weight_permanent(prices, date):
    return {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "BIL": 0.25}


def weight_all_weather_simple(prices, date):
    return {"SPY": 0.3, "TLT": 0.4, "IWM": 0.15, "GLD": 0.075, "XLK": 0.075}


def weight_momentum_annual(prices, date):
    # Pick Top 3 performers from last year
    prev_year = date.year - 1
    past_prices = prices[prices.index.year == prev_year]
    if past_prices.empty:
        return {"SPY": 1.0}

    perf = (past_prices.iloc[-1] / past_prices.iloc[0]) - 1
    top3 = perf.sort_values(ascending=False).head(3).index.tolist()
    return {t: 1 / 3 for t in top3}


def main():
    logger.info("Starting Annual Strategy Showdown...")

    all_prices = {}
    for t in ALL_ASSETS:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
        df = df[df.index >= START_DATE]
        all_prices[t] = df["Close"]

    prices = pd.DataFrame(all_prices).ffill()

    results = []
    results.append(run_strategy("60/40 (Classic)", prices, weight_60_40))
    results.append(run_strategy("Permanent Portfolio", prices, weight_permanent))
    results.append(
        run_strategy("All Weather (Simple)", prices, weight_all_weather_simple)
    )
    results.append(
        run_strategy("Annual Momentum (Top 3)", prices, weight_momentum_annual)
    )

    # Benchmark: SPY Buy & Hold (Taxes only at end or annually)
    # For fair comparison, we calculate SPY's annual tax as well
    results.append(run_strategy("SPY Buy & Hold", prices, lambda p, d: {"SPY": 1.0}))

    report_path = os.path.join(REPORTS_DIR, "annual_showdown_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Annual Strategy Showdown: 가장 이상적인 연간 리밸런싱 모델\n\n")
        f.write(
            "모든 비용(슬리피지 0.1%, 양도세 22%)을 차감한 후, 1년에 단 한 번의 움직임으로 최고의 성과를 내는 모델을 찾습니다.\n\n"
        )

        f.write("## 1. 전수 시뮬레이션 결과 (2010 ~ 현재)\n")
        f.write("| 전략명 | **세후 CAGR (실질)** | **MDD (낙폭)** | 특징 |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for res in sorted(results, key=lambda x: x["cagr"], reverse=True):
            f.write(
                f"| {res['name']} | **{res['cagr'] * 100:.2f}%** | {res['mdd'] * 100:.2f}% | {'최고 수익' if res == max(results, key=lambda x: x['cagr']) else '-'} |\n"
            )

        f.write("\n## 2. '가장 이상적인' 방식의 조건\n")
        f.write(
            "1. **낮은 회전율**: 연간 리밸런싱은 슬리피지를 연 1회로 제한하여 '복리 훼손'을 원천 차단합니다.\n"
        )
        f.write(
            "2. **세금 이연 효과**: 잦은 매매로 세금을 매번 내는 대신, 연간 단위로 정산함으로써 실질적인 '절세 복리'를 누립니다.\n"
        )
        f.write("3. **데이터의 증명**: (시뮬레이션 결과에 따라 업데이트 예정)\n")

    logger.info(f"Showdown complete. Report: {report_path}")
    print("\n[Showdown Results Summary]")
    for res in results:
        print(f"{res['name']}: CAGR {res['cagr']:.2%}, MDD {res['mdd']:.2%}")


if __name__ == "__main__":
    main()
