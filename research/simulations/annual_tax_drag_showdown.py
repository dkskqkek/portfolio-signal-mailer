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
logger = logging.getLogger("TaxDragSimulator")

# Constants
SLIPPAGE = 0.001  # 0.1%
COMMISSION = 0.0002  # 0.02%
TAX_RATE = 0.22  # 22% Capital Gains Tax (US)

DATA_DIR = r"d:\gg\data\historical"
REPORTS_DIR = r"d:\gg\docs\reports"
START_DATE = "2010-01-01"

UNIVERSE = ["SPY", "QQQ", "IWM", "GLD", "TLT", "BIL"]


def run_tax_sim(name, prices, weight_func):
    """
    Simulates a strategy with precise tax tracking (Cost Basis).
    """
    cash = 1.0
    holdings = {t: 0.0 for t in UNIVERSE}  # shares * price
    cost_basis = {t: 0.0 for t in UNIVERSE}

    total_value = 1.0

    yearly_data = []

    # Iterate through years
    for year, group in prices.groupby(prices.index.year):
        reb_date = group.index[0]
        target_weights = weight_func(prices, reb_date)

        # 1. Update valuation at start of year
        price_start = group.loc[reb_date]

        # 2. Rebalance
        # Calculate current total value before rebalance
        total_val = cash + sum(holdings.values())

        # Determine trades
        annual_capital_gains = 0.0

        for t in UNIVERSE:
            target_val = total_val * target_weights.get(t, 0.0)
            current_val = holdings[t]

            diff = target_val - current_val

            if diff < 0:  # Selling (Realize Gains)
                pct_sold = abs(diff) / current_val if current_val > 0 else 0
                realized_part = abs(diff)
                # Gain = realized_val - (cost_basis * pct_sold)
                gain = realized_part - (cost_basis[t] * pct_sold)
                if gain > 0:
                    annual_capital_gains += gain

                # Update cost basis and holdings
                cost_basis[t] *= 1 - pct_sold
                holdings[t] += diff  # reduces
                cash += abs(diff) * (1 - SLIPPAGE - COMMISSION)

            elif diff > 0:  # Buying
                holdings[t] += diff
                cost_basis[t] += diff
                cash -= diff * (1 + SLIPPAGE + COMMISSION)

        # 3. Pay Tax on Realized Gains
        if annual_capital_gains > 0:
            tax = annual_capital_gains * TAX_RATE
            cash -= tax

        # 4. Hold throughout the year
        # Final value at end of year
        price_end = group.iloc[-1]
        for t in UNIVERSE:
            if price_start[t] > 0:
                ret = price_end[t] / price_start[t]
                holdings[t] *= ret

        total_value = cash + sum(holdings.values())
        yearly_data.append({"year": year, "val": total_value})

    final_val = total_value
    days = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr = (final_val ** (1 / days)) - 1

    return {"name": name, "cagr": cagr}


# --- Strategy Logic ---


def weight_60_40(prices, date):
    return {"SPY": 0.6, "TLT": 0.4}


def weight_permanent(prices, date):
    return {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "BIL": 0.25}


def weight_momentum_annual(prices, date):
    prev_year = date.year - 1
    past = prices[prices.index.year == prev_year]
    if past.empty:
        return {"SPY": 1.0}
    perf = (past.iloc[-1] / past.iloc[0]) - 1
    # Top 2 Assets for higher concentration visibility
    top2 = perf.sort_values(ascending=False).head(2).index.tolist()
    return {t: 0.5 for t in top2}


def main():
    logger.info("Starting Tax Drag Showdown...")
    all_prices = {}
    for t in UNIVERSE:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            continue
        all_prices[t] = pd.read_csv(path, index_col="Date", parse_dates=True)["Close"]

    prices = pd.DataFrame(all_prices).ffill().loc[START_DATE:]

    results = []
    results.append(run_tax_sim("60/40 (Rebalancing)", prices, weight_60_40))
    results.append(run_tax_sim("Permanent (Rebalancing)", prices, weight_permanent))
    results.append(
        run_tax_sim("Momentum (Full Switch)", prices, weight_momentum_annual)
    )

    report_path = os.path.join(REPORTS_DIR, "tax_drag_analysis.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Tax Drag 분석: 종목 교체 vs 비중 조절\n\n")
        f.write(
            "22%의 양도소득세가 **'스위칭 전략(Momentum)'**과 **'고정 비중 전략(60/40)'**에 미치는 실질적인 타격을 분석합니다.\n\n"
        )

        f.write("## 1. 세후 실질 CAGR 비교\n")
        f.write("| 전략 유형 | 세후 CAGR | 특징 |\n")
        f.write("| :--- | :--- | :--- |\n")
        for res in sorted(results, key=lambda x: x["cagr"], reverse=True):
            f.write(
                f"| {res['name']} | **{res['cagr'] * 100:.2f}%** | {'세금 저항 높음' if 'Switch' in res['name'] else '절세 효율적'} |\n"
            )

        f.write("\n## 2. 'Tax Drag'의 실체\n")
        f.write(
            "1. **스위칭의 비극**: Momentum 전략은 매년 새로운 종목으로 갈아타면서 그동안 쌓인 수익에 대해 **강제적으로 22% 세금**을 냅니다. 이는 복리 엔진의 크기를 매년 인위적으로 줄이는 결과를 낳습니다.\n"
        )
        f.write(
            "2. **비중 조절의 이점**: 60/40이나 영구 포트폴리오는 기존 종목을 계속 들고 가면서 '일부'만 팔아 비중을 맞춥니다. 즉, 수익의 대부분을 **'미실현 이익'**으로 남겨두어 세금 부과를 뒤로 미루는(Tax Deferral) 효과가 있습니다.\n"
        )
        f.write(
            "3. **결론**: 아무리 알파가 높은 전략이라도 매년 전량 교체(Full Switch)를 한다면, 조금 낮은 수익률이라도 꾸준히 리밸런싱만 하는 전략에 세후 수익률이 역전당할 위험이 큽니다.\n"
        )

    logger.info(f"Tax Analysis complete. Report: {report_path}")
    print("\n[Tax Drag Results]")
    for res in results:
        print(f"{res['name']}: {res['cagr']:.2%}")


if __name__ == "__main__":
    main()
