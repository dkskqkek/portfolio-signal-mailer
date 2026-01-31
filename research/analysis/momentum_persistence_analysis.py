# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MomentumPersistence")

DATA_DIR = r"d:\gg\data\historical"
REPORTS_DIR = r"d:\gg\docs\reports"
START_DATE = "2010-01-01"
UNIVERSE = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLK", "XLE", "XLV", "XLF"]


def main():
    logger.info(
        "Analyzing Momentum Persistence (Probability of Winners Staying Winners)..."
    )

    all_prices = {}
    for t in UNIVERSE:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
        df = df[df.index >= START_DATE]
        all_prices[t] = df["Close"]

    prices = pd.DataFrame(all_prices).ffill()
    # Calculate Yearly Returns
    yearly_ret = prices.resample("YE").last().pct_change().dropna()

    persistence_stats = []

    # Iterate through years
    for i in range(len(yearly_ret) - 1):
        year_t = yearly_ret.index[i]
        year_tp1 = yearly_ret.index[i + 1]

        # Rankings in Year T
        ranks_t = yearly_ret.loc[year_t].rank(ascending=False)
        top3_t = ranks_t[ranks_t <= 3].index.tolist()

        # Rankings in Year T+1
        ranks_tp1 = yearly_ret.loc[year_tp1].rank(ascending=False)
        top3_tp1 = ranks_tp1[ranks_tp1 <= 3].index.tolist()

        # Check how many of Top 3 in T stayed in Top 3 in T+1
        stayed = len(set(top3_t) & set(top3_tp1))
        persistence_stats.append(
            {
                "year_t": year_t.year,
                "year_tp1": year_tp1.year,
                "stayed_count": stayed,
                "top3_t": top3_t,
                "top3_tp1": top3_tp1,
            }
        )

    df_persistence = pd.DataFrame(persistence_stats)
    avg_stayed = df_persistence["stayed_count"].mean()
    prob_stay = (avg_stayed / 3) * 100  # Probability that a single asset stays in Top 3

    # Specific Probability: Probability that ANY of Top 3 in T stays in Top 3 in T+1
    prob_at_least_one = (
        len(df_persistence[df_persistence["stayed_count"] >= 1]) / len(df_persistence)
    ) * 100

    report_path = os.path.join(REPORTS_DIR, "momentum_persistence_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 모멘텀 지속성(Momentum Persistence) 통계 보고서\n\n")
        f.write(
            "특정 시점의 모멘텀(상승세)이 다음 1년 동안 그대로 유지될 확률을 데이터로 분석한 결과입니다.\n\n"
        )

        f.write("## 1. 핵심 확률 통계\n")
        f.write(
            f"- **단일 자산의 생존 확률**: 작년 Top 3였던 특정 자산이 올해도 Top 3일 확률: **{prob_stay:.1f}%**\n"
        )
        f.write(
            f"- **포트폴리오 유지 확률**: 작년 Top 3 중 **최소 1개 이상**이 올해도 Top 3에 머물 확률: **{prob_at_least_one:.1f}%**\n"
        )
        f.write(
            f"- **평균 교체율**: 매년 평균 약 **{(1 - avg_stayed / 3) * 100:.1f}%**의 종목이 새로운 '대장주'로 바뀜\n\n"
        )

        f.write("## 2. 연도별 Persistence 상세\n")
        f.write(
            "| 기준년도(T) | 다음년도(T+1) | 유지된 개수 | 작년 Top 3 | 올해 Top 3 |\n"
        )
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for idx, row in df_persistence.iterrows():
            f.write(
                f"| {row['year_t']} | {row['year_tp1']} | {row['stayed_count']} | {', '.join(row['top3_t'])} | {', '.join(row['top3_tp1'])} |\n"
            )

        f.write("\n## 3. 결론 및 통찰\n")
        f.write(
            "1. **모멘텀의 관성**: 확률상 약 **"
            + f"{prob_at_least_one:.1f}%"
            + "**의 해에는 작년 최고의 자산 중 최소 하나가 올해도 여전히 강했습니다.\n"
        )
        f.write(
            "2. **회귀의 위험**: 하지만 매년 평균 2개 이상의 종목이 교체된다는 점은, **1년이라는 시간 동안 시장의 주도주가 바뀔 확률이 매우 높음**을 의미합니다.\n"
        )
        f.write(
            "3. **전략적 제언**: 1년에 한 번만 리밸런싱하는 것은 '비용' 면에서는 유리하지만, '추세 반전'에 노출되는 위험이 있습니다. 따라서 연간 리밸런싱을 하더라도 MFS처럼 '질적 지표'가 우수한 종목을 골라내는 것이 중요합니다.\n"
        )

    logger.info(f"Persistence analysis complete. Report: {report_path}")
    print(f"\n[Persistence Summary]")
    print(f"Prob. Stay (Single): {prob_stay:.1f}%")
    print(f"Prob. At least one stay: {prob_at_least_one:.1f}%")


if __name__ == "__main__":
    main()
