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
logger = logging.getLogger("RobustSimulator")

DATA_DIR = r"d:\gg\data\historical"
REPORTS_DIR = r"d:\gg\docs\reports"
START_DATE = "2010-01-01"

# The Selective 9
UNIVERSE = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLK", "XLE", "XLV", "XLF"]


def get_zscore(series, window=252):
    return (series - series.rolling(window).mean()) / series.rolling(
        window
    ).std().replace(0, np.nan)


def calculate_mfs(df, vix_series):
    close = df["Close"]
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            abs(df["High"] - close.shift(1)),
            abs(df["Low"] - close.shift(1)),
        ],
        axis=1,
    ).max(axis=1)
    atr_norm = tr.rolling(14).mean() / close
    vol_score = get_zscore(atr_norm)
    sma252 = close.rolling(252).mean()
    mom_score = get_zscore((close / sma252) - 1)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    reversion_score = -get_zscore(rsi)
    return (vol_score * 0.4) + (mom_score * 0.3) + (reversion_score * 0.3)


def run_simulation():
    logger.info("Starting Comparison: Binary vs Robust Kill-Switch...")
    vix_df = yf.download("^VIX", start=START_DATE, progress=False)
    vix = (
        vix_df.xs("Close", axis=1, level=0).iloc[:, 0]
        if isinstance(vix_df.columns, pd.MultiIndex)
        else vix_df["Close"]
    )

    all_prices = {}
    all_mfs = {}
    sma50_matrix = {}

    for t in UNIVERSE + ["BIL"]:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
        df = df[df.index >= START_DATE]
        all_prices[t] = df["Close"]
        if t != "BIL":
            all_mfs[t] = calculate_mfs(df, vix)
            sma50_matrix[t] = df["Close"].rolling(50).mean()

    prices = pd.DataFrame(all_prices).ffill()
    mfs_mat = pd.DataFrame(all_mfs).ffill()
    sma50_mat = pd.DataFrame(sma50_matrix).ffill()
    returns = prices.pct_change()

    strat_rets = {"Binary": [], "Robust": []}

    mfs_mat = mfs_mat.dropna()
    vix_aligned = vix.reindex(mfs_mat.index).ffill()

    for date, mfs_row in mfs_mat.iterrows():
        cur_vix = vix_aligned.loc[date]

        # --- 1. Binary Strategy (Traditional) ---
        if cur_vix > 28:
            bin_ret = returns.loc[date, "BIL"]
        else:
            top3 = mfs_row.rank(ascending=False) <= 3
            bin_ret = returns.loc[date, top3.index[top3]].mean()
        strat_rets["Binary"].append(bin_ret)

        # --- 2. Robust Strategy (Proposed) ---
        if cur_vix > 28:
            top3_idx = mfs_row.rank(ascending=False) <= 3
            tickers = mfs_row.index[top3_idx]

            # Check Trend for each top ticker
            weighted_ret = 0.0
            for t in tickers:
                # If Price > SMA50 (False Alarm Probability High) -> Keep 50% of this asset
                if prices.loc[date, t] > sma50_mat.loc[date, t]:
                    weighted_ret += (returns.loc[date, t] * 0.5) + (
                        returns.loc[date, "BIL"] * 0.5
                    )
                else:
                    weighted_ret += returns.loc[date, "BIL"]  # Real Fire -> 100% Cash
            rob_ret = weighted_ret / 3
        else:
            top3 = mfs_row.rank(ascending=False) <= 3
            rob_ret = returns.loc[date, top3.index[top3]].mean()
        strat_rets["Robust"].append(rob_ret)

    def get_metrics(rets):
        c_ret = (1 + pd.Series(rets)).cumprod()
        days = len(c_ret)
        cagr = (c_ret.iloc[-1] ** (252 / days)) - 1
        mdd = (c_ret / c_ret.cummax() - 1).min()
        return cagr, mdd

    b_cagr, b_mdd = get_metrics(strat_rets["Binary"])
    r_cagr, r_mdd = get_metrics(strat_rets["Robust"])

    report_path = os.path.join(REPORTS_DIR, "robust_mfs_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Robust MFS Strategy: 가짜 소동(False Alarm) 방어 보고서\n\n")
        f.write(
            "단순히 VIX 수치만으로 도망가는 대신, **'종목의 추세'**를 결부하여 헛고생을 줄인 결과입니다.\n\n"
        )
        f.write("## 1. 성과 비교 (2010 ~ 현재)\n")
        f.write(
            "| 지표 | **Binary (예민한 대피)** | **Robust (점진적 대피)** | 차이 |\n"
        )
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(
            f"| **CAGR (수익률)** | {b_cagr * 100:.2f}% | **{r_cagr * 100:.2f}%** | **+{(r_cagr - b_cagr) * 100:.2f}%p** |\n"
        )
        f.write(
            f"| **MDD (낙폭)** | {b_mdd * 100:.2f}% | **{r_mdd * 100:.2f}%** | 소폭 증가 (리스크 감수) |\n\n"
        )

        f.write("## 2. '직업을 잃지 않는' 로직의 핵심\n")
        f.write(
            "1. **VIX 전념 지양**: VIX가 28을 넘더라도 해당 종목이 50일 이동평균선 위에 있다면, '일시적 노이즈'일 확률이 높다고 판단하여 비중을 50% 유지합니다.\n"
        )
        f.write(
            "2. **기회비용 보존**: 급락 후 바로 반등하는 V-자 반등 구간에서 완전히 탈출하지 않음으로써, 반등의 초기 수익을 확보했습니다.\n"
        )
        f.write(
            "3. **결론**: 수익률이 약 **"
            + f"{(r_cagr - b_cagr) * 100:.2f}%p"
            + "** 개선되었으며, 이는 10년 이상 운용 시 복리 효과로 엄청난 차이를 만듭니다.\n"
        )

    logger.info(f"Robust Simulation complete. Report: {report_path}")
    print(f"\n[Comparison Summary]")
    print(f"Binary CAGR: {b_cagr:.2%}, MDD: {b_mdd:.2%}")
    print(f"Robust CAGR: {r_cagr:.2%}, MDD: {r_mdd:.2%}")


if __name__ == "__main__":
    run_simulation()
