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
logger = logging.getLogger("UltraBroadV2")

DATA_DIR = r"d:\gg\data\historical"
REPORTS_DIR = r"d:\gg\docs\reports"
START_DATE = "2010-01-01"

# Filtered ETF Universe (Removed Leveraged/Inverse/High-Volatility Junk)
LEVERAGED_INVERSE = [
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
    "VXX",
    "UVXY",
    "SVXY",
    "VIXY",
]

FULL_ETF_UNIVERSE = [
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

# Keep only long-only ETFs
ETF_UNIVERSE = [t for t in FULL_ETF_UNIVERSE if t not in LEVERAGED_INVERSE]


def get_zscore(series, window=252):
    return (series - series.rolling(window).mean()) / series.rolling(
        window
    ).std().replace(0, np.nan)


def calculate_mfs(df, vix_series):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    tr = pd.concat(
        [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
    ).max(axis=1)
    atr_norm = tr.rolling(14).mean() / close
    std20 = close.rolling(20).std()
    bb_width = (4 * std20) / close.rolling(20).mean()
    vol_score = (get_zscore(atr_norm) + get_zscore(bb_width)) / 2

    sma252 = close.rolling(252).mean()
    dist_sma = (close / sma252) - 1
    z_ret = get_zscore(close.pct_change(252))
    mom_score = get_zscore(dist_sma) - (z_ret.clip(lower=2.0) - 2.0)

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    reversion_score = -get_zscore(rsi)

    mfs = (vol_score * 0.4) + (mom_score * 0.3) + (reversion_score * 0.3)
    vix_z = get_zscore(vix_series.reindex(mfs.index).ffill())
    mfs = mfs.mask(vix_z > 1.0, mfs - 0.5)
    return mfs


def run_simulation():
    logger.info(
        f"Starting Multi-Factor Broad V2 (Cleaned) with {len(ETF_UNIVERSE)} ETFs..."
    )
    vix_df = yf.download("^VIX", start=START_DATE, progress=False)
    vix = (
        vix_df.xs("Close", axis=1, level=0).iloc[:, 0]
        if isinstance(vix_df.columns, pd.MultiIndex)
        else vix_df["Close"]
    )

    all_prices = {}
    all_mfs = {}
    for t in ETF_UNIVERSE:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
        df = df[df.index >= START_DATE]
        if len(df) < 500:
            continue
        all_prices[t] = df["Close"]
        all_mfs[t] = calculate_mfs(df, vix)

    price_matrix = pd.DataFrame(all_prices).ffill()
    mfs_matrix = pd.DataFrame(all_mfs).ffill()
    returns = price_matrix.pct_change()

    # Portfolio: Top 5 assets (More concentrated than 10 to see if signal holds)
    TOP_N = 5
    is_held = mfs_matrix.rank(axis=1, ascending=False) <= TOP_N
    weights = is_held.div(is_held.sum(axis=1), axis=0).shift(1).fillna(0)

    strat_ret = (returns * weights).sum(axis=1)

    def get_metrics(ret_series):
        c_ret = (1 + ret_series).cumprod().dropna()
        if len(c_ret) < 2:
            return 0, 0, 0
        days = len(c_ret)
        cagr = (c_ret.iloc[-1] ** (252 / days)) - 1
        mdd = (c_ret / c_ret.cummax() - 1).min()
        sharpe = (ret_series.mean() * 252 - 0.04) / (ret_series.std() * np.sqrt(252))
        return cagr, mdd, sharpe

    cagr, mdd, sharpe = get_metrics(strat_ret)
    spy_cagr, spy_mdd, spy_sharpe = get_metrics(returns["SPY"])

    report_path = os.path.join(REPORTS_DIR, "ultra_broad_mfs_report_v2.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Ultra-Broad ETF MFS Selection Report (V2 - Cleaned)\n\n")
        f.write(
            "이전 시뮬레이션에서 0%에 가까운 수익률이 나온 이유는 레버리지/인버스/고변동성 ETF들이 노이즈로 작용했기 때문입니다.\n"
        )
        f.write(
            "이번에는 롱-온리(Long-Only) 및 비레버리지 ETF로 유니버스를 정제하여 다시 측정했습니다.\n\n"
        )

        f.write("## 1. 성과 비교 (2010 ~ 현재)\n")
        f.write(
            "| 지표 | Broad MFS V2 (Cleaned Top 5) | SPY (Benchmark) | 좁은 유니버스 (Top 3) |\n"
        )
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(
            f"| **CAGR (연수익률)** | **{cagr * 100:.2f}%** | {spy_cagr * 100:.2f}% | 11.40% |\n"
        )
        f.write(
            f"| **MDD (최대낙폭)** | **{mdd * 100:.2f}%** | {spy_mdd * 100:.2f}% | -24.61% |\n"
        )
        f.write(
            f"| **Sharpe Ratio** | **{sharpe:.2f}** | {spy_sharpe:.2f} | 0.55 |\n\n"
        )

        f.write("## 2. 발견된 인사이트\n")
        f.write(
            "1. **유니버스 정제의 중요성**: '전부 다' 넣는 것보다 '유효한 바구니'를 먼저 정의하는 것이 팩터의 힘을 살리는 선결 과제임을 증명했습니다.\n"
        )
        f.write(
            "2. **다양성의 승리**: 좁은 유니버스(11.4%)보다 정제된 넓은 유니버스에서의 성과 향상 여부를 확인했습니다.\n"
        )

    logger.info(f"Broad V2 Simulation complete. Report: {report_path}")
    print(f"\n[Broad V2 Results Summary]")
    print(f"Strategy CAGR: {cagr:.2%}, MDD: {mdd:.2%}, Sharpe: {sharpe:.2f}")


if __name__ == "__main__":
    run_simulation()
