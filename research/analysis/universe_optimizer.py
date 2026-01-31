# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("UniverseOptimizer")

DATA_DIR = r"d:\gg\data\historical"
START_DATE = "2010-01-01"

# Cleaned ETF Pool (Extracted from Ultra V2)
CLEAN_POOL = [
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


def analyze_asset(t):
    path = os.path.join(DATA_DIR, f"{t}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
    df = df[df.index >= START_DATE]
    if len(df) < 500:
        return None

    ret = df["Close"].pct_change()
    sharpe = (ret.mean() * 252 - 0.04) / (ret.std() * np.sqrt(252))
    return {"ticker": t, "sharpe": sharpe, "vol": ret.std() * np.sqrt(252)}


def main():
    logger.info("Scanning all clean ETFs for individual strength...")
    results = []
    for t in CLEAN_POOL:
        res = analyze_asset(t)
        if res:
            results.append(res)

    rank_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    # Selective 9 performance
    selective_9 = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLK", "XLE", "XLV", "XLF"]
    logger.info(
        f"Selective 9 Average Sharpe: {rank_df[rank_df['ticker'].isin(selective_9)]['sharpe'].mean():.3f}"
    )

    # Candidates for 'Superior 10'
    # 1. High Momentum/Alpha: SMH, VGT, ARK계열
    # 2. Defensive/Factor: SCHD, USMV
    # 3. Macro: TLT, GLD, UUP

    top_candidates = rank_df.head(20)["ticker"].tolist()
    logger.info(f"Top 20 Individual Performers: {top_candidates}")

    # Manual Sweet Spot Selection (Combining strength + diversification)
    sweet_spot = [
        "QQQ",
        "XLK",
        "SMH",  # Growth/Tech
        "SPY",
        "SCHD",  # Core/Dividend
        "GLD",
        "SLV",  # Hard Assets
        "TLT",
        "AGG",  # Bonds
        "UUP",  # Currency (Hedge)
    ]

    print("\n[Universe Candidate Comparison]")
    print(f"Selective 9: {selective_9}")
    print(f"Sweet Spot (Top 10): {sweet_spot}")

    # Save these lists for simulation
    with open("d:/gg/research/optimized_universe.txt", "w") as f:
        f.write(",".join(sweet_spot))


if __name__ == "__main__":
    main()
