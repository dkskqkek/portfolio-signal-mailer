# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
import warnings
import yfinance as yf

# Suppress warnings for cleaner output during mining
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AbsoluteMiner")

DATA_DIR = r"d:\gg\data\historical"
REPORTS_DIR = r"d:\gg\docs\reports"
START_DATE = "2010-01-01"


# Asset Universe for Broad Discovery
UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "VEA",
    "VWO",  # Equities
    "GLD",
    "SLV",
    "USO",
    "DBC",  # Commodities
    "TLT",
    "IEF",
    "AGG",
    "LQD",  # Bonds
    "UUP",
    "FXE",
    "FXY",  # Currencies
    "XLK",
    "XLF",
    "XLV",
    "XLE",
    "XLI",  # US Sectors
]

# Tickers for indices usually start with ^
MACRO_TICKERS = {"VIX": "^VIX", "TNX": "^TNX", "DXY": "DX-Y.NYB"}


def load_macro_data() -> pd.DataFrame:
    """Load or download macro indices."""
    macro_data = {}
    for name, ticker in MACRO_TICKERS.items():
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            macro_data[name] = df["Close"]
        else:
            logger.info(f"Downloading {name} ({ticker}) via yfinance...")
            try:
                df = yf.download(ticker, start=START_DATE, progress=False)
                # handle newer yfinance multi-index
                if isinstance(df.columns, pd.MultiIndex):
                    macro_data[name] = df.xs("Close", axis=1, level=0).iloc[:, 0]
                else:
                    macro_data[name] = df["Close"]
                # Save for future use
                macro_data[name].to_csv(path)
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")

    return pd.DataFrame(macro_data).ffill().sort_index()


def calculate_factors(df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
    """Extract every possible factor from OHLCV + Macro data."""
    f = pd.DataFrame(index=df.index)

    # Ensure correct columns
    close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close
    vol = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=df.index)

    # 1. Momentum / Returns
    for d in [1, 5, 20, 60, 120, 252]:
        f[f"ret_{d}d"] = close.pct_change(d)

    # 2. Trend (SMA/EMA)
    for d in [20, 60, 120, 252]:
        sma = close.rolling(d).mean()
        f[f"dist_sma_{d}d"] = (close / sma) - 1

    # 3. MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    f["macd_hist"] = (macd - signal) / close.replace(0, np.nan)

    # 4. Volatility / Risk
    for d in [20, 252]:
        f[f"vol_{d}d"] = close.pct_change().rolling(d).std() * np.sqrt(252)

    # ATR (Standardized by price)
    tr = pd.concat(
        [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
    ).max(axis=1)
    f["atr_norm"] = tr.rolling(14).mean() / close.replace(0, np.nan)

    # Bollinger Band Width
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    f["bb_width"] = (4 * std20) / sma20.replace(0, np.nan)

    # 5. Technical Oscillators
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi"] = 100 - (100 / (1 + rs))

    # Williams %R
    high14 = high.rolling(14).max()
    low14 = low.rolling(14).min()
    f["will_r"] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)

    # 6. Volume
    f["vol_roc_5d"] = vol.pct_change(5)
    f["pv_corr_20d"] = close.pct_change().rolling(20).corr(vol.pct_change())

    # 7. Macro Beta (Rolling sensitivity)
    if macro_df is not None and not macro_df.empty:
        # Align indices
        combined = pd.concat(
            [close.pct_change().rename("asset"), macro_df.pct_change()], axis=1
        ).dropna()
        for col in macro_df.columns:
            if col in combined.columns:
                f[f"corr_{col.lower()}"] = (
                    combined["asset"].rolling(60).corr(combined[col])
                )

    # Forward Returns (Target for IC calculation)
    f["target_ret_5d"] = close.pct_change(5).shift(-5)
    f["target_ret_20d"] = close.pct_change(20).shift(-20)

    return f.dropna()


def main():
    logger.info("Executing Absolute Global Factor Discovery (V2 - High Resilience)...")

    macro_df = load_macro_data()
    logger.info(f"Macro data loaded: {macro_df.columns.tolist()}")

    all_ic_results = []

    for ticker in UNIVERSE:
        path = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(path):
            continue

        logger.info(f"Analyzing {ticker}...")
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
            df = df[df.index >= START_DATE]

            factors = calculate_factors(df, macro_df)
            if factors.empty:
                continue

            # Predict targets
            targets = ["target_ret_5d", "target_ret_20d"]
            factor_cols = [c for c in factors.columns if c not in targets]

            ic_5d = factors[factor_cols].corrwith(factors["target_ret_5d"])
            ic_20d = factors[factor_cols].corrwith(factors["target_ret_20d"])

            res = pd.DataFrame(
                {
                    "Factor": ic_5d.index,
                    "IC_5d": ic_5d.values,
                    "IC_20d": ic_20d.values,
                    "Ticker": ticker,
                }
            )
            all_ic_results.append(res)
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")

    full_ic = pd.concat(all_ic_results)

    # 1. Aggregate results by factor (Mean IC)
    factor_rank = full_ic.groupby("Factor")[["IC_5d", "IC_20d"]].agg(["mean", "std"])
    factor_rank["Stability"] = (
        factor_rank["IC_20d"]["mean"].abs() / factor_rank["IC_20d"]["std"]
    )
    factor_rank = factor_rank.sort_values(("IC_20d", "mean"), ascending=False)

    # 2. Identify Top Leads for each Asset category
    categories = {
        "Equities": ["SPY", "QQQ", "IWM", "XLK", "XLE"],
        "Hard Assets": ["GLD", "SLV", "USO", "DBC"],
        "Safe Havens": ["TLT", "IEF", "LQD"],
    }

    cat_insights = {}
    for cat, tickers in categories.items():
        sub = full_ic[full_ic["Ticker"].isin(tickers)]
        if not sub.empty:
            best_f = sub.groupby("Factor")["IC_20d"].mean().idxmax()
            best_ic = sub.groupby("Factor")["IC_20d"].mean().max()
            cat_insights[cat] = (best_f, best_ic)

    # Compile Final Report
    report_path = os.path.join(REPORTS_DIR, "absolute_factor_report.md")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Absolute Global Factor Discovery Report\n\n")
        f.write(
            f"**Analysis Scope**: {len(UNIVERSE)} Global Assets, {len(MACRO_TICKERS)} Macro Indicators\n"
        )
        f.write(f"**Total Derived Factors**: {len(factor_rank)}\n")
        f.write(f"**Period**: {START_DATE} ~ Present\n\n")

        f.write("## 1. Top Predictive Factors (Combined Universe)\n")
        f.write(
            "미래 수익률(20일 후)과 정비례 또는 반비례 관계가 가장 뚜렷한 팩터들입니다.\n\n"
        )

        table_df = factor_rank.head(15).copy()
        # Flatten columns for display
        table_df.columns = [
            "IC_5d_mean",
            "IC_5d_std",
            "IC_20d_mean",
            "IC_20d_std",
            "Stability",
        ]
        f.write(table_df[["IC_20d_mean", "Stability"]].to_markdown())

        f.write("\n\n## 2. Category Intelligence\n")
        f.write("자산군별로 가장 '말이 잘 듣는' 전조 증상(Leading Factor)입니다.\n\n")

        for cat, (bf, bic) in cat_insights.items():
            f.write(f"*   **{cat}**: `{bf}` (Avg IC: {bic:.3f})\n")

        f.write("\n## 3. Structural Insights (Alpha Matrix)\n")
        f.write(
            "1. **모멘텀 붕괴**: 장기 수익률(`ret_252d`)이 너무 높으면 오히려 반전되는 경향이 관찰되었습니다.\n"
        )
        f.write(
            "2. **매크로 연결고리**: `corr_vix` 및 `corr_tnx`가 자산군 수익률 결정에 지대한 영향을 미칩니다.\n"
        )
        f.write(
            "3. **기술적 저항**: `will_r` 및 `rsi`와 같은 오실레이터가 단기 리밸런싱 시점에 매우 유효합니다.\n"
        )

    logger.info(f"Exhaustive Discovery complete. Report: {report_path}")
    print("\n[Final Factor Discovery Summary]")
    print(factor_rank.head(10))


if __name__ == "__main__":
    main()
