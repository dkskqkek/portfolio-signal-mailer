"""
미국 주식 당일 1분봉 수집 스크립트 (KIS API)
실행 시각: 매일 06:30 KST (미국 장마감 후)

Usage:
    python scripts/collect_intraday_us.py
"""

import logging
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_mailer.kis_api_wrapper import KISAPIWrapper

# 로그 디렉토리 생성
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정 (Console + File)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "intraday_us.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# MAMA Lite 포트폴리오 종목
US_TICKERS = [
    # Core Holdings
    {"ticker": "QQQ", "exchange": "NAS"},
    {"ticker": "SPY", "exchange": "NYS"},
    # GNN - Big Tech
    {"ticker": "AAPL", "exchange": "NAS"},
    {"ticker": "MSFT", "exchange": "NAS"},
    {"ticker": "GOOGL", "exchange": "NAS"},
    {"ticker": "AMZN", "exchange": "NAS"},
    {"ticker": "META", "exchange": "NAS"},
    {"ticker": "NVDA", "exchange": "NAS"},
    {"ticker": "TSLA", "exchange": "NAS"},
    {"ticker": "NFLX", "exchange": "NAS"},
    {"ticker": "AVGO", "exchange": "NAS"},
    # Defensive Assets
    {"ticker": "BIL", "exchange": "NYS"},
    {"ticker": "TLT", "exchange": "NAS"},
    {"ticker": "GLD", "exchange": "NYS"},
    {"ticker": "UUP", "exchange": "NYS"},
    {"ticker": "BTAL", "exchange": "NYS"},
    {"ticker": "PFIX", "exchange": "NYS"},
    {"ticker": "DBMF", "exchange": "NYS"},
    {"ticker": "AGG", "exchange": "NYS"},
    {"ticker": "SHY", "exchange": "NAS"},
    # Value/Dividend
    {"ticker": "SCHD", "exchange": "NAS"},
    {"ticker": "VTI", "exchange": "NYS"},
]


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """1분봉 → 5분봉 변환"""
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Group by ticker and resample
    resampled_list = []
    for ticker, group in df.groupby("ticker"):
        group_resampled = (
            group.set_index("datetime")
            .resample("5min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        group_resampled["ticker"] = ticker
        group_resampled.reset_index(inplace=True)
        resampled_list.append(group_resampled)

    if resampled_list:
        return pd.concat(resampled_list, ignore_index=True)
    return pd.DataFrame()


def collect_us_intraday():
    """미국 주식 1분봉 수집 → 5분봉 변환"""
    # 1. Load Config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "signal_mailer", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # 2. Collect Data
    all_bars = []
    logger.info(f"Collecting 1min bars for {len(US_TICKERS)} US tickers...")

    for item in US_TICKERS:
        ticker = item["ticker"]
        exchange = item["exchange"]

        bars = kis.get_us_intraday_bars(ticker, exchange=exchange, period="1")
        if bars:
            for bar in bars:
                try:
                    # Parse time: "093000" → datetime
                    time_str = bar.get("xhms", "093000")
                    dt = datetime.strptime(
                        f"{datetime.now().strftime('%Y-%m-%d')} {time_str}",
                        "%Y-%m-%d %H%M%S",
                    )

                    all_bars.append(
                        {
                            "ticker": ticker,
                            "datetime": dt,
                            "open": float(bar.get("open", 0)),
                            "high": float(bar.get("high", 0)),
                            "low": float(bar.get("low", 0)),
                            "close": float(bar.get("last", 0)),
                            "volume": int(bar.get("evol", 0)),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error parsing bar for {ticker}: {e}")
                    continue

            logger.info(f"✓ {ticker}: {len(bars)} bars")
        else:
            logger.warning(f"✗ {ticker}: No data")

    # 3. Convert to DataFrame and Resample to 5min
    if not all_bars:
        logger.warning("No data collected")
        return

    df = pd.DataFrame(all_bars)
    logger.info(f"Collected {len(df)} 1min bars")

    df_5min = resample_to_5min(df)
    logger.info(f"Resampled to {len(df_5min)} 5min bars")

    # 4. Save to Parquet
    output_dir = Path("data/intraday/us")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{datetime.now().strftime('%Y-%m-%d')}.parquet"
    filepath = output_dir / filename

    df_5min.to_parquet(filepath, compression="snappy", index=False)
    logger.info(f"✅ Saved {len(df_5min)} records (5min bars) to {filepath}")
    logger.info(f"   Tickers: {df_5min['ticker'].nunique()}")
    logger.info(f"   File size: {filepath.stat().st_size / 1024:.1f} KB")

    # 5. Discord Notification
    try:
        from signal_mailer.notification.discord_webhook import send_discord_msg

        send_discord_msg(
            config,
            "📊 [Data] US Intraday Collection",
            f"수집 완료: {df_5min['ticker'].nunique()} 종목\n1분봉: {len(df)} → 5분봉: {len(df_5min)}\n파일: `{filename}`",
            color=0x00BFFF,
        )
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")


if __name__ == "__main__":
    collect_us_intraday()
