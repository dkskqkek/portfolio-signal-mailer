"""
한국 주식 당일 5분봉 수집 스크립트
실행 시각: 매일 15:40 (장마감 후)

Usage:
    python scripts/collect_intraday_kr.py
"""

import logging
import yaml
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_mailer.kis_api_wrapper import KISAPIWrapper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def collect_kr_intraday():
    """한국 주식 5분봉 데이터 수집"""
    # 1. Load Config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "signal_mailer", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # 2. Load Universe
    universe_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "kr_universe.json"
    )
    with open(universe_path, "r", encoding="utf-8") as f:
        universe = json.load(f)[:200]  # Top 200

    # 3. Collect Data
    all_bars = []
    logger.info(f"Collecting 5min bars for {len(universe)} tickers...")

    for i, item in enumerate(universe):
        ticker = item["ticker"]
        name = item.get("name", ticker)

        bars = kis.get_intraday_bars(ticker, period="5")
        if bars:
            for bar in bars:
                try:
                    # Parse KIS response
                    time_str = bar.get("stck_cntg_hour", "0900")
                    dt = datetime.strptime(
                        f"{datetime.now().strftime('%Y-%m-%d')} {time_str}",
                        "%Y-%m-%d %H%M",
                    )

                    all_bars.append(
                        {
                            "ticker": ticker,
                            "name": name,
                            "datetime": dt,
                            "open": float(bar.get("stck_oprc", 0)),
                            "high": float(bar.get("stck_hgpr", 0)),
                            "low": float(bar.get("stck_lwpr", 0)),
                            "close": float(bar.get("stck_prpr", 0)),
                            "volume": int(bar.get("cntg_vol", 0)),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error parsing bar for {ticker}: {e}")
                    continue

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(universe)}")

    # 4. Save to Parquet
    if not all_bars:
        logger.warning("No data collected")
        return

    df = pd.DataFrame(all_bars)
    output_dir = Path("data/intraday/kr")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{datetime.now().strftime('%Y-%m-%d')}.parquet"
    filepath = output_dir / filename

    df.to_parquet(filepath, compression="snappy", index=False)
    logger.info(f"✅ Saved {len(df)} records to {filepath}")
    logger.info(f"   Tickers: {df['ticker'].nunique()}")
    logger.info(f"   File size: {filepath.stat().st_size / 1024:.1f} KB")

    # 5. Send Discord Notification
    try:
        from signal_mailer.notification.discord_webhook import send_discord_msg

        send_discord_msg(
            config,
            "📊 [Data] KR Intraday Collection",
            f"수집 완료: {df['ticker'].nunique()} 종목, {len(df)} bars\n파일: `{filename}`",
            color=0x00BFFF,
        )
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")


if __name__ == "__main__":
    collect_kr_intraday()
