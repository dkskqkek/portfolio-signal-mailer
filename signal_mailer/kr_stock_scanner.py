# -*- coding: utf-8 -*-
import logging
import json
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from signal_mailer.kis_api_wrapper import KISAPIWrapper

logger = logging.getLogger(__name__)


class KRStockScanner:
    """
    Live Scanner for KOSPI/KOSDAQ stocks using Hybrid Alpha logic.
    Logic: (Close > SMA_5) AND (ROC_1 > 0)
    DEPENDENCY: KIS API Only (pykrx dependency removed for reliability)
    """

    def __init__(self, kis_wrapper: KISAPIWrapper):
        self.kis = kis_wrapper
        self.universe_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "kr_universe.json",
        )
        self.universe: List[Dict[str, str]] = []
        self._load_universe()

    def _load_universe(self) -> None:
        """Load the pre-generated stock universe."""
        if os.path.exists(self.universe_path):
            try:
                with open(self.universe_path, "r", encoding="utf-8") as f:
                    self.universe = json.load(f)
                logger.info(f"Loaded {len(self.universe)} tickers from universe cache.")
            except Exception as e:
                logger.error(f"Failed to load universe: {e}")
        else:
            logger.warning(
                f"Universe file not found at {self.universe_path}. Scan might be limited."
            )

    def get_universe(self, market: str = "ALL") -> List[str]:
        """Fetch all active tickers for KOSPI and KOSDAQ."""
        # This method is now deprecated or needs re-evaluation given the new universe loading.
        # For now, returning tickers from the loaded universe.
        return [item["ticker"] for item in self.universe]

    def scan_full_market(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Scan the market using the cached universe and KIS API.
        """
        # 1. Select Tickers from Universe
        active_tickers = self.universe[:limit]
        if not active_tickers:
            logger.error("No active tickers to scan.")
            return []

        logger.info(
            f"Scanning {len(active_tickers)} tickers using Hybrid Alpha (KIS Only)..."
        )

        # 2. Parallel Logic Check
        candidates = []
        with ThreadPoolExecutor(max_workers=15) as executor:
            future_to_info = {
                executor.submit(self._check_logic, item): item
                for item in active_tickers
            }
            for future in as_completed(future_to_info):
                res = future.result()
                if res:
                    candidates.append(res)

        # 3. Sort by ROC_1 descending
        candidates.sort(key=lambda x: x["roc_1"], reverse=True)
        return candidates

    def _check_logic(self, item: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Check logic for a ticker info dictionary."""
        ticker = item["ticker"]
        name = item["name"]
        try:
            # 1. Get recent OHLCV
            res = self.kis.get_ohlcv_recent(ticker)
            if not res or "output" not in res:
                return None

            output = res["output"]
            if not isinstance(output, list) or len(output) < 5:
                # Handle cases where KIS returns different structure or fewer bars
                return None

            # output is usually sorted by date desc
            today = output[0]
            prev = output[1]

            curr_price = float(today["stck_clpr"])
            prev_price = float(prev["stck_clpr"])

            # SMA_5 calculation
            last_5 = [float(x["stck_clpr"]) for x in output[:5]]
            sma_5 = sum(last_5) / 5

            # Logic: Close > SMA_5 AND Close > PrevClose (ROC_1 > 0)
            if curr_price > sma_5 and curr_price > prev_price:
                return {
                    "ticker": ticker,
                    "name": name,
                    "price": curr_price,
                    "sma_5": sma_5,
                    "roc_1": (curr_price - prev_price)
                    / (prev_price if prev_price != 0 else 1),
                    "dist_sma": (curr_price - sma_5) / (sma_5 if sma_5 != 0 else 1),
                }
        except Exception as e:
            logger.debug(f"Error checking {ticker}: {e}")

        return None
