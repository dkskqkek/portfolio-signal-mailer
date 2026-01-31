# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataLoader")


class DataLoader:
    """
    High-performance Data Loader for Strategy Mining.
    Loads thousands of CSV files in parallel and aligns them to a common index.
    """

    def __init__(self, data_dir: str = "d:/gg/data/historical"):
        self.data_dir = data_dir
        self.tickers: List[str] = []
        self.data: Dict[str, pd.DataFrame] = {}  # Raw Data

    def load_all(self, limit: Optional[int] = None) -> None:
        """
        Loads all CSV files from the data directory.
        Args:
            limit: Optional limit for testing (e.g., load only first 100 tickers)
        """
        files = [
            f
            for f in os.listdir(self.data_dir)
            if f.endswith(".csv") and not f.startswith("_")
        ]
        if limit:
            files = files[:limit]

        logger.info(f"files to load: {len(files)}")

        results = {}
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            future_to_file = {executor.submit(self._read_file, f): f for f in files}

            for future in tqdm(
                as_completed(future_to_file), total=len(files), desc="Loading Data"
            ):
                filename = future_to_file[future]
                try:
                    ticker, df = future.result()
                    if df is not None:
                        results[ticker] = df
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")

        self.data = results
        self.tickers = list(results.keys())
        logger.info(f"Successfully loaded {len(self.data)} tickers.")

    def _read_file(self, filename: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        Helper to read a single CSV file.
        """
        try:
            path = os.path.join(self.data_dir, filename)
            ticker = filename.replace(".csv", "")

            # Read CSV
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

            # Basic validation
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if not all(col in df.columns for col in required_cols):
                return ticker, None

            # Optimization: Cast to float32 to save memory
            df = df[required_cols].astype("float32")

            return ticker, df
        except Exception as e:
            return filename, None

    def get_aligned_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns aligned DataFrames for O, H, L, C, V.
        Columns = Tickers, Index = Date (Union of all dates)
        """
        logger.info("Aligning data...")

        # Determine global date range
        # Identifying the most common date index/union might be slow.
        # Let's use concatenation.

        closes = {}
        opens = {}
        highs = {}
        lows = {}
        volumes = {}

        for ticker, df in tqdm(self.data.items(), desc="Structuring"):
            closes[ticker] = df["Close"]
            opens[ticker] = df["Open"]
            highs[ticker] = df["High"]
            lows[ticker] = df["Low"]
            volumes[ticker] = df["Volume"]

        logger.info("Concatenating DataFrames (This may take a moment)...")
        df_close = pd.concat(closes, axis=1).sort_index()
        df_open = pd.concat(opens, axis=1).sort_index()
        df_high = pd.concat(highs, axis=1).sort_index()
        df_low = pd.concat(lows, axis=1).sort_index()
        df_volume = pd.concat(volumes, axis=1).sort_index()

        logger.info("Data alignment complete.")
        return df_open, df_high, df_low, df_close, df_volume


if __name__ == "__main__":
    # Internal Test
    loader = DataLoader()
    loader.load_all(limit=100)  # Test with 100 files
    o, h, l, c, v = loader.get_aligned_data()
    print(f"Shape: {c.shape}")
    print(f"Sample Close:\n{c.iloc[-5:, :5]}")
