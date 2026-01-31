import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataFetcher")


def fetch_simulation_data():
    # Tickers involved in MAMA Lite
    tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "NFLX",
        "AVGO",  # GNN
        "BIL",
        "TLT",
        "SPY",
        "QQQ",
        "^VIX",
        "^TNX",
    ]  # Macro/Defensive

    start_date = "2008-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(
        f"Downloading data from {start_date} to {end_date} for {len(tickers)} tickers..."
    )
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]

    # Clean data
    # BIL started in 2007-05-30, TLT in 2002. META started in 2012.
    # For simulation, we need a common period or handle missingness.
    # Let's use daily returns.
    returns = data.pct_change().dropna(how="all")

    # Save to CSV for the simulator
    output_path = "d:/gg/data/mama_lite_historical_returns.csv"
    returns.to_csv(output_path)
    logger.info(f"Saved historical returns to {output_path}")
    logger.info(f"Data shape: {returns.shape}")


if __name__ == "__main__":
    fetch_simulation_data()
