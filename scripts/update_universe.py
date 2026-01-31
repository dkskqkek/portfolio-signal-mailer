"""
Dynamic Universe Updater (Top 20 Hybrid)
- Source: Wikipedia S&P 500 List
- Logic:
    1. Growth Core (Tech, Comm, ConsDisc): Top 12 by Market Cap
    2. Sector Leaders (Others): Top 1 from each of 8 remaining sectors
    3. Update config.yaml with new 20 tickers
"""

import pandas as pd
import yfinance as yf
import yaml
import os
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("UniverseUpdater")

# Path Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "signal_mailer", "config.yaml")

# Sector Definitions
GROWTH_SECTORS = [
    "Information Technology",
    "Communication Services",
    "Consumer Discretionary",
]

BALANCE_SECTORS = [
    "Health Care",
    "Financials",
    "Energy",
    "Consumer Staples",
    "Industrials",
    "Utilities",
    "Materials",
    "Real Estate",
]


def fetch_sp500_list():
    """Fetch S&P 500 components from Wikipedia with User-Agent."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    logger.info(f"Fetching S&P 500 list from {url}...")
    try:
        import requests

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        dfs = pd.read_html(response.text)
        df = dfs[0]
        # Clean Tickers (e.g., BRK.B -> BRK-B)
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df[["Symbol", "GICS Sector", "GICS Sub-Industry"]]
    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 list: {e}")
        sys.exit(1)


def get_market_cap(ticker):
    """Fetch market cap for a single ticker."""
    try:
        info = yf.Ticker(ticker).info
        cap = info.get("marketCap", 0)
        return ticker, cap
    except Exception:
        return ticker, 0


def fetch_market_caps_parallel(tickers, max_workers=20):
    """Fetch market caps in parallel."""
    logger.info(
        f"Fetching market caps for {len(tickers)} tickers (threads={max_workers})..."
    )
    ticker_caps = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(get_market_cap, t): t for t in tickers}
        for i, future in enumerate(as_completed(future_to_ticker)):
            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i + 1}/{len(tickers)}")
            ticker, cap = future.result()
            ticker_caps[ticker] = cap
    return ticker_caps


def select_top_tickers():
    # 1. Get S&P 500 List
    df = fetch_sp500_list()

    # 2. Fetch Market Caps
    # 모든 종목을 다 조회하면 오래 걸리므로, 섹터별로 이미 어느 정도 순서를 알 수 있는 방법이 없음.
    # 하지만 시가총액 상위권을 뽑는 것이므로, 500개 다 조회하는 게 가장 확실함.
    # 500개 조회 배치는 ThreadPool로 하면 1-2분 내로 가능.
    caps = fetch_market_caps_parallel(df["Symbol"].tolist())
    df["MarketCap"] = df["Symbol"].map(caps)

    # Sort by Market Cap Descending
    df = df.sort_values("MarketCap", ascending=False)

    selected_tickers = []

    # 3. Growth Core Selection (Top 12)
    growth_df = df[df["GICS Sector"].isin(GROWTH_SECTORS)].head(12)
    growth_tickers = growth_df["Symbol"].tolist()
    logger.info(f"Adding Growth Core (12): {growth_tickers}")
    selected_tickers.extend(growth_tickers)

    # 4. Sector Leaders Selection (Top 1 from each Balance Sector)
    for sector in BALANCE_SECTORS:
        sector_df = df[df["GICS Sector"] == sector]
        if not sector_df.empty:
            leader = sector_df.iloc[0]["Symbol"]
            logger.info(f"Adding {sector} Leader: {leader}")
            selected_tickers.append(leader)

    # Remove duplicates if any (though logic separates lists)
    final_tickers = list(dict.fromkeys(selected_tickers))  # preserve order

    logger.info(f"Final Selection ({len(final_tickers)}): {final_tickers}")
    return final_tickers


def update_config(new_tickers):
    """Update config.yaml with new tickers."""
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config file not found at {CONFIG_PATH}")
        return

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config_lines = f.readlines()

        # We need to find the gnn_tickers block and replace it carefully to preserve comments/structure
        # OR use yaml library to dump.
        # Using yaml dump might break formatting/comments.
        # Manual replacement approach for safety of comments.

        # First, verifying load
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config_obj = yaml.safe_load(f)

        old_tickers = config_obj.get("strategy_info", {}).get("gnn_tickers", [])

        if set(old_tickers) == set(new_tickers):
            logger.info("Tickers unchanged. Skipping update.")
            return False

        # Backup config
        backup_path = CONFIG_PATH + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(backup_path, "w", encoding="utf-8") as f:
            f.writelines(config_lines)
        logger.info(f"Config backed up to {backup_path}")

        # Prepare new list string with indentation
        new_list_str = "    [\n"
        for t in new_tickers:
            new_list_str += f'      "{t}",\n'
        new_list_str += "    ]\n"

        # Rewrite file
        new_lines = []
        in_tickers_block = False

        for line in config_lines:
            strip_line = line.strip()
            if "gnn_tickers:" in strip_line:
                new_lines.append(line)  # Keep the key line
                # Determine if inline or block
                if "[" in strip_line and "]" in strip_line:
                    # Inline format: gnn_tickers: ["A", "B"] - replace immediately
                    new_lines[-1] = f"  gnn_tickers: {str(new_tickers)}\n"
                elif "[" in strip_line:
                    # Start of multi-line block
                    in_tickers_block = True
                    new_lines.append(new_list_str)  # Insert new block
                else:
                    # Likely block start on next line? Or maybe empty?
                    # Assuming standard formatting:
                    # gnn_tickers:
                    #   [
                    #     ...
                    #   ]
                    in_tickers_block = True
                    new_lines.append(new_list_str)
                continue

            if in_tickers_block:
                if "]" in strip_line:
                    in_tickers_block = False  # End of block
                # Skip lines inside the block
                continue

            new_lines.append(line)

        # Write back
        # The above logic is a bit fragile for replacing [ ... ].
        # Let's use a simpler approach: Read full text, use regex or just overwrite with YAML dump if format allows.
        # But user wants to preserve comments.
        # Let's try the safer manual replacement tailored to current file structure.

        # Current format in file:
        #   gnn_tickers:
        #     [
        #       "AAPL",
        #       ...
        #     ]

        final_output = []
        skip_mode = False

        for line in config_lines:
            if "gnn_tickers:" in line:
                final_output.append("  gnn_tickers:\n")
                final_output.append("    [\n")
                for t in new_tickers:
                    final_output.append(f'      "{t}",\n')
                final_output.append("    ]\n")
                skip_mode = True
                # Detect if the original was single line or multi-line start
                if "[" in line and "]" in line:
                    skip_mode = False  # Should have been replaced inline if logic supported, but here we force multi-line
                elif "[" in line:
                    pass  # skipping started
                else:
                    # 'gnn_tickers:\n' case
                    pass
            elif skip_mode:
                if "]" in line:
                    skip_mode = False
            else:
                final_output.append(line)

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.writelines(final_output)

        logger.info("Config updated successfully.")
        return True

    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return False


if __name__ == "__main__":
    logger.info("Starting Universe Update...")
    new_tickers = select_top_tickers()
    if new_tickers:
        updated = update_config(new_tickers)
        if updated:
            print("Universe Updated.")
        else:
            print("No changes made.")
    else:
        logger.error("No tickers selected.")
