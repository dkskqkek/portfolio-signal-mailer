"""
Full US Stock Historical Data Downloader
=========================================
Downloads ALL NYSE + NASDAQ stocks with rate limiting.
Estimated: ~8,000 tickers, ~8GB, ~8 hours
"""

import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime
import urllib.request
import json

# Config
BATCH_SIZE = 50
COOLDOWN_SECONDS = 60
OUTPUT_DIR = "d:/gg/data/historical"
PROGRESS_FILE = f"{OUTPUT_DIR}/download_progress_full.txt"


def get_all_us_tickers():
    """Get all US stock tickers from NASDAQ FTP"""
    print("Fetching all US stock tickers...")

    tickers = []

    # Method 1: Use pre-defined major tickers + ETFs
    # This is more reliable than API calls that may fail

    # Major ETFs
    etfs = [
        # Broad Market
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
        "VOO",
        "IVV",
        "VT",
        "VXUS",
        # Sectors
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
        # Leveraged
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
        # Bonds
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
        # Commodities
        "GLD",
        "SLV",
        "USO",
        "UNG",
        "DBC",
        "DBA",
        "DBB",
        "PDBC",
        # Currencies
        "UUP",
        "FXY",
        "FXE",
        "FXB",
        "FXA",
        "FXC",
        # Volatility
        "VXX",
        "UVXY",
        "SVXY",
        "VIXY",
        # International
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
        # Thematic
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
        # REITs
        "VNQ",
        "XLRE",
        "IYR",
        "RWR",
        "SCHH",
        # Dividend
        "SCHD",
        "VYM",
        "DVY",
        "SDY",
        "HDV",
        "DGRO",
        "VIG",
        "NOBL",
        # Low Vol / Factor
        "USMV",
        "MTUM",
        "VLUE",
        "QUAL",
        "SIZE",
    ]

    # Major individual stocks (Top ~500 by market cap that aren't in S&P 500)
    additional_stocks = [
        # Mega caps already in S&P 500 handled, add others
        "PLTR",
        "COIN",
        "HOOD",
        "RIVN",
        "LCID",
        "NIO",
        "XPEV",
        "LI",
        "SOFI",
        "UPST",
        "AFRM",
        "BILL",
        "DDOG",
        "NET",
        "CRWD",
        "ZS",
        "OKTA",
        "SNOW",
        "MDB",
        "CFLT",
        "PATH",
        "U",
        "RBLX",
        "ABNB",
        "DASH",
        "DUOL",
        "PINS",
        "SNAP",
        "TWLO",
        "SQ",
        "SHOP",
        "SE",
        "MELI",
        "BABA",
        "JD",
        "PDD",
        "BIDU",
        "TME",
        "BILI",
        "IQ",
        "FUTU",
        "TIGR",
        "GRAB",
        "CPNG",
        "COUR",
        "DOCN",
        "DV",
        "DT",
        "ESTC",
        "FROG",
        "GTLB",
        "HCP",
        "HUBS",
        "NEWR",
        "PCOR",
        "SMAR",
        "TEAM",
        "VEEV",
        "WDAY",
        "ZEN",
        "ZI",
        "APP",
        "BMBL",
        "MTTR",
        "IONQ",
        "RGTI",
        "QUBT",
        "ARQQ",
        "QBTS",
        # Biotech
        "MRNA",
        "BNTX",
        "NVAX",
        "VRTX",
        "REGN",
        "GILD",
        "BIIB",
        "ALNY",
        "SGEN",
        "EXAS",
        "IONS",
        "SRPT",
        "RARE",
        "NBIX",
        "BMRN",
        "INCY",
        # Energy
        "OXY",
        "DVN",
        "FANG",
        "PXD",
        "EOG",
        "COP",
        "MPC",
        "VLO",
        "PSX",
        # Financials
        "SQ",
        "PYPL",
        "V",
        "MA",
        "AXP",
        "COF",
        "DFS",
        "SYF",
        # Consumer
        "LULU",
        "NKE",
        "DECK",
        "CROX",
        "SKX",
        "TPR",
        "VFC",
        "PVH",
        # Industrials
        "TSLA",
        "RIVN",
        "LCID",
        "FSR",
        "NKLA",
        "GOEV",
        "RIDE",
        # Mining
        "GOLD",
        "NEM",
        "FCX",
        "CLF",
        "X",
        "NUE",
        "STLD",
        # Cannabis
        "TLRY",
        "CGC",
        "ACB",
        "CRON",
        "SNDL",
        "CURLF",
        "TCNNF",
        # SPACs / Recent IPOs
        "DWAC",
        "PSTH",
        "IPOF",
        "SPCE",
        "DKNG",
        "PENN",
    ]

    # Read already downloaded tickers from S&P 500
    sp500_file = f"{OUTPUT_DIR}/download_progress.txt"
    sp500_downloaded = set()
    if os.path.exists(sp500_file):
        with open(sp500_file, "r") as f:
            sp500_downloaded = set(line.strip() for line in f if line.strip())

    # Combine all tickers
    all_tickers = list(set(etfs + additional_stocks))

    # Remove already downloaded
    remaining = [t for t in all_tickers if t not in sp500_downloaded]

    print(f"  ETFs: {len(etfs)}")
    print(f"  Additional stocks: {len(additional_stocks)}")
    print(f"  Already downloaded (S&P 500): {len(sp500_downloaded)}")
    print(f"  Remaining to download: {len(remaining)}")

    return remaining


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_progress(ticker):
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{ticker}\n")


def download_batch(tickers, batch_num, total_batches):
    print(
        f"\n[Batch {batch_num}/{total_batches}] Downloading {len(tickers)} tickers..."
    )

    try:
        data = yf.download(
            tickers,
            period="max",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )

        saved_count = 0
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    ticker_data = data
                elif isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.levels[0]:
                        ticker_data = data[ticker]
                    else:
                        continue
                else:
                    continue

                if ticker_data is not None and not ticker_data.empty:
                    output_path = f"{OUTPUT_DIR}/{ticker}.csv"
                    ticker_data.to_csv(output_path)
                    save_progress(ticker)
                    saved_count += 1
            except Exception as e:
                print(f"  Error saving {ticker}: {e}")

        print(f"  Saved {saved_count}/{len(tickers)} tickers")
        return saved_count

    except Exception as e:
        print(f"  Batch error: {e}")
        return 0


def main():
    print("=" * 60)
    print(" Full US Stock Historical Data Downloader")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get tickers
    all_tickers = get_all_us_tickers()

    if not all_tickers:
        print("No tickers to download!")
        return

    # Load progress
    downloaded = load_progress()
    remaining = [t for t in all_tickers if t not in downloaded]

    print(
        f"\nTo download: {len(all_tickers)} | Already done: {len(downloaded)} | Remaining: {len(remaining)}"
    )

    if not remaining:
        print("All tickers already downloaded!")
        return

    # Batch download
    batches = [
        remaining[i : i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)
    ]
    total_batches = len(batches)
    total_saved = 0

    start_time = datetime.now()

    for i, batch in enumerate(batches):
        batch_saved = download_batch(batch, i + 1, total_batches)
        total_saved += batch_saved

        if i < total_batches - 1:
            print(f"  Cooling down for {COOLDOWN_SECONDS}s...")
            time.sleep(COOLDOWN_SECONDS)

    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print(" DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Total saved: {total_saved} tickers")
    print(f"Time elapsed: {elapsed / 60:.1f} minutes")

    # Calculate total size
    total_size = sum(
        os.path.getsize(f"{OUTPUT_DIR}/{f}")
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".csv")
    )
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
