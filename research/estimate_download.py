import time
import yfinance as yf
import pandas as pd
import requests


def estimate_kr_market():
    # 1. Estimate Ticker Count
    # KOSPI ~950, KOSDAQ ~1700
    # We don't have a live list, so we use conservative estimates
    kospi_count = 950
    kosdaq_count = 1750
    total_tickers = kospi_count + kosdaq_count

    print(f"--- Market Scope ---")
    print(f"KOSPI: ~{kospi_count}")
    print(f"KOSDAQ: ~{kosdaq_count}")
    print(f"Total Tickers: {total_tickers}")

    # 2. Measure Bandwidth / API Latency
    print(f"\n--- Speed Test (Downloading 20 random tickers) ---")
    test_tickers = [
        "005930.KS",
        "000660.KS",
        "035420.KS",
        "207940.KS",
        "005380.KS",
        "005935.KS",
        "068270.KS",
        "000270.KS",
        "105560.KS",
        "055550.KS",  # KOSPI 10
        "091990.KS",
        "086520.KQ",
        "247540.KQ",
        "066970.KQ",
        "028300.KQ",  # Mixed
        "293490.KQ",
        "112040.KQ",
        "022100.KQ",
        "035900.KQ",
        "036930.KQ",
    ]

    start_time = time.time()
    # Fetch 10 years of data
    data = yf.download(test_tickers, period="10y", progress=False, group_by="ticker")
    end_time = time.time()

    duration = end_time - start_time
    # Approximate size in memory (CSV size is roughly similar)
    # Pandas memory usage is decent proxy for disk CSV size
    size_bytes = 0
    if isinstance(data.columns, pd.MultiIndex):
        # Check size of dataframe
        size_bytes = data.memory_usage(deep=True).sum()

    size_mb = size_bytes / (1024 * 1024)
    speed_mbps = (size_mb * 8) / duration
    tickers_per_sec = len(test_tickers) / duration

    print(f"Time Taken: {duration:.2f} seconds")
    print(f"Downloaded Size (Approx): {size_mb:.2f} MB")
    print(f"Effective Speed: {tickers_per_sec:.2f} tickers/sec")

    # 3. Projections
    # Size Calculation:
    # 20 tickers = X MB -> 2700 tickers = ?
    total_size_est_mb = (size_mb / len(test_tickers)) * total_tickers
    total_time_est_sec = total_tickers / tickers_per_sec

    print(f"\n--- Estimates for Full Download ({total_tickers} tickers, 10y Daily) ---")
    print(
        f"Total Storage Required: ~{total_size_est_mb:.1f} MB ({total_size_est_mb / 1024:.2f} GB)"
    )
    print(f"Estimated Time: {total_time_est_sec / 60:.1f} minutes")
    print(f"Note: Bottleneck is usually API Latency, not bandwidth.")


if __name__ == "__main__":
    estimate_kr_market()
