import os
import time
from pykrx import stock
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def setup_directory():
    base_dir = "d:/gg/data/historical"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir


def get_recent_business_day():
    candidates = []
    base = datetime.now()
    for i in range(5):
        d = base - timedelta(days=i)
        if d.weekday() < 5:
            candidates.append(d.strftime("%Y%m%d"))
    return candidates


def get_kr_tickers():
    print("Fetching KOSPI & KOSDAQ ticker list via PyKRX...")

    dates = get_recent_business_day()
    tickers = []

    for date in dates:
        try:
            print(f"Trying date: {date}")
            # KOSPI
            df_kospi = stock.get_market_cap(date, market="KOSPI")

            if df_kospi.empty:
                print("  -> Empty, trying next date.")
                continue

            print(f"  -> Found {len(df_kospi)} KOSPI entries.")

            # KOSDAQ
            df_kosdaq = stock.get_market_cap(date, market="KOSDAQ")
            print(f"  -> Found {len(df_kosdaq)} KOSDAQ entries.")

            # Parse - Skip Names for Speed/Reliability
            for code in df_kospi.index:
                tickers.append(
                    {"ticker": f"{code}.KS", "name": str(code), "market": "KOSPI"}
                )

            for code in df_kosdaq.index:
                tickers.append(
                    {"ticker": f"{code}.KQ", "name": str(code), "market": "KOSDAQ"}
                )

            break  # Success

        except Exception as e:
            print(f"  -> Error on {date}: {e}")
            continue

    print(f"Found {len(tickers)} tickers total.")
    return tickers


def download_data():
    save_dir = setup_directory()
    all_tickers = get_kr_tickers()

    if not all_tickers:
        print("❌ Failed to fetch ticker list.")
        return

    # Use 10 Years
    start_date = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Target Period: {start_date} ~ {end_date}")
    print(f"Saving to: {save_dir}")

    # Batch size
    BATCH_SIZE = 50
    total = len(all_tickers)
    success_count = 0
    fail_count = 0

    # Prepare batch (Use index for progress)
    # We iterate until completion

    for i in range(0, total, BATCH_SIZE):
        batch = all_tickers[i : i + BATCH_SIZE]
        batch_tickers = [item["ticker"] for item in batch]

        print(
            f"[{i + 1}/{total}] Processing batch of {len(batch_tickers)} (Start: {batch_tickers[0]})"
        )

        try:
            # Download with threads
            data = yf.download(
                batch_tickers,
                start=start_date,
                end=end_date,
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                # print("  -> Batch Empty.")
                fail_count += len(batch_tickers)
                continue

            # Handle Multi-level vs Single
            # If multi tickers, columns are (Ticker, PriceType) or MultiIndex

            for ticker_obj in batch:
                ticker = ticker_obj["ticker"]

                try:
                    df = pd.DataFrame()
                    if len(batch_tickers) == 1:
                        df = data
                    else:
                        if (
                            isinstance(data.columns, pd.MultiIndex)
                            and ticker in data.columns.levels[0]
                        ):
                            df = data[ticker]

                    if df.empty:
                        pass
                    elif df.isnull().all().all():
                        pass
                    else:
                        df = df.dropna(how="all")
                        if len(df) > 10:
                            path = os.path.join(save_dir, f"{ticker}.csv")
                            df.to_csv(path)
                            success_count += 1
                            continue

                    fail_count += 1

                except Exception as e:
                    fail_count += 1

        except Exception as e:
            print(f"  -> Batch Exception: {e}")
            fail_count += len(batch_tickers)

    print("\n--- Download Summary ---")
    print(f"Total Tickers: {total}")
    print(f"Success: {success_count}")
    print(f"Failed/Empty: {fail_count}")

    # Metadata
    meta_df = pd.DataFrame(all_tickers)
    meta_path = os.path.join(save_dir, "_KR_METADATA.csv")
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    download_data()
