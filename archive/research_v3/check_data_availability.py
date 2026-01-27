import yfinance as yf
import pandas as pd
from datetime import datetime


def check_data():
    tickers = ["^KS11", "KRW=X", "QQQ"]
    print("Fetching data for 2025-2026...")
    try:
        data = yf.download(
            tickers,
            start="2025-01-01",
            end="2026-02-01",
            progress=False,
            group_by="ticker",
        )

        for t in tickers:
            if t in data.columns.levels[0]:
                df = data[t]
                if not df.empty:
                    last_date = df.index[-1]
                    last_price = df["Close"].iloc[-1]
                    print(
                        f"[{t}] Last Date: {last_date.date()} | Last Price: {last_price:.2f}"
                    )

                    # Check 2025 return if possible
                    try:
                        start_2025 = df.loc[df.index >= "2025-01-01"].iloc[0]["Close"]
                        end_2025 = df.loc[df.index <= "2025-12-31"].iloc[-1]["Close"]
                        ret_2025 = (end_2025 - start_2025) / start_2025
                        print(f"    2025 Return: {ret_2025 * 100:.2f}%")
                    except Exception as e:
                        print(f"    Could not calc 2025 return: {e}")
                else:
                    print(f"[{t}] No data found.")
            else:
                print(f"[{t}] Symbol not in download result.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    check_data()
