import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta


def fetch_validated_data(tickers, start_date="2000-01-01", max_gap_days=5):
    """
    Fetches data from yfinance and performs strict validation.
    Includes ^IRX proxy logic for early BIL simulation.
    """
    print(f"[Data Loader] Fetching {len(tickers)} tickers from {start_date}...")

    # Ensure list
    if isinstance(tickers, str):
        tickers = [tickers]

    # Force inclusion of key assets
    if "SPY" not in tickers:
        tickers.append("SPY")

    # Need ^IRX for BIL proxy if BIL is in tickers
    download_tickers = tickers.copy()
    if "BIL" in tickers and "^IRX" not in tickers:
        download_tickers.append("^IRX")

    # Auto adjust = True handles splits and dividends for OHLC
    data = yf.download(
        download_tickers,
        start=start_date,
        progress=False,
        group_by="ticker",
        auto_adjust=True,
    )

    if data.empty:
        raise ValueError("No data fetched. Check internet connection or tickers.")

    # --- BIL Proxy Logic ---
    if "BIL" in tickers and "^IRX" in data.columns.levels[0]:
        print("[Data Loader] Synthesizing BIL history using ^IRX...")
        bil_col = data["BIL"]["Close"]
        irx_col = data["^IRX"]["Close"]

        # IRX is Annual Yield %. Daily return approx = (Yield/100)/252
        # Use ffill to avoid NaNs on holidays for IRX
        daily_rf = (irx_col.fillna(method="ffill") / 100.0) / 252.0

        # Create synthetic index (start 1.0)
        # We start from the beginning of IRX data
        cash_index = (1 + daily_rf).cumprod()

        # Align: Find first valid real BIL datum
        first_valid = bil_col.first_valid_index()
        if first_valid:
            # Scale synth index to match BIL at inception
            # synth_price_t = cash_index_t * (bil_inception / cash_index_inception)
            inception_price = bil_col.loc[first_valid]
            inception_idx_val = cash_index.loc[first_valid]

            if pd.notna(inception_idx_val) and inception_idx_val != 0:
                scale_factor = inception_price / inception_idx_val
                synthetic_bil = cash_index * scale_factor

                # Fill NaNs in BIL OHLC
                # We apply the same synthetic 'Close' pattern to Open/High/Low to simulate 'no volatility' except yield
                # Just filling with the same value is acceptable for Cash proxy.
                cols_to_fill = ["Open", "High", "Low", "Close"]
                for col in cols_to_fill:
                    if col in data["BIL"].columns:
                        # Slice allows setting values
                        chunk = data["BIL"][col]
                        # Only fill NaNs before inception
                        mask = chunk.index < first_valid
                        # Generate filling series
                        fill_vals = synthetic_bil[mask]
                        # Update
                        data.loc[mask, ("BIL", col)] = fill_vals

    # Remove ^IRX if not requested
    if "^IRX" not in tickers:
        if "^IRX" in data.columns.levels[0]:
            data = data.drop("^IRX", axis=1, level=0)

    # --- Validation ---
    spy_idx = data.index
    diffs = spy_idx.to_series().diff().dt.days
    gaps = diffs[diffs > max_gap_days]

    if not gaps.empty:
        print(f"\n[CRITICAL WARNING] Data Gaps Detected (> {max_gap_days} days):")
        print(gaps)
        if gaps.max() > 20:
            raise ValueError(f"FATAL: Massive data gap of {gaps.max()} days detected.")

    # Fill minor holes (holidays)
    data = data.ffill(limit=5)

    print("[Data Loader] Validation Passed.")
    return data


if __name__ == "__main__":
    df = fetch_validated_data(["SPY", "BIL"], start_date="2005-01-01")
    print(df["BIL"].head())
