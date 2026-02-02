import yfinance as yf
import pandas as pd
import numpy as np


# Re-implement logic briefly for analysis
def get_signals():
    symbols = ["SPY", "VEA", "VWO", "BND"]
    data = yf.download(symbols, start="2020-01-01", progress=False)

    if "Adj Close" in data.columns:
        data = data["Adj Close"]
    elif "Close" in data.columns:
        data = data["Close"]

    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten or select level 1 if needed, but let's try direct access
        try:
            spy = (
                data.xs("SPY", level=1, axis=1)
                if "SPY" in data.columns.get_level_values(1)
                else data["SPY"]
            )
        except:
            # Fallback for simple columns
            pass

    # PAA
    window = 252
    momentum = data.pct_change(window)
    breadth = (momentum > 0).sum(axis=1)
    # Using 4 assets: SPY, VEA, VWO, BND
    # Threshold < 2 means 0 or 1 positive -> Cash. >= 2 -> Invest
    # Logic in previous code: np.where(breadth < (total_assets / 2), 0, 1) -> < 2.0 -> 0, else 1
    paa_signal = np.where(breadth < (len(data.columns) / 2), 0, 1)
    paa_series = pd.Series(paa_signal, index=data.index)

    # Buffer
    price = data["SPY"]
    # Check if price is Series or DataFrame
    if isinstance(price, pd.DataFrame):
        price = price.iloc[:, 0]

    ma = price.rolling(window=185).mean()
    buffer = 0.03

    buffer_sig = pd.Series(index=price.index, data=np.nan)
    state = 1

    # Vector loop
    price_val = price.values
    ma_val = ma.values

    sig_vals = []

    for i in range(len(price)):
        if np.isnan(ma_val[i]):
            sig_vals.append(0)
            continue

        upper = ma_val[i] * (1 + buffer)
        lower = ma_val[i] * (1 - buffer)

        curr = price_val[i]

        if curr > upper:
            state = 1
        elif curr < lower:
            state = 0
        sig_vals.append(state)

    buffer_series = pd.Series(sig_vals, index=price.index)

    return paa_series, buffer_series


try:
    paa, buf = get_signals()

    print("\n--- Strategy Statistics (2020 ~ Now) ---")
    print(f"PAA Trades: {(paa.diff() != 0).sum() // 2} round trips")
    print(f"Buffer Trades: {(buf.diff() != 0).sum() // 2} round trips")

    print("\n--- Recent Signal Changes ---")
    # Get last 5 changes for each
    paa_changes = paa[paa.diff() != 0].tail(5)
    buf_changes = buf[buf.diff() != 0].tail(5)

    print("PAA Last 5 Changes:")
    for d, s in paa_changes.items():
        print(f"  {d.date()}: {'BUY' if s == 1 else 'SELL'}")

    print("\nBuffer Last 5 Changes:")
    for d, s in buf_changes.items():
        print(f"  {d.date()}: {'BUY' if s == 1 else 'SELL'}")

except Exception as e:
    print(f"Analysis Error: {e}")
