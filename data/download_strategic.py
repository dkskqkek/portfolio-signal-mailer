import yfinance as yf
import os

DATA_DIR = r"d:\gg\data\historical"
os.makedirs(DATA_DIR, exist_ok=True)

tickers = ["DBMF", "JEPI"]

for t in tickers:
    print(f"Downloading {t}...")
    try:
        # Auto adjust = True gives Adj Close as 'Close' usually, or separate.
        # Let's simple download and save 'Close' if available
        df = yf.download(t, start="2010-01-01", progress=False)

        # yfinance structure varies by version.
        # Usually checking columns.
        if "Adj Close" in df.columns:
            series = df["Adj Close"]
        elif "Close" in df.columns:
            series = df["Close"]
        else:
            # Multi-level column fix for newer yfinance
            # if df.columns is MultiIndex, try to access
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    # Try getting Close/ticker
                    series = df.xs("Close", axis=1, level=0)[t]
            except:
                series = df.iloc[:, 0]  # Fallback

        # Save as single column CSV with Date index
        series.name = "Close"
        series.to_csv(os.path.join(DATA_DIR, f"{t}.csv"))
        print(f"Saved {t}.csv")

    except Exception as e:
        print(f"Failed {t}: {e}")
