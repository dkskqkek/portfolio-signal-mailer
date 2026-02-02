import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from custom_indicators import calculate_mmi, calculate_t3, calculate_supertrend
import os


def check_indicators():
    print("Downloading SPY data...")
    df = yf.download("SPY", start="2023-01-01", progress=False)

    if "Adj Close" in df.columns:
        df = df["Adj Close"]
    elif "Close" in df.columns:
        df = df["Close"]
    else:
        # MultiIndex fallback
        try:
            df = df.xs("Adj Close", level=0, axis=1)
        except:
            df = df.xs("Close", level=0, axis=1)

    # yfinance returns DataFrame with ticker column if only 1 ticker is requested sometimes
    # Force Series
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]

    # Create main DF
    data = pd.DataFrame({"Close": df})
    data["High"] = data["Close"]  # Approximation for SuperTrend if OHLC not full
    data["Low"] = data["Close"]

    # Re-download full OHLC for SuperTrend accuracy
    print("Re-downloading full OHLC for SuperTrend...")
    full_df = yf.download("SPY", start="2023-01-01", progress=False)
    # Handle MultiIndex for OHLC
    # Dropping level if exists
    if isinstance(full_df.columns, pd.MultiIndex):
        full_df.columns = full_df.columns.get_level_values(0)

    data = full_df[
        ["Open", "High", "Low", "Close"]
    ].copy()  # Use Close as Adj Close not always available in OHLC bundle easily

    print("Calculating Indicators...")
    # 1. MMI
    data["MMI"] = calculate_mmi(data["Close"], length=20)

    # 2. T3
    data["T3"] = calculate_t3(data["Close"], length=50, vfactor=0.4)

    # 3. SuperTrend
    st_res = calculate_supertrend(data, period=10, factor=3.0)
    data["SuperTrend"] = st_res["SuperTrend"]
    data["ST_Dir"] = st_res["Direction"]

    # Plotting
    print("Plotting...")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    # Ax1: Price, T3, SuperTrend
    ax1.plot(data.index, data["Close"], label="Price", color="black", alpha=0.6)
    ax1.plot(data.index, data["T3"], label="T3(50, 0.4)", color="orange")

    # Plot SuperTrend Line based on Direction
    # Green for Bull, Red for Bear
    up_trend = data[data["ST_Dir"] == 1]
    down_trend = data[data["ST_Dir"] == -1]

    ax1.scatter(
        up_trend.index,
        up_trend["SuperTrend"],
        s=1,
        color="green",
        label="SuperTrend Buy",
    )
    ax1.scatter(
        down_trend.index,
        down_trend["SuperTrend"],
        s=1,
        color="red",
        label="SuperTrend Sell",
    )

    ax1.set_title("SPY Price with T3 & SuperTrend")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ax2: MMI
    ax2.plot(data.index, data["MMI"], label="MMI(20)", color="purple")
    ax2.axhline(0.75, color="gray", linestyle="--", label="Mean Rev (0.75)")
    ax2.axhline(0.3, color="red", linestyle="--", label="Trend (0.3)")

    # Shade MMI < 0.3 zones
    # These should correspond to strong moves
    mmi_trend_zone = data["MMI"] < 0.3
    ax2.fill_between(
        data.index,
        0,
        1,
        where=mmi_trend_zone,
        color="red",
        alpha=0.1,
        transform=ax2.get_xaxis_transform(),
    )

    ax2.set_title("Market Meanness Index (MMI)")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.getcwd(), "indicator_check.png")
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.show()  # Attempt show if local


if __name__ == "__main__":
    check_indicators()
