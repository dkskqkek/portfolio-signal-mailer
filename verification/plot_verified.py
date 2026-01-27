# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf


def plot_results():
    # 1. Load Data
    try:
        df_qld = pd.read_csv(
            "research/verified_backtest_qld.csv", parse_dates=["Date"], index_col="Date"
        )
        df_qqq = pd.read_csv(
            "research/verified_backtest_qqq.csv", parse_dates=["Date"], index_col="Date"
        )
    except FileNotFoundError:
        print("Run backtest_v4_1_verified.py first!")
        return

    # 2. Fetch Benchmark (SPY)
    start = df_qld.index[0]
    end = df_qld.index[-1]

    print("Fetching SPY benchmark...")
    spy_data = yf.download(
        "SPY", start=start, end=end, progress=False, auto_adjust=True
    )

    spy = pd.Series(dtype=float)

    # Robust Extraction
    try:
        if isinstance(spy_data.columns, pd.MultiIndex):
            # Check levels for 'Close'
            if "Close" in spy_data.columns.levels[0]:
                spy = spy_data["Close"]
            elif "Close" in spy_data.columns.levels[1]:
                spy = spy_data.xs("Close", level=1, axis=1)
            else:
                if "Adj Close" in spy_data.columns.levels[0]:
                    spy = spy_data["Adj Close"]
        else:
            # Flat
            if "Close" in spy_data.columns:
                spy = spy_data["Close"]
            elif "Adj Close" in spy_data.columns:
                spy = spy_data["Adj Close"]

        # If result is DataFrame with 1 col, squeeze it
        if isinstance(spy, pd.DataFrame):
            if spy.shape[1] == 1:
                spy = spy.iloc[:, 0]
            elif "SPY" in spy.columns:
                spy = spy["SPY"]

    except Exception as e:
        print(f"SPY Extraction Error: {e}")

    if spy.empty:
        print("⚠️ Failed to fetch SPY data. Using flat line.")
        spy = pd.Series(data=df_qld["Equity"].iloc[0], index=df_qld.index)

    spy = spy.reindex(df_qld.index).ffill()

    # Normalize
    initial = df_qld["Equity"].iloc[0]
    spy_first = spy.iloc[0]
    if hasattr(spy_first, "item"):
        spy_first = spy_first.item()

    spy_norm = (spy / spy_first) * initial

    # 3. Plot
    plt.figure(figsize=(12, 7))

    # Strategy 2x
    plt.plot(
        df_qld.index,
        df_qld["Equity"],
        label="Antigravity 2x (QLD)",
        color="#2ecc71",
        linewidth=2,
    )

    # Strategy 1x
    plt.plot(
        df_qqq.index,
        df_qqq["Equity"],
        label="Antigravity 1x (QQQ)",
        color="#3498db",
        linewidth=2,
        linestyle="-",
    )

    # Benchmark
    plt.plot(
        spy_norm.index,
        spy_norm,
        label="SPY (B&H)",
        color="gray",
        linestyle="--",
        alpha=0.6,
    )

    plt.title(f"Leverage Comparison: 2x vs 1x ({start.year}-{end.year})", fontsize=14)
    plt.yscale("log")
    plt.ylabel("Equity (Log Scale)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    # Annotate Final Values
    def annotate(series, color, offset):
        val = series.iloc[-1]
        if hasattr(val, "item"):
            val = val.item()
        plt.annotate(
            f"${val:,.0f}",
            xy=(series.index[-1], val),
            xytext=(10, offset),
            textcoords="offset points",
            color=color,
            fontweight="bold",
        )

    annotate(df_qld["Equity"], "#2ecc71", 10)
    annotate(df_qqq["Equity"], "#3498db", 0)
    annotate(spy_norm, "gray", -15)

    output_path = "research/verified_backtest_chart_comparison.png"
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    plot_results()
