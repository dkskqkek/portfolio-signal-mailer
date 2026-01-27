import pandas as pd
import numpy as np
import data_loader


def analyze_defensive_selections():
    print("--- Expanded Pool: Selection History Analysis ---\n")

    START_DATE = "2007-06-01"

    # Expanded Pool
    def_pool = [
        "XLP",
        "XLU",
        "GLD",
        "FXY",
        "UUP",
        "DBC",
        "TLT",
        "IEF",
        "BIL",
        "SHY",
        "SCHP",
        "GSY",
        "XLK",
        "XLV",
        "XLF",
        "XLE",
        "XLI",
        "XLB",
        "XLRE",
        "VNQ",
        "EEM",
        "EFA",
        "VEA",
        "AGG",
        "LQD",
        "HYG",
        "TIP",
    ]

    tickers = list(set(["SPY", "QLD", "QQQ", "KRW=X", "BIL"] + def_pool))

    try:
        df = data_loader.fetch_validated_data(tickers, start_date="2006-01-01")
    except Exception as e:
        print(f"Data Error: {e}")
        return

    close = df.xs("Close", level=1, axis=1)

    sma110 = close["QQQ"].rolling(110).mean()
    sma250 = close["QQQ"].rolling(250).mean()

    start_idx = 260
    trade_dates = close.index

    monthly = close.resample("M").last()
    mom = monthly.pct_change(8)

    # Track selections
    selection_log = []
    prev_signal = "DANGER"

    for i in range(start_idx, len(trade_dates) - 1):
        today = trade_dates[i]
        if today < pd.Timestamp(START_DATE):
            continue

        next_day = trade_dates[i + 1]
        is_month_end = today.month != next_day.month

        # Signal
        q = close.loc[today, "QQQ"]
        s1 = sma110.loc[today]
        s2 = sma250.loc[today]

        signal = "DANGER" if not (q > s1 and q > s2) else "NORMAL"

        # Log selection only when DANGER and month-end (rebalance point)
        if is_month_end and signal == "DANGER":
            m_idx = mom.index.asof(today)
            if pd.notna(m_idx):
                try:
                    row = mom.loc[m_idx]
                    available = [t for t in def_pool if t in row.index]
                    pos = row[available].dropna().sort_values(ascending=False)
                    pos = pos[pos > 0]

                    if len(pos) > 0:
                        top3 = pos.head(3)
                        selection_log.append(
                            {
                                "Date": today.strftime("%Y-%m"),
                                "Top1": top3.index[0] if len(top3) > 0 else "-",
                                "Top1_Mom": f"{top3.iloc[0] * 100:.1f}%"
                                if len(top3) > 0
                                else "-",
                                "Top2": top3.index[1] if len(top3) > 1 else "-",
                                "Top2_Mom": f"{top3.iloc[1] * 100:.1f}%"
                                if len(top3) > 1
                                else "-",
                                "Top3": top3.index[2] if len(top3) > 2 else "-",
                                "Top3_Mom": f"{top3.iloc[2] * 100:.1f}%"
                                if len(top3) > 2
                                else "-",
                            }
                        )
                    else:
                        selection_log.append(
                            {
                                "Date": today.strftime("%Y-%m"),
                                "Top1": "BIL (Default)",
                                "Top1_Mom": "N/A",
                                "Top2": "-",
                                "Top2_Mom": "-",
                                "Top3": "-",
                                "Top3_Mom": "-",
                            }
                        )
                except:
                    pass

        prev_signal = signal

    # Output
    result = pd.DataFrame(selection_log)
    print("=" * 80)
    print("DANGER Mode Selection History (Month-End Rebalancing Points)")
    print("=" * 80)
    print(result.to_string(index=False))

    # Count frequency
    print("\n" + "=" * 80)
    print("Selection Frequency (How often each asset was Top 3)")
    print("=" * 80)

    all_selected = []
    for _, r in result.iterrows():
        for col in ["Top1", "Top2", "Top3"]:
            if r[col] not in ["-", "BIL (Default)"]:
                all_selected.append(r[col])

    freq = pd.Series(all_selected).value_counts()
    print(freq.to_string())

    result.to_csv("research/defensive_selection_history.csv", index=False)
    print(f"\nSaved to: research/defensive_selection_history.csv")


if __name__ == "__main__":
    analyze_defensive_selections()
