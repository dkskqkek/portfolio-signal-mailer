import pandas as pd

# Load Results
df = pd.read_csv(
    "d:/gg/strategy_ma185_yield_results.csv", index_col=0, parse_dates=True
)

# Filter 100% Cash Days (Signal == 0.0)
cash_days = df[df["Signal"] == 0.0]

print(f"💰 Total Trading Days: {len(df)}")
print(
    f"🛑 Total 100% Cash Days: {len(cash_days)} ({len(cash_days) / len(df) * 100:.1f}%)"
)
print("-" * 50)
print("📅 Major Cash Periods (Consecutive Days):")

if not cash_days.empty:
    # Group consecutive dates
    cash_days["group"] = (cash_days.index.to_series().diff().dt.days > 3).cumsum()

    for _, group in cash_days.groupby("group"):
        start_date = group.index[0].date()
        end_date = group.index[-1].date()
        duration = len(group)
        if duration > 5:  # Report only significant periods (> 1 week)
            print(f"  • {start_date} ~ {end_date} ({duration} days)")
        # Check specific crisis periods

else:
    print("  None")

print("-" * 50)
# Check 2008
print("🔍 2008 Financial Crisis Check:")
print(df.loc["2008-01-01":"2009-01-01"]["Signal"].value_counts())

print("\n🔍 2022 Instability Check:")
print(df.loc["2022-01-01":"2023-01-01"]["Signal"].value_counts())
