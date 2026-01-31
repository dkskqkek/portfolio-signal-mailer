import pandas as pd

# USD/KRW assumed (as of early 2026/late 2025 context)
EXCHANGE_RATE = 1380.0

data = [
    # Ticker, Value, Currency, Expected_Return (%)
    ("KODEX 미국S&P500", 34095600, "KRW", 10.0),
    ("GOOGL", 20434.58, "USD", 14.0),
    ("KODEX 미국나스닥100", 24739950, "KRW", 12.0),
    ("VGT", 14489.16, "USD", 12.0),
    ("TIGER 미국배당다우존스", 18142880, "KRW", 9.0),
    ("COWZ", 9754.38, "USD", 11.0),
    ("XLV", 8017.07, "USD", 8.0),
    ("CVX", 7876.02, "USD", 7.0),
    ("VXUS", 7481.67, "USD", 7.0),
    ("TIGER 코리아TOP10", 10556150, "KRW", 6.0),
    ("KODEX 코리아밸류업", 10437700, "KRW", 7.5),
    ("GLDM", 6746.09, "USD", 5.0),
    ("ACE KRX금현물", 6131120, "KRW", 5.0),
    ("NVR", 1231.37, "USD", 10.0),
    ("KRW 현금", 458867, "KRW", 3.5),
]

df = pd.DataFrame(data, columns=["Asset", "Value", "Currency", "ER_Pct"])

# Convert to KRW
df["Value_KRW"] = df.apply(
    lambda x: x["Value"] * EXCHANGE_RATE if x["Currency"] == "USD" else x["Value"],
    axis=1,
)

total_value = df["Value_KRW"].sum()
df["Weight"] = df["Value_KRW"] / total_value
df["Contribution"] = df["Weight"] * df["ER_Pct"]

portfolio_er = df["Contribution"].sum()

print(f"--- Portfolio Expected Return Analysis ---")
print(f"Total Valuation: ₩{total_value:,.0f}")
print(f"Exchange Rate used: {EXCHANGE_RATE}")
print(f"\nExpected Return: {portfolio_er:.2f}% per year")
print(f"\nWeighted Contribution by Category:")
# Grouping for summary
df["Category"] = df["Asset"].apply(
    lambda x: "미국 주식/ETF"
    if any(
        s in x for s in ["미국", "VGT", "GOOGL", "COWZ", "XLV", "CVX", "VXUS", "NVR"]
    )
    else (
        "한국 주식/ETF"
        if "코리아" in x
        else ("금" if "금" in x or "GLD" in x else "현금")
    )
)
summary = df.groupby("Category").agg({"Weight": "sum", "Contribution": "sum"})
summary["Avg_ER"] = summary["Contribution"] / summary["Weight"]
print(summary[["Weight", "Avg_ER"]])
