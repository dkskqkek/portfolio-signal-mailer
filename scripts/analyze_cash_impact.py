import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv(
    "d:/gg/strategy_ma185_yield_results.csv", index_col=0, parse_dates=True
)

# Identify days where we were 100% Cash
cash_only_days = df[df["Position_VTI"] == 0.0].copy()

if cash_only_days.empty:
    print("No 100% Cash days found in simulation.")
else:
    print(f"Analyzing {len(cash_only_days)} days of 100% Cash position...")

    # 1. Total Return during Cash periods
    # Strategy Return (Cash Return/BIL)
    cash_cum_ret = (1 + cash_only_days["Strategy_Ret"]).cumprod().iloc[-1] - 1

    # Market Return (VTI) if we had stayed Invested (100%)
    market_cum_ret = (1 + cash_only_days["VTI_Pct"]).cumprod().iloc[-1] - 1

    # Difference
    impact = cash_cum_ret - market_cum_ret

    print("-" * 50)
    print(f"During 100% Cash Periods ({len(cash_only_days)} days):")
    print(f"  Market (VTI) Return: {market_cum_ret * 100:+.2f}%")
    print(f"  Strategy (Cash) Return: {cash_cum_ret * 100:+.2f}%")
    print(f"  Net Impact: {impact * 100:+.2f}%")

    if impact > 0:
        print("  POSITIVE IMPACT: Avoiding the market saved money.")
    else:
        print("  NEGATIVE IMPACT: Missed market gains (Opportunity Cost).")

    print("-" * 50)

    # 2. Detailed Breakdown by Period
    # Group consecutive days
    cash_only_days["group"] = (
        cash_only_days.index.to_series().diff().dt.days > 3
    ).cumsum()

    print("Period breakdown:")
    for _, group in cash_only_days.groupby("group"):
        start = group.index[0].date()
        end = group.index[-1].date()

        grp_mkt_ret = (1 + group["VTI_Pct"]).cumprod().iloc[-1] - 1
        grp_cash_ret = (1 + group["Strategy_Ret"]).cumprod().iloc[-1] - 1
        diff = grp_cash_ret - grp_mkt_ret

        status = "Saved Loss" if diff > 0 else "Missed Gain"
        print(
            f"  * {start}~{end} ({len(group)}d): VTI {grp_mkt_ret * 100:6.2f}% vs Cash {grp_cash_ret * 100:5.2f}% -> {status} ({diff * 100:+.2f}%)"
        )


import json
analysis = {'total_days': len(cash_only_days), 'market_return': market_cum_ret, 'cash_return': cash_cum_ret, 'net_impact': impact}
with open(r'd:\gg\data\cash_impact.json', 'w') as f: json.dump(analysis, f)
print('Saved JSON')
