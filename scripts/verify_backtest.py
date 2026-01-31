import pandas as pd
import numpy as np
import os
import glob

# Find latest backtest file
result_dir = r"d:\gg\data\backtest_results"
list_of_files = glob.glob(os.path.join(result_dir, "backtest_daily_*.csv"))
latest_file = max(list_of_files, key=os.path.getctime)

print(f"Analyzing: {latest_file}")
df = pd.read_csv(latest_file)
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

initial_capital = 100000
final_value = df["value"].iloc[-1]
years = (df.index[-1] - df.index[0]).days / 365.25

# CAGR
cagr = (final_value / initial_capital) ** (1 / years) - 1

# MDD
cummax = df["value"].cummax()
drawdown = (df["value"] - cummax) / cummax
mdd = drawdown.min()

# Sharpe
daily_ret = df["value"].pct_change().dropna()
sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)

print(f"CAGR: {cagr:.2%}")
print(f"MDD: {mdd:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"Final Value: ${final_value:,.2f}")
