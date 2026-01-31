import numpy as np
import pandas as pd

# Constants
INITIAL_PRINCIPAL = 209_484_136  # Current valuation
ANNUAL_ER = 0.0983  # 9.83%
ANNUAL_VOL = 0.165  # Estimated 16.5% volatility for this mix
YEARS = 20
SIMULATIONS = 10000


def run_monte_carlo(principal, er, vol, years, n_sim):
    # Daily steps
    dt = 1 / 252
    steps = int(years * 252)

    # Drift and shock
    # r = (mu - 0.5 * sigma^2) * dt + sigma * epsilon * sqrt(dt)
    drift = (er - 0.5 * vol**2) * dt
    shock = vol * np.sqrt(dt)

    returns = np.random.normal(drift, shock, (steps, n_sim))
    price_paths = principal * np.exp(np.cumsum(returns, axis=0))

    return price_paths


# Run simulation
paths = run_monte_carlo(INITIAL_PRINCIPAL, ANNUAL_ER, ANNUAL_VOL, YEARS, SIMULATIONS)

# Results analysis
ending_values = paths[-1, :]
median_end = np.median(ending_values)
p95 = np.percentile(ending_values, 95)
p5 = np.percentile(ending_values, 5)
loss_prob = np.mean(ending_values < INITIAL_PRINCIPAL) * 100

# Annual milestones (Median)
milestones = {year: np.median(paths[year * 252 - 1, :]) for year in range(1, YEARS + 1)}

print(f"--- Monte Carlo Simulation ({SIMULATIONS} trials, {YEARS} years) ---")
print(f"Initial Principal: ₩{INITIAL_PRINCIPAL:,.0f}")
print(f"Expected Return: {ANNUAL_ER * 100:.2f}%, Volatility: {ANNUAL_VOL * 100:.2f}%")

print(f"\n[Result at {YEARS} Years]")
print(f"Median (50th): ₩{median_end:,.0f} (x{median_end / INITIAL_PRINCIPAL:.1f})")
print(f"Optimistic (Top 5%): ₩{p95:,.0f}")
print(f"Pessimistic (Bottom 5%): ₩{p5:,.0f}")
print(f"Probability of Loss: {loss_prob:.2f}%")

print(f"\n[Timeline (Median Growth)]")
for y, val in milestones.items():
    print(f"Year {y}: ₩{val:,.0f}")
