import numpy as np
import pandas as pd

# Parameters for Strategic Portfolio
INITIAL_PRINCIPAL = 209_484_136
STRATEGIC_ER = 0.1668  # 16.68%
STRATEGIC_VOL = 0.1291  # 12.91%
SIMULATIONS = 10000


def run_monte_carlo(principal, er, vol, years, n_sim):
    dt = 1 / 252
    steps = int(years * 252)
    drift = (er - 0.5 * vol**2) * dt
    shock = vol * np.sqrt(dt)

    returns = np.random.normal(drift, shock, (steps, n_sim))
    price_paths = principal * np.exp(np.cumsum(returns, axis=0))

    return price_paths


def analyze_results(paths, years):
    ending_values = paths[-1, :]
    median_end = np.median(ending_values)
    p95 = np.percentile(ending_values, 95)
    p5 = np.percentile(ending_values, 5)
    loss_prob = np.mean(ending_values < INITIAL_PRINCIPAL) * 100
    return median_end, p95, p5, loss_prob


print(f"--- Strategic Portfolio Monte Carlo Simulation ---")
print(
    f"Principal: ₩{INITIAL_PRINCIPAL:,.0f} | ER: {STRATEGIC_ER * 100:.2f}% | Vol: {STRATEGIC_VOL * 100:.2f}%"
)

for years in [10, 20]:
    paths = run_monte_carlo(
        INITIAL_PRINCIPAL, STRATEGIC_ER, STRATEGIC_VOL, years, SIMULATIONS
    )
    median_end, p95, p5, loss_prob = analyze_results(paths, years)

    print(f"\n[{years}-Year Result]")
    print(
        f"  Median (50th):   ₩{median_end:15,.0f} (x{median_end / INITIAL_PRINCIPAL:,.1f})"
    )
    print(f"  Optimistic (5th): ₩{p95:15,.0f}")
    print(f"  Pessimistic (95th): ₩{p5:15,.0f}")
    print(f"  Prob. of Loss:   {loss_prob:.2f}%")
