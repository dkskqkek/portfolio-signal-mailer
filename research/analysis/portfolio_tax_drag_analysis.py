import numpy as np
import pandas as pd

# Initial Principal
INITIAL_PRINCIPAL = 209_484_136

# Portfolio Assumptions (User Strategic)
# ER (Gross): 16.68%, Vol: 12.91%
# Let's break down into Appreciation vs Dividend Yield for tax modeling
# Estimated Avg Div Yield: ~2.5% (SCHD 3.5%, GOOGL 0.5%, GLD 0%, etc.)
ANNUAL_DIV_YIELD = 0.025
ANNUAL_APPRECIATION = 0.1418  # 16.68% - 2.5%
ANNUAL_VOL = 0.1291

# Tax Rules (South Korea)
DIVIDEND_TAX = 0.15  # 15%
CAPITAL_GAINS_TAX = 0.22  # 22% (Overseas stocks)
DEDUCTION = 2_500_000  # KRW annual deduction for capital gains

SIMULATIONS = 5000
YEARS = 20


def run_tax_sim(principal, appreciation, div_yield, vol, years, n_sim):
    dt = 1  # Simulation on yearly steps for tax clarity

    # Paths to track
    # 0: Gross (Reinvested, No Tax)
    # 1: Net (Reinvested after Tax)
    gross_paths = np.zeros((years + 1, n_sim))
    net_paths = np.zeros((years + 1, n_sim))

    gross_paths[0] = principal
    net_paths[0] = principal

    for y in range(1, years + 1):
        # Generate annual return (lognormal)
        # Drift = mu - 0.5 * sigma^2
        drift = appreciation - 0.5 * vol**2
        shock = vol * np.random.normal(0, 1, n_sim)
        growth_rate = np.exp(drift + shock)

        # 1. Gross Growth (Simulating previous results)
        total_er = appreciation + div_yield
        gross_drift = total_er - 0.5 * vol**2
        gross_growth = np.exp(gross_drift + vol * np.random.normal(0, 1, n_sim))
        gross_paths[y] = gross_paths[y - 1] * gross_growth

        # 2. Net Growth (Realistic)
        # Price appreciation
        net_appreciation = net_paths[y - 1] * (growth_rate - 1)
        # Dividend income
        div_income = net_paths[y - 1] * div_yield
        # Tax on dividends
        net_div = div_income * (1 - DIVIDEND_TAX)

        # New balance before capital gains tax (unrealized)
        net_paths[y] = net_paths[y - 1] + net_appreciation + net_div

    # After 'years', we realize capital gains tax on the total profit
    total_profit = net_paths[years] - principal
    taxable_profit = np.maximum(
        0, total_profit - (DEDUCTION * years)
    )  # Simplified deduction over time
    final_net = net_paths[years] - (taxable_profit * CAPITAL_GAINS_TAX)

    return gross_paths, net_paths, final_net


gross_p, net_p, final_n = run_tax_sim(
    INITIAL_PRINCIPAL,
    ANNUAL_APPRECIATION,
    ANNUAL_DIV_YIELD,
    ANNUAL_VOL,
    YEARS,
    SIMULATIONS,
)


def report(years):
    idx = years
    m_gross = np.median(gross_p[idx])
    m_net_realized = np.median(
        final_n if years == YEARS else net_p[idx]
    )  # Only apply CGT at the very end

    drag = (1 - (m_net_realized / m_gross)) * 100

    print(f"\n[{years}-Year Comparison]")
    print(f"  Gross (No Tax): ₩{m_gross:15,.0f}")
    print(f"  Net (After Tax): ₩{m_net_realized:15,.0f}")
    print(f"  Tax Drag:       {drag:.2f}% loss to taxes")


print(f"--- Tax & Dividend Impact Analysis ---")
print(
    f"Assumptions: Div Yield {ANNUAL_DIV_YIELD * 100}%, Div Tax {DIVIDEND_TAX * 100}%, CGT {CAPITAL_GAINS_TAX * 100}%"
)

report(10)
report(20)
