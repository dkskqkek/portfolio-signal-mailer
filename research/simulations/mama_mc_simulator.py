import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MAMA-MC-Simulator")


class MAMAMonteCarlo:
    def __init__(self, historical_returns_path: str):
        self.raw_data = pd.read_csv(
            historical_returns_path, index_col=0, parse_dates=True
        )
        # Drop macro indicators from the pool of investable assets
        self.macro_cols = ["^VIX", "^TNX", "SPY", "QQQ"]
        self.gnn_pool = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            "NFLX",
            "AVGO",
        ]
        self.defensive_pool = ["BIL", "TLT"]

        # Clean data: Use common period for correlation but handle existence
        # For simulation, we can bootstrap or use parametric dist
        self.returns = self.raw_data.copy()

    def _get_regime_stats(self) -> Dict[str, Dict]:
        """
        Calculates simple regime statistics.
        In production, MAMA Lite uses a complex GNN/SRL model.
        For MC, we simplify:
        - High Vol/Bear: SPY < 50MA or VIX > 25
        - Normal/Bull: Otherwise
        """
        # We'll use the SPY column to proxy the market
        spy_ret = self.returns["SPY"]
        # Simplified: Use 200-day rolling mean of SPY to define regimes in history
        # (Though we'll sample randomly, we want to know the 'state' of the samples)
        # However, for Monte Carlo, we often sample 'Blocks' or use a Markov Chain.

        # Let's simplify: 1. Bull (High tech returns), 2. Defensive (Bonds lead)
        stats = {
            "BULL": {
                "tickers": self.gnn_pool,
                "prob": 0.65,  # Historical bull freq
            },
            "DEFENSIVE": {"tickers": self.defensive_pool, "prob": 0.35},
        }
        return stats

    def run_simulation(self, iterations=1000, years=20, initial_capital=100000):
        days = int(years * 252)
        all_results = []

        # Covariance matrix for consistent sampling
        # We need to fill NaNs for the cov matrix calculation
        clean_returns = self.returns.fillna(0)
        cov_matrix = clean_returns.cov()
        mean_returns = clean_returns.mean()

        logger.info(f"Running {iterations} iterations for {years} years...")

        for i in range(iterations):
            # Parametric Multi-variate Normal Sample (Geometric Brownian Motion)
            # Alternatively, we could do Bootstrapping. Let's do Bootstrapping for 'Fat Tails' (MAMA focus)

            # 1. Randomly sample indices with replacement
            indices = np.random.choice(len(self.returns), size=days, replace=True)
            sampled_rets = self.returns.iloc[indices]

            # Start Portfolio
            portfolio_val = initial_capital
            path = [portfolio_val]

            # For MAMA Lite, we rebalance periodically (e.g. Weekly or Daily based on regime)
            # We'll simulate 'Daily' rebalance for simplicity in this MC

            for d in range(days):
                curr_row = sampled_rets.iloc[d]

                # Logic: MAMA Lite SRL Simulation
                # If VIX z-score > 0 (High risk) -> Defensive
                # In bootstrapped data, we check the VIX of THAT day
                vix_val = curr_row.get("^VIX", 0)  # This is % change, not level.
                # Note: Our CSV has pct_change. For macro, we might need levels.
                # Let's use SPY performance as a proxy for regime in this MC.

                # Simplified Engine:
                # If SPY return of the sampled day is < -1%, 40% chance of being in "Defensive" mode
                # Or just use the original MAMA Lite target:
                # If in Bull -> Top 3 GNN (Equally weighted)
                # If in Defensive -> 50/50 BIL/TLT

                # We simulate the selection by picking Top 3 Tech stocks based on the sampled day's drift
                # (Assuming GNN finds the ones with momentum)
                if np.random.random() < 0.65:  # 65% Bullish/Neutral prob
                    # Pick 3 random stocks from GNN pool to simulate selection
                    selected = np.random.choice(self.gnn_pool, 3, replace=False)
                    day_ret = (
                        sum(curr_row[s] for s in selected if not np.isnan(curr_row[s]))
                        / 3.0
                    )
                else:
                    # Defensive (50/50 BIL/TLT)
                    day_ret = (curr_row.get("BIL", 0) + curr_row.get("TLT", 0)) / 2.0

                # Apply return (minus slippage/fees 0.05%)
                portfolio_val *= 1 + day_ret - 0.0002
                path.append(portfolio_val)

            all_results.append(path)

        return np.array(all_results)

    def analyze(self, results, initial_capital=100000):
        # Final values
        final_values = results[:, -1]

        # CAGR
        years = (results.shape[1] - 1) / 252
        cagr = (final_values / initial_capital) ** (1 / years) - 1

        # MDD
        def calculate_mdd(path):
            path_series = pd.Series(path)
            roll_max = path_series.cummax()
            drawdown = (path_series - roll_max) / roll_max
            return drawdown.min()

        mdds = np.array([calculate_mdd(p) for p in results])

        # Sharpe (Approx)
        returns_paths = np.diff(results, axis=1) / results[:, :-1]
        sharpes = np.array(
            [
                (np.mean(r) * 252 - 0.04) / (np.std(r) * np.sqrt(252))
                if np.std(r) > 0
                else 0
                for r in returns_paths
            ]
        )

        print("\n=== MAMA Lite Monte Carlo Results (20 Years) ===")
        print(f"Iterations: {len(results)}")
        print("-" * 40)
        print(f"Median CAGR: {np.median(cagr) * 100:.2f}%")
        print(f"Mean CAGR:   {np.mean(cagr) * 100:.2f}%")
        print(f"Worst Case (5%): {np.percentile(cagr, 5) * 100:.2f}%")
        print(f"Best Case (95%): {np.percentile(cagr, 95) * 100:.2f}%")
        print("-" * 40)
        print(f"Median MDD: {np.median(mdds) * 100:.2f}%")
        print(f"Worst MDD (Max): {np.min(mdds) * 100:.2f}%")
        print("-" * 40)
        print(f"Median Sharpe Ratio: {np.median(sharpes):.2f}")

        # Plotting
        plt.figure(figsize=(12, 6))
        # Plot first 50 paths
        for i in range(min(50, len(results))):
            plt.plot(results[i], color="gray", alpha=0.1)

        # Plot Median path
        plt.plot(
            np.median(results, axis=0), color="blue", linewidth=2, label="Median Path"
        )
        plt.yscale("log")
        plt.title("MAMA Lite 20-Year Monte Carlo (Log Scale)")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("d:/gg/research/mama_mc_20yr.png")
        print(f"\nSimulation chart saved to d:/gg/research/mama_mc_20yr.png")


if __name__ == "__main__":
    simulator = MAMAMonteCarlo("d:/gg/data/mama_lite_historical_returns.csv")
    results = simulator.run_simulation(iterations=100, years=5)
    simulator.analyze(results)
