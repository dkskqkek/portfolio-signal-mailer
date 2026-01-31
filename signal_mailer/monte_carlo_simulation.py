"""
Monte Carlo Simulation for Antigravity v3.1 (Week 3)
백테스트 결과의 신뢰도 검증 및 최악 시나리오 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json


class MonteCarloSimulator:
    def __init__(self, backtest_results_path="d:/gg/backtest_results.json"):
        """Monte Carlo 시뮬레이터 초기화"""
        # 백테스트 결과 로드
        with open(backtest_results_path, "r") as f:
            self.bt_results = json.load(f)

        print(f"Loaded backtest results:")
        print(f"  CAGR: {self.bt_results['cagr'] * 100:.2f}%")
        print(f"  MDD: {self.bt_results['max_drawdown'] * 100:.2f}%")
        print(f"  Sharpe: {self.bt_results['sharpe_ratio']:.2f}")

    def simulate(self, n_simulations=1000, seed=42):
        """Monte Carlo 시뮬레이션 실행

        Args:
            n_simulations: 시뮬레이션 횟수
            seed: 재현성을 위한 랜덤 시드
        """
        np.random.seed(seed)

        print(f"\nRunning {n_simulations} Monte Carlo simulations...")

        # 백테스트 기본 통계
        base_cagr = self.bt_results["cagr"]
        base_vol = self.bt_results["volatility"]
        base_sharpe = self.bt_results["sharpe_ratio"]

        # 시뮬레이션 기간 (5년 = 252*5 거래일)
        n_days = 252 * 5

        # 일별 기대 수익률 및 변동성
        daily_return = (1 + base_cagr) ** (1 / 252) - 1
        daily_vol = base_vol / np.sqrt(252)

        # 결과 저장
        sim_results = {"final_values": [], "cagrs": [], "mdds": [], "sharpes": []}

        for i in range(n_simulations):
            # 일별 수익률 생성 (정규분포)
            daily_returns = np.random.normal(daily_return, daily_vol, n_days)

            # 누적 포트폴리오 가치
            portfolio_values = [self.bt_results["initial_capital"]]
            for r in daily_returns:
                portfolio_values.append(portfolio_values[-1] * (1 + r))

            portfolio_values = np.array(portfolio_values)

            # 최종 가치
            final_value = portfolio_values[-1]
            sim_results["final_values"].append(final_value)

            # CAGR
            total_return = final_value / self.bt_results["initial_capital"]
            cagr = (total_return ** (1 / 5)) - 1
            sim_results["cagrs"].append(cagr)

            # MDD
            cummax = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - cummax) / cummax
            mdd = drawdown.min()
            sim_results["mdds"].append(mdd)

            # Sharpe Ratio
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
            sim_results["sharpes"].append(sharpe)

            if (i + 1) % 200 == 0:
                print(f"  Progress: {i + 1}/{n_simulations}")

        print(f"Simulation complete!\n")

        # DataFrame으로 변환
        self.sim_df = pd.DataFrame(sim_results)

        return self.sim_df

    def analyze_results(self):
        """시뮬레이션 결과 분석"""
        print("\n" + "=" * 80)
        print("MONTE CARLO SIMULATION RESULTS")
        print("=" * 80)

        # 백테스트 vs 시뮬레이션 비교
        print("\nBacktest vs Simulation Average\n")

        comparison = pd.DataFrame(
            {
                "Metric": ["CAGR", "MDD", "Sharpe"],
                "Backtest": [
                    f"{self.bt_results['cagr'] * 100:.2f}%",
                    f"{self.bt_results['max_drawdown'] * 100:.2f}%",
                    f"{self.bt_results['sharpe_ratio']:.2f}",
                ],
                "MC Average": [
                    f"{self.sim_df['cagrs'].mean() * 100:.2f}%",
                    f"{self.sim_df['mdds'].mean() * 100:.2f}%",
                    f"{self.sim_df['sharpes'].mean():.2f}",
                ],
            }
        )
        print(comparison.to_string(index=False))

        # 분포 통계
        print("\nSimulation Distribution Stats\n")

        percentiles = [5, 25, 50, 75, 95]

        print("CAGR 분포:")
        for p in percentiles:
            val = np.percentile(self.sim_df["cagrs"], p)
            print(f"  {p:2d}%ile: {val * 100:6.2f}%")

        print("\nMDD 분포:")
        for p in percentiles:
            val = np.percentile(self.sim_df["mdds"], p)
            print(f"  {p:2d}%ile: {val * 100:6.2f}%")

        print("\nSharpe Ratio 분포:")
        for p in percentiles:
            val = np.percentile(self.sim_df["sharpes"], p)
            print(f"  {p:2d}%ile: {val:6.2f}")

        # 최악 시나리오 (5% 하위)
        print("\nWorst Case Scenario (5% percentile)\n")

        worst_cagr = np.percentile(self.sim_df["cagrs"], 5)
        worst_mdd = np.percentile(self.sim_df["mdds"], 5)
        worst_sharpe = np.percentile(self.sim_df["sharpes"], 5)

        print(f"  CAGR:   {worst_cagr * 100:6.2f}%")
        print(f"  MDD:    {worst_mdd * 100:6.2f}%")
        print(f"  Sharpe: {worst_sharpe:6.2f}")

        # 목표 달성 확률
        print("\nTarget Probability\n")

        target_cagr = 0.15
        target_mdd = -0.30
        target_sharpe = 1.0

        prob_cagr = (self.sim_df["cagrs"] >= target_cagr).mean()
        prob_mdd = (self.sim_df["mdds"] >= target_mdd).mean()
        prob_sharpe = (self.sim_df["sharpes"] >= target_sharpe).mean()

        print(f"  CAGR >= 15%:   {prob_cagr * 100:5.1f}%")
        print(f"  MDD >= -30%:   {prob_mdd * 100:5.1f}%")
        print(f"  Sharpe >= 1.0: {prob_sharpe * 100:5.1f}%")

        print("\n" + "=" * 80)

        return {
            "worst_case": {
                "cagr": worst_cagr,
                "mdd": worst_mdd,
                "sharpe": worst_sharpe,
            },
            "probabilities": {
                "target_cagr": prob_cagr,
                "target_mdd": prob_mdd,
                "target_sharpe": prob_sharpe,
            },
        }

    def visualize(self):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Monte Carlo Simulation Results (1000 runs)", fontsize=16, weight="bold"
        )

        # 1. CAGR 분포
        ax = axes[0, 0]
        ax.hist(
            self.sim_df["cagrs"] * 100,
            bins=50,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax.axvline(
            self.bt_results["cagr"] * 100,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Backtest",
        )
        ax.axvline(
            np.percentile(self.sim_df["cagrs"], 5) * 100,
            color="orange",
            linestyle="--",
            linewidth=2,
            label="5%ile (Worst Case)",
        )
        ax.set_xlabel("CAGR (%)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("CAGR Distribution", fontsize=14, weight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. MDD 분포
        ax = axes[0, 1]
        ax.hist(
            self.sim_df["mdds"] * 100,
            bins=50,
            alpha=0.7,
            color="salmon",
            edgecolor="black",
        )
        ax.axvline(
            self.bt_results["max_drawdown"] * 100,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Backtest",
        )
        ax.axvline(
            np.percentile(self.sim_df["mdds"], 5) * 100,
            color="darkred",
            linestyle="--",
            linewidth=2,
            label="5%ile (Worst Case)",
        )
        ax.set_xlabel("Max Drawdown (%)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("MDD Distribution", fontsize=14, weight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. Sharpe Ratio 분포
        ax = axes[1, 0]
        ax.hist(
            self.sim_df["sharpes"],
            bins=50,
            alpha=0.7,
            color="lightgreen",
            edgecolor="black",
        )
        ax.axvline(
            self.bt_results["sharpe_ratio"],
            color="red",
            linestyle="--",
            linewidth=2,
            label="Backtest",
        )
        ax.axvline(
            np.percentile(self.sim_df["sharpes"], 5),
            color="darkgreen",
            linestyle="--",
            linewidth=2,
            label="5%ile (Worst Case)",
        )
        ax.set_xlabel("Sharpe Ratio", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Sharpe Ratio Distribution", fontsize=14, weight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. CAGR vs MDD 산점도
        ax = axes[1, 1]
        scatter = ax.scatter(
            self.sim_df["mdds"] * 100,
            self.sim_df["cagrs"] * 100,
            alpha=0.5,
            c=self.sim_df["sharpes"],
            cmap="viridis",
            s=20,
        )
        ax.scatter(
            self.bt_results["max_drawdown"] * 100,
            self.bt_results["cagr"] * 100,
            color="red",
            s=200,
            marker="*",
            edgecolor="black",
            linewidth=2,
            label="Backtest",
            zorder=5,
        )
        ax.set_xlabel("Max Drawdown (%)", fontsize=12)
        ax.set_ylabel("CAGR (%)", fontsize=12)
        ax.set_title("CAGR vs MDD (colored by Sharpe)", fontsize=14, weight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")

        plt.tight_layout()
        plt.savefig("d:/gg/monte_carlo_results.png", dpi=150, bbox_inches="tight")
        print(f"\nVisualization saved: d:/gg/monte_carlo_results.png")

        return fig


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MONTE CARLO SIMULATION - ANTIGRAVITY v3.1")
    print("=" * 80)

    # 시뮬레이터 생성
    mc = MonteCarloSimulator()

    # 시뮬레이션 실행 (1000회)
    sim_df = mc.simulate(n_simulations=1000)

    # 결과 분석
    analysis = mc.analyze_results()

    # Save to JSON
    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(r"d:\gg\data\monte_carlo_results.json", "w") as f:
        json.dump(analysis, f, cls=NumpyEncoder)
    print(f"Results saved to d:\\gg\\data\\monte_carlo_results.json")

    # 시각화
    fig = mc.visualize()

    print("\n" + "=" * 80)
    print("Monte Carlo Simulation 완료!")
    print("=" * 80)
