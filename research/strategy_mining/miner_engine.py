# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from research.strategy_mining.data_loader import DataLoader
from research.strategy_mining.alpha_lib import AlphaLib
from typing import Dict, List, Tuple, Any
import logging
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MinerEngine")


class MinerEngine:
    def __init__(self):
        self.loader = DataLoader()
        self.features = {}
        self.raw_data = {}  # O, H, L, C, V

    def prepare_data(self, limit: int = None):
        """Loads data and pre-calculates common features."""
        self.loader.load_all(limit=limit)
        o, h, l, c, v = self.loader.get_aligned_data()
        self.raw_data = {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}

        logger.info("Generating Massive Feature Store...")
        # 1. Moving Averages
        for w in [5, 10, 20, 60, 120, 200]:
            self.features[f"SMA_{w}"] = AlphaLib.sma(c, w)

        # 2. RSI
        for w in [7, 14, 21]:
            self.features[f"RSI_{w}"] = AlphaLib.rsi(c, w)

        # 3. Bollinger Bands & Width
        for w in [20]:
            up, lo = AlphaLib.bollinger_bands(c, w, 2)
            self.features[f"BB_UP_{w}"] = up
            self.features[f"BB_LO_{w}"] = lo
            self.features[f"BB_WIDTH_{w}"] = (up - lo) / self.features[f"SMA_{w}"]

        # 4. Momentum (Returns)
        for w in [1, 5, 20]:
            self.features[f"ROC_{w}"] = c.pct_change(w, fill_method=None)

        # 5. Volume
        self.features["VOL_SMA_20"] = AlphaLib.sma(v, 20)
        self.features["VOL_RATIO"] = v / self.features["VOL_SMA_20"]

        logger.info(f"Feature Store Ready: {len(self.features)} indicators.")

    def run_backtest(self, signal: pd.DataFrame) -> Dict[str, float]:
        """Vectorized Backtest engine."""
        close = self.raw_data["Close"]
        daily_ret = close.pct_change(fill_method=None)

        # Position at T determines Return at T+1
        strategy_ret = signal.shift(1).astype(float) * daily_ret

        ann_factor = 252
        port_ret = strategy_ret.mean(axis=1)
        port_cagr = port_ret.mean() * ann_factor
        port_vol = port_ret.std() * np.sqrt(ann_factor)
        port_sharpe = port_cagr / port_vol if port_vol > 0 else 0

        active_days = signal.shift(1) == True
        if active_days.sum().sum() > 0:
            active_rets = strategy_ret.values[active_days.values]
            wins = active_rets[active_rets > 0]
            losses = active_rets[active_rets <= 0]

            win_rate = len(wins) / len(active_rets)
            avg_win = wins.mean() if len(wins) > 0 else 0.0
            avg_loss = losses.mean() if len(losses) > 0 else 0.0
            pl_ratio = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 0.0
        else:
            win_rate, pl_ratio = 0.0, 0.0

        return {
            "port_sharpe": port_sharpe,
            "port_cagr": port_cagr,
            "win_rate": win_rate,
            "pl_ratio": pl_ratio,
        }

    def mine_massive_composite(self, n_iter: int = 300):
        """
        Phase 4: Combinatorial Logic Search.
        Base Logic (SMA_5) + Random Sub-conditions.
        """
        logger.info(f"Starting Massive Composite Search ({n_iter} iterations)...")

        # Base Winning Logic
        base_name = "Close > SMA_5"
        base_feat = self.raw_data["Close"]
        base_ref = self.features["SMA_5"]
        base_signal = base_feat > base_ref

        # Benchmark
        bnh_metrics = self.run_backtest(
            pd.DataFrame(True, index=base_signal.index, columns=base_signal.columns)
        )
        print(
            f"\n{'[BENCHMARK] Buy & Hold':<45} | Sharpe: {bnh_metrics['port_sharpe']:.4f} | CAGR: {bnh_metrics['port_cagr'] * 100:.1f}%"
        )

        base_metrics = self.run_backtest(base_signal)
        print(
            f"{'[BASE] ' + base_name:<45} | Sharpe: {base_metrics['port_sharpe']:.4f} | CAGR: {base_metrics['port_cagr'] * 100:.1f}%"
        )

        keys = list(self.features.keys())
        results = []

        for _ in tqdm(range(n_iter), desc="Mining Composites"):
            try:
                # Random Condition
                feat_b_name = random.choice(keys)
                feat_b = self.features[feat_b_name]

                op = random.choice([">", "<"])

                if random.random() < 0.4:  # Compare vs Constant
                    if "RSI" in feat_b_name:
                        val = random.choice([30, 40, 50, 60, 70])
                    elif "ROC" in feat_b_name:
                        val = random.choice([-0.05, 0, 0.05])
                    elif "VOL_RATIO" in feat_b_name:
                        val = random.choice([1.0, 1.5, 2.0])
                    else:
                        continue

                    sub_name = f"{feat_b_name} {op} {val}"
                    sub_signal = (feat_b > val) if op == ">" else (feat_b < val)
                else:  # Compare vs another feature
                    feat_c_name = random.choice(keys + ["Close"])
                    if feat_b_name == feat_c_name:
                        continue
                    feat_c = self.features.get(feat_c_name, self.raw_data["Close"])

                    sub_name = f"{feat_b_name} {op} {feat_c_name}"
                    sub_signal = (feat_b > feat_c) if op == ">" else (feat_b < feat_c)

                # Composite (AND)
                combined_signal = base_signal & sub_signal
                combined_name = f"({base_name}) AND ({sub_name})"

                metrics = self.run_backtest(combined_signal)
                metrics["name"] = combined_name
                results.append(metrics)

            except Exception:
                continue

        # Sort and Report
        results.sort(key=lambda x: x["port_sharpe"], reverse=True)

        # Save to CSV
        res_df = pd.DataFrame(results)
        res_path = "d:/gg/research/strategy_mining/composite_results.csv"
        res_df.to_csv(res_path, index=False)
        logger.info(f"Full results saved to {res_path}")

        print(
            f"\n{'Top 10 Composite Strategies (Sorted by Sharpe)':<50} | {'Sharpe':<8} | {'CAGR':<8} | {'WinRate':<8}"
        )
        print("-" * 105)
        for r in results[:10]:
            print(
                f"{r['name']:<50} | {r['port_sharpe']:<8.4f} | {r['port_cagr'] * 100:>6.1f}% | {r['win_rate'] * 100:>6.1f}%"
            )

    def run_with_exit_logic(
        self, entry_signal: pd.DataFrame, exit_type: str, params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Path-dependent simulation for complex exit logic (Trailing Stop, etc.)
        This is slower than purely vectorized backtest but necessary for stops.
        """
        close = self.raw_data["Close"]
        daily_ret = close.pct_change(fill_method=None)

        # Output Signal
        final_signal = pd.DataFrame(False, index=close.index, columns=close.columns)

        # Current status per stock
        in_pos = np.zeros(len(close.columns), dtype=bool)
        entry_price = np.zeros(len(close.columns))
        high_since_entry = np.zeros(len(close.columns))
        bars_held = np.zeros(len(close.columns), dtype=int)

        close_values = close.values
        rsi_values = (
            self.features.get("RSI_14").values if "RSI_14" in self.features else None
        )
        entry_signals = entry_signal.values

        for t in range(1, len(close)):
            # 1. Exit Logic Check
            for i in range(len(close.columns)):
                if in_pos[i]:
                    high_since_entry[i] = max(
                        high_since_entry[i], close_values[t - 1][i]
                    )
                    bars_held[i] += 1

                    exit_triggered = False

                    if exit_type == "trailing_stop":
                        threshold = params.get("threshold", 0.05)
                        if close_values[t - 1][i] < high_since_entry[i] * (
                            1 - threshold
                        ):
                            exit_triggered = True

                    elif exit_type == "rsi":
                        rsi_threshold = params.get("threshold", 70)
                        if (
                            rsi_values is not None
                            and rsi_values[t - 1][i] > rsi_threshold
                        ):
                            exit_triggered = True

                    elif exit_type == "time":
                        days = params.get("days", 5)
                        if bars_held[i] >= days:
                            exit_triggered = True

                    # Also exit if entry signal flips to False (Binary Exit)
                    if not entry_signals[t - 1][i]:
                        exit_triggered = True

                    if exit_triggered:
                        in_pos[i] = False

                # 2. Entry Logic
                elif entry_signals[t - 1][i]:
                    in_pos[i] = True
                    entry_price[i] = close_values[t - 1][i]
                    high_since_entry[i] = close_values[t - 1][i]
                    bars_held[i] = 0

            final_signal.iloc[t] = in_pos

        return self.run_backtest(final_signal)

    def mine_exits(self):
        """Phase 5: Search for optimal exit logic."""
        logger.info("Starting Exit Strategy Mining...")

        # Base Winning Entry: (Close > SMA_5) AND (ROC_1 > 0)
        entry_signal = (self.raw_data["Close"] > self.features["SMA_5"]) & (
            self.features["ROC_1"] > 0
        )

        exit_scenarios = [
            ("Binary (Signal Flip Only)", "binary", {}),  # Default
            ("Trailing Stop 3%", "trailing_stop", {"threshold": 0.03}),
            ("Trailing Stop 5%", "trailing_stop", {"threshold": 0.05}),
            ("RSI Exit 70", "rsi", {"threshold": 70}),
            ("Time Exit 5D", "time", {"days": 5}),
        ]

        print(
            f"\n{'Exit Logic':<25} | {'Sharpe':<8} | {'CAGR':<8} | {'WinRate':<8} | {'P/L Ratio':<8}"
        )
        print("-" * 95)

        for name, etype, params in exit_scenarios:
            if etype == "binary":
                metrics = self.run_backtest(entry_signal)
            else:
                metrics = self.run_with_exit_logic(entry_signal, etype, params)
            print(
                f"{name:<25} | {metrics['port_sharpe']:<8.4f} | {metrics['port_cagr'] * 100:>6.1f}% | {metrics['win_rate'] * 100:>6.1f}% | {metrics['pl_ratio']:<8.2f}"
            )


if __name__ == "__main__":
    miner = MinerEngine()
    miner.prepare_data(limit=1000)  # Increased to 1000 stocks
    miner.mine_exits()
