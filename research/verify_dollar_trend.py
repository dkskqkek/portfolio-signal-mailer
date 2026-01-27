# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


class MarketRegimeDetector:
    """
    Market Regime Detection for 2026 Strategy

    Key Regimes:
    - Dollar Weakness: KRW=X rising (won strengthening)
    - KOSPI Boom: Strong momentum + structural reform
    - Risk-Off: VIX spike, global selloff
    """

    def __init__(self):
        self.regime_cache = {}

    def verify_dollar_trend(self):
        """
        Verify dollar weakness hypothesis with robust logic
        """
        print("\n" + "=" * 70)
        print(" DOLLAR TREND VERIFICATION")
        print("=" * 70)

        print("\nFetching Currency Data (2024-2026)...")
        tickers = ["KRW=X", "GLD", "DX-Y.NYB"]

        try:
            data = yf.download(
                tickers,
                start="2024-01-01",
                end="2026-02-01",
                progress=False,
                group_by="ticker",
            )

            df = pd.DataFrame()
            for t in tickers:
                try:
                    target = data[t]
                    col = "Adj Close" if "Adj Close" in target.columns else "Close"
                    df[t] = target[col]
                except:
                    pass

            if df.empty:
                print("❌ No data fetched.")
                return None

            df = df.ffill()

            # 1. 2025 Performance Check
            try:
                start_2025 = df.loc["2025-01-01":"2025-01-10"].iloc[0]
                end_2025 = df.loc["2025-12-20":"2025-12-31"].iloc[-1]
                print("\n📊 2025 Performance Check:")
                for t in df.columns:
                    ret = (end_2025[t] - start_2025[t]) / start_2025[t]
                    print(f"   {t:<12}: {'↑' if ret > 0 else '↓'} {ret * 100:>7.2f}%")
            except:
                pass

            # 2. Current Trend (Last 60 Days)
            print("\n📈 Current Trend (Last 60 Trading Days):")
            recent = df.iloc[-60:]
            trends = {}
            for t in df.columns:
                ret = (recent[t].iloc[-1] - recent[t].iloc[0]) / recent[t].iloc[0]
                trends[t] = ret
                print(
                    f"   {t:<12}: {'↑' if ret > 0 else '↓'} {ret * 100:>7.2f}% (Curr: {recent[t].iloc[-1]:.2f})"
                )

            # 3. Decision Logic (SMA Crossover)
            series = df["KRW=X"]
            curr_p, ma_50, ma_200 = (
                series.iloc[-1],
                series.rolling(50).mean().iloc[-1],
                series.rolling(200).mean().iloc[-1],
            )
            is_death_cross = ma_50 < ma_200

            print(f"\n🎯 Verification Outcome (KRW=X):")
            print(
                f"   Status: {'📉 WEAK DOLLAR' if is_death_cross else '📈 STRONG DOLLAR'}"
            )
            print(f"   SMA 50/200: {ma_50:.1f} / {ma_200:.1f}")

            return "WEAK" if is_death_cross else "STRONG"

        except Exception as e:
            print(f"Error: {e}")
            return None


class Portfolio2026:
    """
    2026 Era Adaptive Portfolio Strategy (Survival Mode)
    """

    def __init__(self, mdd_cutoff=-0.40):
        self.mdd_cutoff = mdd_cutoff
        self.leverage_cost = 0.015 / 252
        self.tx_cost = 0.002
        self.emergency_mode = False
        self.peak_value = 1.0
        self.regime_detector = MarketRegimeDetector()

    def fetch_data(self):
        tickers = ["^KS11", "QQQ", "GLD", "BIL", "KRW=X", "SCHD"]
        data = yf.download(
            tickers,
            start="2024-01-01",
            end="2026-02-01",
            progress=False,
            group_by="ticker",
        )
        df = pd.DataFrame()
        for t in tickers:
            col = "Adj Close" if "Adj Close" in data[t].columns else "Close"
            df[t] = data[t][col]
        return df.ffill().dropna()

    def run_simulation(self):
        regime = self.regime_detector.verify_dollar_trend()
        if regime == "STRONG":
            print("⚠️ WARNING: Dollar strength detected. Proceeding with caution.")

        df = self.fetch_data()
        df["USD_KRW"] = df["KRW=X"]
        df["QQQ_KRW"] = df["QQQ"] * df["USD_KRW"]

        # Look-Ahead Free Momentum & Selection
        df["KOSPI_Mom"] = df["^KS11"].pct_change(126)
        df["QQQ_Mom_KRW"] = df["QQQ_KRW"].pct_change(126)
        df["Selected"] = np.where(
            df["KOSPI_Mom"].shift(1) > df["QQQ_Mom_KRW"].shift(1), "KOSPI", "QQQ"
        )

        # Trend Filter (Zero Look-Ahead)
        df["K_SMA"] = df["^KS11"].rolling(130).mean()
        df["Q_SMA"] = df["QQQ"].rolling(130).mean()

        # Returns with Decay & FX
        k_vol = df["^KS11"].pct_change().rolling(20).std()
        k_decay = 0.5 * 2.0 * (2.0 - 1) * (k_vol**2)
        df["Ret_K_2x"] = (df["^KS11"].pct_change() * 2.0) - k_decay - self.leverage_cost

        q_ret = df["QQQ"].pct_change()
        q_vol = q_ret.rolling(20).std()
        q_decay = 0.5 * 2.0 * (2.0 - 1) * (q_vol**2)
        df["Ret_QLD_USD"] = (q_ret * 2.0) - q_decay - self.leverage_cost
        df["Ret_QLD_KRW"] = (1 + df["Ret_QLD_USD"]) * (
            1 + df["USD_KRW"].pct_change()
        ) - 1

        # Core
        df["Ret_SCHD_KRW"] = (1 + df["SCHD"].pct_change()) * (
            1 + df["USD_KRW"].pct_change()
        ) - 1
        df["Ret_GLD_KRW"] = (1 + df["GLD"].pct_change()) * (
            1 + df["USD_KRW"].pct_change()
        ) - 1

        # Simulation
        strat_rets, p_vals = [], [1.0]
        prev_s = None
        for i in range(1, len(df)):
            cur_v = p_vals[-1]
            self.peak_value = max(self.peak_value, cur_v)
            if (cur_v / self.peak_value - 1) <= self.mdd_cutoff:
                self.emergency_mode = True

            sel = df["Selected"].iloc[i - 1]
            price = df["^KS11" if sel == "KOSPI" else "QQQ"].iloc[i - 1]
            sma = df["K_SMA" if sel == "KOSPI" else "Q_SMA"].iloc[i - 1]

            # Warmup & Data Quality Check
            if (
                pd.isna(sma)
                or pd.isna(df["Ret_K_2x"].iloc[i])
                or pd.isna(df["Ret_QLD_KRW"].iloc[i])
            ):
                tactical_r = 0.035 / 252
            elif self.emergency_mode or price < sma:
                tactical_r = 0.035 / 252
            else:
                tactical_r = (
                    df["Ret_K_2x"].iloc[i]
                    if sel == "KOSPI"
                    else df["Ret_QLD_KRW"].iloc[i]
                )
                if prev_s and prev_s != sel:
                    tactical_r -= self.tx_cost

            prev_s = sel

            # Core Return (Handle NaNs in early days)
            c_vals = [
                0.25 * df["^KS11"].pct_change().iloc[i],
                0.20 * df["Ret_SCHD_KRW"].iloc[i],
                0.10 * df["Ret_GLD_KRW"].iloc[i],
            ]
            core_r = sum([v if not pd.isna(v) else 0.0 for v in c_vals])

            val_change = core_r + 0.45 * tactical_r
            if pd.isna(val_change):
                val_change = 0.0

            strat_rets.append(val_change)
            p_vals.append(cur_v * (1 + val_change))

        cum_ret = p_vals[-1] - 1
        mdd = (pd.Series(p_vals) / pd.Series(p_vals).cummax() - 1).min()
        print(f"\n[Final] Return: {cum_ret:.1%}, MDD: {mdd:.1%}")


if __name__ == "__main__":
    Portfolio2026().run_simulation()
