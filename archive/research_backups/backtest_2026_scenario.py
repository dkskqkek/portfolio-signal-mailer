# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


class Portfolio2026:
    """
    2026 Era Adaptive Portfolio Strategy

    Core Philosophy:
    - Dynamic Asset Selection: KOSPI vs QQQ (6-month momentum)
    - Currency-Adjusted Returns: All assets in KRW perspective
    - Risk Management: SMA 130 trend filter + Hard MDD cutoff
    - Realistic Modeling: Dividend adjustment, leverage decay, transaction costs
    """

    def __init__(
        self,
        mdd_cutoff=-0.40,
        leverage_cost_annual=0.015,
        transaction_cost=0.002,
        kospi_dividend_yield=0.025,
    ):
        # Risk Parameters
        self.mdd_cutoff = mdd_cutoff
        self.leverage_cost_annual = leverage_cost_annual
        self.transaction_cost = transaction_cost
        self.kospi_div_yield = kospi_dividend_yield

        # Portfolio Weights
        self.W_KOSPI_CORE = 0.25  # Original user preference
        self.W_SCHD = 0.20
        self.W_GLD = 0.10
        self.W_TACTICAL = 0.45

        # Strategy Parameters
        self.MOMENTUM_WINDOW = 126  # 6 months
        self.SMA_WINDOW = 130

        # State
        self.emergency_mode = False
        self.peak_value = 1.0

    def fetch_data(self, start="2024-01-01", end=None):
        """Fetch market data with proper error handling"""
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        tickers = ["^KS11", "QQQ", "GLD", "BIL", "KRW=X", "SCHD"]

        print(f"Fetching data: {start} to {end}...")

        try:
            data = yf.download(
                tickers, start=start, end=end, progress=False, group_by="ticker"
            )

            df = pd.DataFrame()

            for ticker in tickers:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker in data.columns.levels[0]:
                            target = data[ticker]
                        else:
                            continue
                    else:
                        target = data

                    if "Adj Close" in target.columns:
                        df[ticker] = target["Adj Close"]
                    elif "Close" in target.columns:
                        df[ticker] = target["Close"]

                except Exception:
                    pass

            df = df.ffill().dropna()

            if df.empty:
                raise ValueError("No data retrieved")

            print(f"✓ Data loaded: {len(df)} days")
            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def calculate_features(self, df):
        """Calculate currencies and indicators (No Look-Ahead)"""
        df["USD_KRW"] = df["KRW=X"]
        df["QQQ_KRW"] = df["QQQ"] * df["USD_KRW"]
        df["SCHD_KRW"] = df["SCHD"] * df["USD_KRW"]
        df["GLD_KRW"] = df["GLD"] * df["USD_KRW"]

        # KOSPI TR approx
        kospi_div_daily = self.kospi_div_yield / 252
        df["Ret_KOSPI_Daily"] = df["^KS11"].pct_change() + kospi_div_daily
        df["KOSPI_TR"] = (1 + df["Ret_KOSPI_Daily"].fillna(0)).cumprod()

        # Momentum
        df["KOSPI_Mom"] = df["KOSPI_TR"].pct_change(self.MOMENTUM_WINDOW)
        df["QQQ_Mom_KRW"] = df["QQQ_KRW"].pct_change(self.MOMENTUM_WINDOW)

        # Relative Selection (Shifted to avoid Look-Ahead)
        df["Selected_Asset"] = np.where(
            df["KOSPI_Mom"].shift(1) > df["QQQ_Mom_KRW"].shift(1), "KOSPI", "QQQ"
        )

        # Trend Filter
        df["KOSPI_SMA"] = df["^KS11"].rolling(self.SMA_WINDOW).mean()
        df["QQQ_SMA"] = df["QQQ"].rolling(self.SMA_WINDOW).mean()

        return df

    def generate_signals(self, df):
        """Signals based on T-1 state"""
        signals = []
        for i in range(len(df)):
            if i == 0:
                signals.append("WAIT")
                continue

            selected = df["Selected_Asset"].iloc[i - 1]
            if selected == "KOSPI":
                price, sma = df["^KS11"].iloc[i - 1], df["KOSPI_SMA"].iloc[i - 1]
            else:
                price, sma = df["QQQ"].iloc[i - 1], df["QQQ_SMA"].iloc[i - 1]

            if pd.isna(sma):
                signals.append("WAIT")
            elif price > sma:
                signals.append("BULL")
            else:
                signals.append("BEAR")

        df["Signal"] = signals
        return df

    def calculate_returns(self, df):
        """Realistic leveraged returns (Volatility Decay + Compounded FX)"""
        lev_cost_daily = self.leverage_cost_annual / 252

        # KOSPI 2x
        k_vol = df["Ret_KOSPI_Daily"].rolling(20).std()
        k_decay = 0.5 * 2.0 * (2.0 - 1) * (k_vol**2)
        df["Ret_KOSPI_2x"] = (
            (df["Ret_KOSPI_Daily"] * 2.0) - k_decay.fillna(0) - lev_cost_daily
        )

        # QQQ 2x KRW
        q_ret = df["QQQ"].pct_change()
        q_vol = q_ret.rolling(20).std()
        q_decay = 0.5 * 2.0 * (2.0 - 1) * (q_vol**2)
        df["Ret_QLD_USD"] = (q_ret * 2.0) - q_decay.fillna(0) - lev_cost_daily

        # Compounded FX (Proper way)
        df["Ret_FX"] = df["USD_KRW"].pct_change()
        df["Ret_QLD_KRW"] = (1 + df["Ret_QLD_USD"]) * (1 + df["Ret_FX"]) - 1

        # Core
        df["Ret_SCHD_KRW"] = (1 + df["SCHD"].pct_change()) * (1 + df["Ret_FX"]) - 1
        df["Ret_GLD_KRW"] = (1 + df["GLD"].pct_change()) * (1 + df["Ret_FX"]) - 1
        return df

    def simulate_portfolio(self, df):
        strat_returns, port_values = [], [1.0]
        prev_selection = None
        r_cash = 0.035 / 252

        for i in range(1, len(df)):
            cur_v = port_values[-1]
            self.peak_value = max(self.peak_value, cur_v)
            cur_dd = (cur_v / self.peak_value) - 1

            if cur_dd <= self.mdd_cutoff:
                self.emergency_mode = True
            if self.emergency_mode and cur_dd > -0.25:
                self.emergency_mode = False

            signal, selection = (
                df["Signal"].iloc[i - 1],
                df["Selected_Asset"].iloc[i - 1],
            )
            tx_cost = (
                self.transaction_cost
                if (prev_selection and prev_selection != selection)
                else 0.0
            )
            prev_selection = selection

            if self.emergency_mode or signal in ["BEAR", "WAIT"]:
                r_tactical = r_cash
            else:
                r_tactical = (
                    df["Ret_KOSPI_2x"].iloc[i]
                    if selection == "KOSPI"
                    else df["Ret_QLD_KRW"].iloc[i]
                )
                r_tactical -= tx_cost

            r_core = (
                self.W_KOSPI_CORE * df["Ret_KOSPI_Daily"].iloc[i]
                + self.W_SCHD * df["Ret_SCHD_KRW"].iloc[i]
                + self.W_GLD * df["Ret_GLD_KRW"].iloc[i]
            )

            total_r = r_core + (self.W_TACTICAL * r_tactical)
            strat_returns.append(total_r)
            port_values.append(cur_v * (1 + total_r))

        return pd.Series(strat_returns, index=df.index[1:]), port_values[1:]

    def run(self, start="2024-06-01"):
        df = self.fetch_data(start)
        if df.empty:
            return None
        df = self.calculate_features(df)
        df = self.generate_signals(df)
        df = self.calculate_returns(df)
        returns, values = self.simulate_portfolio(df)

        # Simplified Report
        final_ret = values[-1] - 1
        mdd = (pd.Series(values) / pd.Series(values).cummax() - 1).min()
        print(f"\n[Final Results] Return: {final_ret:.1%}, MDD: {mdd:.1%}")
        return {"metrics": {"final_return": final_ret, "mdd": mdd}}


if __name__ == "__main__":
    p = Portfolio2026()
    p.run()
