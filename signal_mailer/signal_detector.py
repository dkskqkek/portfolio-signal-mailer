# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings

# Filter warnings before importing specific modules if they generate noise
warnings.filterwarnings("ignore")

import json
from typing import List, Dict, Optional, Any
from signal_mailer.debate_council import DebateCouncil
from signal_mailer.index_sniper import IndexSniper
# from html_generator import generate_html_report # We will update this later


class SignalDetector:
    """
    [Antigravity v4.1 Live Engine]
    1. Daily Trigger: Check QQQ vs SMA(110, 250). Capture Trend.
    2. Monthly Selection: If DANGER, select Top 3 Defensive Assets (8-month Momentum) as of PREVIOUS MONTH END.
    3. Qualitative Check: The 'Debate Council' (LLM) provides a discount factor on risk.
    4. Index Sniper: Weekly Swing Trading Signals (V8.2)
    """

    @staticmethod
    def _to_py_type(val: Any) -> Any:
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        if isinstance(val, (np.floating, float)):
            return float(val) if not np.isnan(val) else 0.0
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(val, "__dict__"):  # Handle Dataclasses
            return {k: SignalDetector._to_py_type(v) for k, v in val.__dict__.items()}
        return val

    def __init__(self, api_key: Optional[str] = None):
        self.council = DebateCouncil(api_key) if api_key else None
        self.sniper = IndexSniper()  # Initialize Sniper V8.2

        # Defensive Pool (v4.1 standard)
        self.def_pool = [
            "XLP",
            "XLU",
            "GLD",
            "FXY",
            "UUP",
            "DBC",
            "TLT",
            "IEF",
            "BIL",
            "SHY",
            "SCHP",
            "GSY",  # Expanded slightly for robustness
        ]

    def fetch_data(self, days_back: int = 400) -> Optional[pd.DataFrame]:
        # Fetch enough data for SMA250 and 8-month momentum
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        tickers = [
            "SPY",
            "QQQ",
            "^KS200",
            "^VIX",
            "GLD",
            "BIL",
            "KRW=X",
        ] + self.def_pool
        tickers = list(set(tickers))

        try:
            print(f"Fetching data for {len(tickers)} tickers...")
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                group_by="ticker",
                auto_adjust=True,
            )

            if data.empty:
                return None

            # Flatten to simple dict of series for easier handling if needed, but multi-index is fine.
            # We want 'Close' for signals
            # Handle yfinance multi-level columns

            # Extract Adjusted Close (auto_adjust=True makes 'Close' adjusted)
            # Check structure
            if isinstance(data.columns, pd.MultiIndex):
                close = data.xs("Close", level=1, axis=1)
            else:
                # Single ticker case? (unlikely with list)
                close = data["Close"]

            return close.ffill()

        except Exception as e:
            print(f"[Data Error] {e}")
            return None

    def get_monthly_selection(self, close_data: pd.DataFrame) -> List[str]:
        """
        Determines the Defensive Asset to hold based on LAST MONTH END's momentum.
        Returns: list of tickers (Top 3)
        """
        # Resample to Month End
        monthly = close_data.resample("M").last()

        # Calculate 8-month momentum (approx 8*21 days? No, simple pct_change(8))
        # Note: We need enough history. If not enough, fallback.
        if len(monthly) < 9:
            return ["BIL"]

        mom = monthly.pct_change(8)

        # Get the LAST COMPLETED month.
        # If today is Jan 25, we want momentum as of Dec 31.
        # If today is Jan 31 (and market closed?), we might want Jan 31?
        # Safe logic: Look at the last row of 'monthly' that is BEFORE today.

        current_date = pd.Timestamp(datetime.now().date())
        # Filter monthly index strictly less than today (to simulate 'previous month close' usage)
        # Actually, if we are on Feb 1st, we use Jan 31 data.
        # If we are on Jan 31st (during day), we still use Dec 31 data until close?
        # Yes, monthly selection rotates on Month Start (Day 1).

        valid_months = mom.loc[mom.index < current_date]
        if valid_months.empty:
            return ["BIL"]

        last_month_mom = valid_months.iloc[-1]

        # Filter Pool
        candidates = last_month_mom[self.def_pool]
        # Rank
        candidates = candidates.sort_values(ascending=False)
        # Positive only?
        candidates = candidates[candidates > 0]

        top3 = candidates.head(3).index.tolist()
        if not top3:
            return ["BIL"]  # Safety

        return top3

    def detect(self, verbose: bool = True) -> Dict[str, Any]:
        print(">>> Running Signal Detection (v4.1 Live)...")

        # 1. Fetch
        close_data = self.fetch_data(days_back=400)
        if close_data is None:
            return {"error": "Data fetch failed"}

        # 2. Daily Signal (QQQ vs SMA)
        qqq = close_data["QQQ"]
        current_price = qqq.iloc[-1]

        sma110 = qqq.rolling(110).mean().iloc[-1]
        sma250 = qqq.rolling(250).mean().iloc[-1]

        signal = "DANGER"
        if current_price > sma110 and current_price > sma250:
            signal = "NORMAL"

        # 3. Monthly Selection
        defensive_assets = self.get_monthly_selection(close_data)

        # 4. Council Debate (If DANGER)
        # Should we ask Council even in Normal? Yes, for "Discount Factor" on leverage.
        # But expensive. Let's ask only if VIX > 18 or specific condition.
        # For v4.1, let's always ask to show "Intellectual Partner" capability.

        vix_val = close_data["^VIX"].iloc[-1]

        # 5. Index Sniper Scan (Weekly Logic on Daily Data - Approx)
        # Note: Index Sniper is optimized for Weekly. Passing Daily data works but
        # we should ideally resample or use daily params if we want daily precision.
        # However, key features (VIX Fix) work on daily too.
        # Let's run it on QQQ (core asset).

        # Prepare QQQ dataframe for Sniper (needs OHLV usually, but we have Close)
        # We need Open/High/Low for full Sniper precision.
        # Current fetch_data ONLY returns Close series.
        # We need to upgrade fetch_data or do a separate fetch for Sniper.
        # Let's do a quick separate fetch for QQQ OHLV to be accurate.

        sniper_result = None
        try:
            # Fetch detailed QQQ data for Sniper
            # We use 2y daily data to approximate weekly signals or just run daily sniper
            # For V8.2 (Weekly Optimized), we should fetch weekly data
            print("Fetching QQQ weekly data for Sniper...")
            qqq_weekly = yf.download(
                "QQQ",
                period="2y",
                interval="1wk",
                progress=False,
                group_by="ticker",
                auto_adjust=True,
            )

            if qqq_weekly.empty:
                print("QQQ weekly data empty")
            else:
                # Flatten logic similar to index_sniper.py
                df = qqq_weekly.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    if "QQQ" in df.columns.levels[0]:
                        df = df["QQQ"]
                    elif "Close" in df.columns.levels[0]:
                        pass  # Already flat-ish

                # print(f"DEBUG: QQQ Cols: {df.columns}")
                sniper_result = self.sniper.analyze(df, "QQQ")
                print(
                    f"Sniper Result: {sniper_result.current_state if sniper_result else 'None'}"
                )

        except Exception as e:
            print(f"Sniper analysis failed: {e}")
            import traceback

            traceback.print_exc()

        market_ctx = {
            "QQQ_Price": round(current_price, 2),
            "SMA_110": round(sma110, 2),
            "SMA_250": round(sma250, 2),
            "VIX": round(vix_val, 2),
            "Signal_Tech": signal,
            "Sniper_State": sniper_result.current_state if sniper_result else "N/A",
        }

        news_sample = [
            "Market awaiting Fed decision",
            "Tech earnings mixed",
            "Geopolitical tension remains",
        ]  # Placeholder. Ideally fetch real news.

        council_verdict = "Council not convened."
        discount = 1.0

        if self.council:
            try:
                discount, council_verdict = self.council.convene_council(
                    market_ctx, news_sample
                )
            except Exception as e:
                print(f"Council skipped: {e}")

        # 6. Construct Report Data
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "signal": signal,
            "qqq_price": current_price,
            "dist_sma110": (current_price / sma110) - 1,
            "dist_sma250": (current_price / sma250) - 1,
            "vix": vix_val,
            "defensive_selection": defensive_assets,
            "council_discount": discount,
            "council_verdict": council_verdict,
            "sniper_signal": sniper_result,  # Add Sniper Result Object
            "action_plan": self._generate_action_plan(
                signal, defensive_assets, discount, sniper_result
            ),
        }

        if verbose:
            print(json.dumps(report, indent=2, default=self._to_py_type))

        return report

    def _generate_action_plan(
        self,
        signal: str,
        def_assets: List[str],
        discount: float,
        sniper: Optional[Any] = None,
    ) -> str:
        """
        Synthesizes the specific executable instruction.
        """
        plan = ""

        # Core MA Signal
        if signal == "NORMAL":
            alloc = int(45 * discount)  # Apply Council discount
            plan += f"🟢 **BUY/HOLD QLD** (Target: {alloc}%). [Council Modified: {discount}]\n"
        else:
            assets_str = ", ".join(def_assets)
            plan += f"🔴 **DEFENSIVE MODE**: Sell QLD. Buy **{assets_str}** (Equal Weight).\n"

        # Add Sniper Insight
        if sniper:
            if sniper.is_buy:
                plan += "\n🎯 **SNIPER BUY SIGNAL** detected on QQQ! Confirm Weekly Momentum."
            elif sniper.is_sell:
                plan += "\n⚠️ **SNIPER SELL SIGNAL** detected on QQQ! Consider trimming exposure."
            elif sniper.buy_window:
                plan += "\n🟠 In **Sniper Buy Window**. Watch for momentum shift."

        return plan


if __name__ == "__main__":
    detector = SignalDetector()  # No key in local test if env var set or mock
    detector.detect()
