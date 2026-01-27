"""
Antigravity v4.2 - Enhanced with Risk Scoring
==============================================
Base: v4.1 Logic (Daily Trigger + Monthly Selection)
Added:
1. DefensiveSelector class (clean separation)
2. QuantScore with divergence detection
3. Emergency Mode (MDD + Quant Score dual safety)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import data_loader

# ===== Defensive Universe (27종 확장 풀) =====
DEFENSIVE_UNIVERSE = [
    # Core Defensive
    "XLP",
    "XLU",
    "GLD",
    "TLT",
    "IEF",
    "SHY",
    "BIL",
    # Currencies
    "FXY",
    "UUP",
    # Commodities
    "DBC",
    # Inflation
    "SCHP",
    "TIP",
    "GSY",
    # Sectors
    "XLK",
    "XLV",
    "XLF",
    "XLE",
    "XLI",
    "XLB",
    "XLRE",
    "VNQ",
    # International
    "EEM",
    "EFA",
    "VEA",
    # Bonds
    "AGG",
    "LQD",
    "HYG",
]


@dataclass
class QuantScore:
    """Market Health Score with divergence detection"""

    total: float
    macro: float  # VIX based (0-30)
    trend: float  # MA alignment (0-40)
    efficiency: float  # Sharpe-like (0-30)

    def get_status(self) -> str:
        if self.total >= 80:
            return "EXCELLENT"
        elif self.total >= 70:
            return "HEALTHY"
        elif self.total >= 50:
            return "CAUTION"
        elif self.total >= 40:
            return "FRAGILE"
        else:
            return "DANGER"

    def get_divergence_alert(self) -> Optional[str]:
        """Detect dangerous divergences"""
        if self.trend >= 38 and self.efficiency <= 12:
            return "⚠️ LATE CYCLE: High trend but low efficiency - Risk of reversal"
        if self.macro >= 20 and self.trend <= 20:
            return "⚠️ BREAKDOWN RISK: Macro ok but trend weak"
        return None


class QuantScoreCalculator:
    """Calculate market health score"""

    def calculate(self, prices: pd.Series, vix: float) -> QuantScore:
        returns = prices.pct_change().dropna()

        # 1. Macro Score (VIX) - 0 to 30
        if vix < 12:
            macro = 30.0
        elif vix > 25:
            macro = 0.0
        else:
            macro = 30.0 * (25 - vix) / 13

        # 2. Trend Score (MA alignment) - 0 to 40
        if len(prices) < 200:
            trend = 20.0
        else:
            current = prices.iloc[-1]
            ma50 = prices.rolling(50).mean().iloc[-1]
            ma200 = prices.rolling(200).mean().iloc[-1]

            if current > ma50 > ma200:
                trend = 35.0 + min(
                    5.0, ((current / ma50 - 1) + (current / ma200 - 1)) * 50
                )
            elif current > ma50:
                trend = 30.0
            elif ma50 > ma200:
                trend = 10.0
            else:
                trend = 0.0

        # 3. Efficiency Score (Sharpe-like) - 0 to 30
        if len(returns) < 20:
            efficiency = 15.0
        else:
            recent = returns.iloc[-20:]
            mean_ret = recent.mean() * 252
            vol = recent.std() * np.sqrt(252)
            sharpe = mean_ret / vol if vol > 0 else 0
            efficiency = min(30.0, max(0.0, 15.0 * sharpe))

        return QuantScore(
            total=macro + trend + efficiency,
            macro=macro,
            trend=trend,
            efficiency=efficiency,
        )


class DefensiveSelector:
    """Select Top-3 defensive assets based on momentum"""

    def __init__(self, universe: List[str] = DEFENSIVE_UNIVERSE):
        self.universe = universe
        self.selection_history = []

    def select_top3(
        self, prices: pd.DataFrame, current_date: pd.Timestamp
    ) -> List[str]:
        """Select Top 3 based on 8-month momentum"""

        # Monthly resampling
        monthly = prices.resample("M").last()
        if len(monthly) < 9:
            return ["BIL"]

        mom = monthly.pct_change(8).iloc[-1]

        # Filter to universe and positive momentum
        available = [t for t in self.universe if t in mom.index]
        positive = mom[available].dropna()
        positive = positive[positive > 0].sort_values(ascending=False)

        if len(positive) >= 3:
            selected = positive.head(3).index.tolist()
        elif len(positive) > 0:
            selected = positive.index.tolist()
            while len(selected) < 3:
                selected.append("BIL")
        else:
            selected = ["BIL", "BIL", "BIL"]

        # Log
        self.selection_history.append(
            {
                "date": current_date,
                "assets": selected,
                "mom": [mom.get(s, 0) for s in selected],
            }
        )

        return selected


class EmergencyMode:
    """MDD + Quant Score dual safety system"""

    def __init__(self, mdd_cutoff: float = -0.40, score_cutoff: float = 40.0):
        self.mdd_cutoff = mdd_cutoff
        self.score_cutoff = score_cutoff
        self.is_active = False
        self.peak_value = 1.0
        self.trigger_reason = None

    def check(self, current_value: float, quant_score: QuantScore) -> bool:
        """Check if emergency mode should be triggered"""

        self.peak_value = max(self.peak_value, current_value)
        current_dd = (current_value / self.peak_value) - 1

        # Trigger conditions
        if current_dd <= self.mdd_cutoff:
            if not self.is_active:
                self.trigger_reason = f"MDD {current_dd * 100:.1f}%"
            self.is_active = True
        elif quant_score.total < self.score_cutoff:
            if not self.is_active:
                self.trigger_reason = f"QuantScore {quant_score.total:.0f}"
            self.is_active = True

        # Recovery conditions
        if self.is_active and current_dd > -0.25 and quant_score.total > 50:
            self.is_active = False
            self.trigger_reason = None

        return self.is_active


def run_backtest_v4_2():
    print("=" * 70)
    print(" Antigravity v4.2: Enhanced Risk Management")
    print("=" * 70)

    # Setup
    START_DATE = "2006-01-01"
    tickers = ["SPY", "QLD", "GLD", "EWY", "QQQ", "KRW=X", "^VIX"] + DEFENSIVE_UNIVERSE
    tickers = list(set(tickers))

    try:
        df = data_loader.fetch_validated_data(tickers, start_date=START_DATE)
    except Exception as e:
        print(f"Data Error: {e}")
        return

    close = df.xs("Close", level=1, axis=1)
    open_p = df.xs("Open", level=1, axis=1)

    # KRW
    krw = (
        close["KRW=X"].fillna(method="ffill").fillna(1200.0)
        if "KRW=X" in close.columns
        else pd.Series(1200.0, index=close.index)
    )

    # VIX
    vix = (
        close["^VIX"].fillna(method="ffill").fillna(16.0)
        if "^VIX" in close.columns
        else pd.Series(16.0, index=close.index)
    )

    # Signal indicators
    sma110 = close["QQQ"].rolling(110).mean()
    sma250 = close["QQQ"].rolling(250).mean()

    # Components
    selector = DefensiveSelector(DEFENSIVE_UNIVERSE)
    scorer = QuantScoreCalculator()
    emergency = EmergencyMode(mdd_cutoff=-0.40, score_cutoff=40.0)

    # State
    start_idx = 260
    trade_dates = close.index
    cash = 100_000.0
    shares = {t: 0.0 for t in tickers}
    prev_signal = "DANGER"
    selected_def = ["BIL"]
    history = []
    emergency_events = []

    print(f"Starting: {trade_dates[start_idx].date()}")

    for i in range(start_idx, len(trade_dates) - 1):
        today = trade_dates[i]
        next_day = trade_dates[i + 1]
        is_month_end = today.month != next_day.month

        # 1. Calculate Quant Score
        qqq_window = close["QQQ"].iloc[: i + 1]
        current_vix = vix.iloc[i]
        quant_score = scorer.calculate(qqq_window, current_vix)

        # 2. Monthly Selection
        if is_month_end:
            def_prices = close[DEFENSIVE_UNIVERSE].iloc[: i + 1]
            selected_def = selector.select_top3(def_prices, today)

        # 3. Daily Signal
        q = close.loc[today, "QQQ"]
        s1, s2 = sma110.loc[today], sma250.loc[today]

        if pd.isna(s1) or pd.isna(s2):
            signal = "DANGER"
        elif q > s1 and q > s2:
            signal = "NORMAL"
        else:
            signal = "DANGER"

        # 4. Current Equity
        prices_exec = open_p.loc[next_day]
        equity = cash
        for t, cnt in shares.items():
            if cnt > 0:
                p = (
                    prices_exec.get(t)
                    if t in prices_exec.index
                    else close.loc[today, t]
                    if t in close.columns
                    else 0
                )
                if pd.notna(p):
                    equity += cnt * p

        # 5. Emergency Mode Check
        is_emergency = emergency.check(equity / 100_000, quant_score)

        if (
            is_emergency
            and emergency.trigger_reason
            and len(emergency_events) == 0
            or (len(emergency_events) > 0 and emergency_events[-1]["end"] is not None)
        ):
            emergency_events.append(
                {"start": today, "reason": emergency.trigger_reason, "end": None}
            )
            print(f"🚨 EMERGENCY @ {today.date()}: {emergency.trigger_reason}")

        if (
            not is_emergency
            and len(emergency_events) > 0
            and emergency_events[-1]["end"] is None
        ):
            emergency_events[-1]["end"] = today
            print(f"✅ RECOVERY @ {today.date()}")

        # 6. Target Weights
        targets = {"SPY": 0.20, "EWY": 0.20, "GLD": 0.15}

        if is_emergency:
            # Full defensive: BIL only
            targets["BIL"] = 0.45
        elif signal == "NORMAL":
            # Adjust size based on Quant Score
            if quant_score.total >= 70:
                multiplier = 1.0
            elif quant_score.total >= 50:
                multiplier = 0.7
            else:
                multiplier = 0.5

            targets["QLD"] = 0.45 * multiplier
            targets["BIL"] = 0.45 * (1 - multiplier)
        else:
            # Defensive mode
            w_each = 0.45 / len(selected_def)
            for d in selected_def:
                targets[d] = targets.get(d, 0) + w_each

        # 7. Rebalance
        needs_trade = (signal != prev_signal) or is_month_end

        if needs_trade:
            # Sell all, buy targets
            cash = equity * (1 - 0.0015)
            shares = {t: 0.0 for t in tickers}

            for t, w in targets.items():
                p = prices_exec.get(t) if t in prices_exec.index else None
                if pd.isna(p):
                    p = close.loc[today, t] if t in close.columns else None
                if pd.notna(p) and p > 0:
                    shares[t] = (cash * w) / p
            cash = 0

        prev_signal = signal

        # 8. Record
        prices_close = close.loc[next_day]
        val_usd = cash
        for t, cnt in shares.items():
            if cnt > 0:
                p = prices_close.get(t) if t in prices_close.index else 0
                if pd.notna(p):
                    val_usd += cnt * p

        val_krw = val_usd * krw.asof(next_day)
        history.append(
            {
                "Date": next_day,
                "PV_KRW": val_krw,
                "Signal": signal,
                "QuantScore": quant_score.total,
                "Emergency": is_emergency,
            }
        )

    # Report
    res = pd.DataFrame(history).set_index("Date")
    res.to_csv("research/backtest_v4_2_results.csv")

    start_v, end_v = res["PV_KRW"].iloc[0], res["PV_KRW"].iloc[-1]
    days = (res.index[-1] - res.index[0]).days
    cagr = (end_v / start_v) ** (365.25 / days) - 1
    mdd = ((res["PV_KRW"] - res["PV_KRW"].cummax()) / res["PV_KRW"].cummax()).min()
    sharpe = (
        res["PV_KRW"].pct_change().mean() / res["PV_KRW"].pct_change().std()
    ) * np.sqrt(252)

    print("\n" + "=" * 70)
    print(" FINAL REPORT")
    print("=" * 70)
    print(f"Period: {res.index[0].date()} ~ {res.index[-1].date()}")
    print(f"\nCAGR (KRW): {cagr * 100:.2f}%")
    print(f"MDD: {mdd * 100:.2f}%")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Final Value: {end_v / 1e8:.2f}억")

    print(f"\nEmergency Events: {len(emergency_events)}")
    for e in emergency_events:
        end_str = e["end"].date() if e["end"] else "Ongoing"
        print(f"  {e['start'].date()} ~ {end_str}: {e['reason']}")

    # Defensive Selection Summary
    print("\nTop Defensive Selections:")
    all_assets = []
    for h in selector.selection_history:
        all_assets.extend(h["assets"])
    freq = pd.Series(all_assets).value_counts().head(10)
    print(freq.to_string())


if __name__ == "__main__":
    run_backtest_v4_2()
