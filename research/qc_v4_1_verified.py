# QuantConnect Algorithm - Antigravity V4.1 Verified
# ==================================================
# Copy this entire code into a new project on QuantConnect.com
# This serves as an independent "Source of Truth" to verify local Python results.

from AlgorithmImports import *


class AntigravityV4_1(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2008, 1, 1)  # Crisis Test
        self.SetEndDate(datetime.now())
        self.SetCash(100000)

        # 1. Assets
        # QQQ: Tech ETF
        # QLD: 2x Tech (Benchmark for Bull)
        # Defensive: GLD (Gold), TLT (Bonds), BIL (Cash-like)
        self.tickers = ["QQQ", "QLD", "GLD", "TLT", "BIL"]
        self.symbols = {}

        for t in self.tickers:
            self.symbols[t] = self.AddEquity(t, Resolution.Daily).Symbol

        # 2. Indicators (QQQ)
        self.qqq = self.symbols["QQQ"]
        self.sma110 = self.SMA(self.qqq, 110, Resolution.Daily)
        self.sma250 = self.SMA(self.qqq, 250, Resolution.Daily)

        # 3. Scheduled Events (Daily Logic)
        self.Schedule.On(
            self.DateRules.EveryDay("QQQ"),
            self.TimeRules.AfterMarketOpen("QQQ", 30),
            self.Rebalance,
        )

        # 4. State
        self.current_month = -1
        self.selected_defensive = self.symbols["BIL"]  # Default safe

    def Rebalance(self):
        if self.IsWarmingUp:
            return

        # A. Trend Check
        # Price > 110 AND Price > 250
        price = self.Securities[self.qqq].Price

        if not (self.sma110.IsReady and self.sma250.IsReady):
            return

        is_bull = (price > self.sma110.Current.Value) and (
            price > self.sma250.Current.Value
        )

        # B. Monthly Selection (Run only on month change)
        if self.Time.month != self.current_month:
            self.current_month = self.Time.month
            self.selected_defensive = self.SelectDefensiveAsset()
            # self.Debug(f"{self.Time.date()} New Month! Selected Defensive: {self.selected_defensive}")

        # C. Trade Execution
        if is_bull:
            # Bull Mode: 100% QLD
            self.SetHoldings("QLD", 1.0)
        else:
            # Defensive Mode: 100% Selected Defensive Asset
            # Liquidate Growth first
            # self.Liquidate("QLD") # SetHoldings handles this if 100%
            self.SetHoldings(self.selected_defensive, 1.0)

    def SelectDefensiveAsset(self):
        """
        Calculate 8-month momentum for [GLD, TLT, BIL]
        Returns the Symbol of the winner.
        """
        candidates = ["GLD", "TLT", "BIL"]
        best_mom = -999.0
        best_sym = self.symbols["BIL"]

        for ticker in candidates:
            sym = self.symbols[ticker]

            # Request history: 8 months (approx 21*8 = 168 days)
            # We need closing price 168 trading days ago vs now
            hist = self.History(sym, 170, Resolution.Daily)

            if hist.empty or "close" not in hist.columns:
                continue

            closes = hist["close"]
            if len(closes) < 165:  # Safety buffer
                continue

            # Current price (yesterday close in backtest context usually, or current)
            # History returns up to yesterday mainly.
            p_now = closes[-1]
            p_past = closes[0]  # Approx 8 months ago

            mom = (p_now - p_past) / p_past

            if mom > best_mom:
                best_mom = mom
                best_sym = sym

        return best_sym
