# -*- coding: utf-8 -*-
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TradeOrder:
    account: str
    ticker: str
    action: str  # BUY / SELL
    amount_krw: float
    reason: str


class PortfolioManager:
    """
    Manages real-world portfolio state and calculates rebalancing orders
    based on system signals.
    """

    def __init__(self, state_file: str = "data/portfolio_state.json"):
        # Resolve absolute path relative to project root usually
        # But here we assume running from root or passed correct path
        self.state_file = state_file
        self.portfolio = self._load_portfolio()

    def _load_portfolio(self) -> Dict:
        if not os.path.exists(self.state_file):
            print(f"Portfolio file not found: {self.state_file}")
            return {}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading portfolio: {e}")
            return {}

    def get_summary(self) -> Dict:
        """Calculate current allocation by class"""
        if not self.portfolio:
            return {}

        total_value = 0
        alloc = {"Growth": 0, "Defensive": 0, "Gold": 0, "Cash": 0}

        for acct in self.portfolio.get("accounts", []):
            # Cash in account
            cash = acct.get("cash", 0)
            alloc["Cash"] += cash
            total_value += cash

            # Assets
            for asset in acct.get("assets", []):
                val = asset.get("value", 0)
                cls = asset.get("class", "Defensive").capitalize()

                if cls in alloc:
                    alloc[cls] += val
                else:
                    # Fallback or new class
                    alloc[cls] = alloc.get(cls, 0) + val

                total_value += val

        # Calculte %
        percentages = {
            k: (v / total_value * 100) if total_value > 0 else 0
            for k, v in alloc.items()
        }

        return {
            "total_krw": total_value,
            "allocation_krw": alloc,
            "allocation_pct": percentages,
        }

    def calculate_rebalancing(
        self, target_weights: Dict[str, float]
    ) -> List[TradeOrder]:
        """
        Generate orders to match target weights.
        target_weights example: {"Growth": 0.50, "Defensive": 0.40, "Gold": 0.10, "Cash": 0.0}
        """
        summary = self.get_summary()
        if not summary:
            return []

        total_val = summary["total_krw"]
        current_alloc = summary["allocation_krw"]

        orders = []

        # Calculate Delta per Class
        class_deltas = {}  # Positive = Buy, Negative = Sell
        for cls, target_pct in target_weights.items():
            target_val = total_val * target_pct
            current_val = current_alloc.get(cls, 0)
            delta = target_val - current_val
            class_deltas[cls] = delta

        # Strategy:
        # 1. We distribute the Delta across accounts based on "preferred assets".
        # 2. Simplification: We assume we fill the bucket in ISA first (Tax free), then Pension, then Toss?
        #    Or keep it simple: Adjust the largest existing holding of that class in each account proportionally?
        #    Let's go with "Adjust Main Tickers" strategy.

        # Defining "Main Tickers" for liquidity per account/class
        # (This is hardcoded logic for now based on user's portfolio structure)
        mapping = {
            "ISA (Tax-Advantaged)": {
                "Growth": "KODEX US Nasdaq100",
                "Defensive": "KODEX US S&P500",
            },
            "Pension (Retirement)": {
                "Growth": "KODEX US Nasdaq100",
                "Defensive": "TIGER US Div+7%",
            },
            "Toss (Direct/Overseas)": {
                "Growth": "VGT (Tech ETF)",  # Active adjustment here
                "Defensive": "SCHD or COWZ",  # SCHD not in list, use COWZ
            },
        }

        # Iterate classes and generate orders
        # Priority: Sell first to generate cash, then Buy?
        # Output is just a list of instructions.

        for cls, delta in class_deltas.items():
            if abs(delta) < 100000:  # Ignore Changes < 100k KRW (noise)
                continue

            action = "BUY" if delta > 0 else "SELL"
            abs_delta = abs(delta)

            # Distribute this delta across accounts
            # Simple Logic: Distribute proportionally to account size?
            # Or Manual Rule:
            #  - Growth buys: Toss (VGT) > ISA > Pension (Aggressive in Taxable for deferral? No, taxable is bad for rebalancing. ISA is best.)
            #  - Let's split evenly for now to avoid complexity.

            # Find accounts that HAVE this class asset (for Sell) or CAN buy (for Buy)
            target_accounts = self.portfolio.get("accounts", [])
            split_count = len(target_accounts)
            amount_per_acct = abs_delta / split_count

            for acct in target_accounts:
                acct_name = acct["name"]
                target_ticker = "Unknown"

                # Find ticker in mapping
                # Simple heuristic mapping based on what they already have
                candidates = [a for a in acct["assets"] if a["class"] == cls]
                if candidates:
                    # Pick largest holding
                    candidates.sort(key=lambda x: x["value"], reverse=True)
                    target_ticker = candidates[0]["ticker"]
                else:
                    # No asset of this class? Use default mapping if exists, or skip
                    # Try fuzzy match from mapping
                    for map_acct_key, map_dict in mapping.items():
                        if map_acct_key in acct_name:  # Partial match
                            target_ticker = map_dict.get(cls, "Generic ETF")

                # If we still don't have a ticker and it's a BUY, we might need new asset.
                # If SELL and we don't have it, we can't sell.

                if action == "SELL" and target_ticker == "Unknown":
                    continue  # Cannot sell what we don't have

                if target_ticker == "Unknown":
                    # Maybe Cash/Gold
                    if cls == "Cash":
                        continue  # Cash handled implicitly
                    target_ticker = f"New {cls} ETF"

                orders.append(
                    TradeOrder(
                        account=acct_name,
                        ticker=target_ticker,
                        action=action,
                        amount_krw=amount_per_acct,
                        reason=f"Rebalance {cls} to {target_weights.get(cls, 0) * 100:.0f}%",
                    )
                )

        return orders

    def get_actionable_report(self, orders: List[TradeOrder]) -> str:
        if not orders:
            return "✅ Portfolio is balanced."

        report = "📋 **Rebalancing Orders**\n"
        for o in orders:
            amt_str = f"{int(o.amount_krw):,}"
            icon = "🔴" if o.action == "SELL" else "🔵"
            report += f"{icon} **{o.action}** {o.ticker} ({o.amount_krw / 10000:.0f}만원) @ {o.account}\n"

        return report


if __name__ == "__main__":
    pm = PortfolioManager(state_file="d:/gg/data/portfolio_state.json")
    # Test Scenario: Bull Market (Growth 50%, Def 40%, Gold 10%)
    target = {"Growth": 0.50, "Defensive": 0.40, "Gold": 0.10}

    print("\n--- Current Status ---")
    s = pm.get_summary()
    print(f"Total: {s['total_krw']:,} KRW")
    print(f"Alloc: {s['allocation_pct']}")

    print("\n--- Simulation: Bull Market Rebalance ---")
    orders = pm.calculate_rebalancing(target)
    print(pm.get_actionable_report(orders))
