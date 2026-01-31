# Antigravity V4 Strategy Core Logic

> [!important] **Antigravity V4 Strategy (2026-01-31 Finalized)**
> **"Safety First, Efficiency Second."**
> - The strategy prioritizes **Defensive Ratio (MDD)** over pure CAGR.
> - It uses a **Hybrid Filter** (Trend + Macro) to determine allocation.

## 1. Core Logic Overview

The portfolio allocation is determined by two primary signals:
1.  **Trend Signal:** `MA185` with `3% Buffer`
2.  **Macro Signal:** `Yield Curve (10Y - 3M)`

### Allocation Table (Matrix)

| Market State        | Trend Condition (VTI) | Macro Condition (Yield Gap) | **Stock Allocation** | **Cash Allocation** | Reason                                                                                |
| :------------------ | :-------------------- | :-------------------------- | :------------------- | :------------------ | :------------------------------------------------------------------------------------ |
| **🚀 Bull**          | Price > MA185 + 3%    | Any                         | **100% (VTI)**       | 0%                  | Ride the trend fully.                                                                 |
| **🐻 Bear (Normal)** | Price < MA185 - 3%    | Normal (> 0)                | **30% (VTI)**        | **70% (Cash)**      | Sortino Optimized. Maintain minimal exposure (30%) for tax deferral & sharp rebound.  |
| **💀 Bear (Crisis)** | Price < MA185 - 3%    | **Inverted (< 0)**          | **0%**               | **100% (Cash)**     | Safety Valve. Deep recessions (2000, 2008) often coincide with inverted yield curves. |

---

## 2. Detailed Parameters

### A. Trend Filter: MA 185 + 3% Buffer
*   **Why 185?** Backtesting (1993-2025) showed `MA185` had the best balance of reaction speed and noise filtering compared to `MA200`.
*   **Why 3% Buffer?**
    *   **Buy Trigger:** Price crosses **above** `MA * 1.03`
    *   **Sell Trigger:** Price crosses **below** `MA * 0.97`
    *   **Effect:** Reduces false signals (Whipsaws) drastically (from ~90 trades to ~15 trades over 30 years).
    *   **Synergy:** MA185 is sensitive, Buffer adds stability. (MA300 is too slow, Buffer makes it slower).

### B. Macro Filter: Yield Curve (10Y - 3M)
*   **Indicator:** `^TNX` (10 Year Treasury Yield) - `^IRX` (13 Week Treasury Yield)
*   **Logic:**
    *   If `Gap < 0` (Inverted): **High Probability of Recession**.
    *   In Bear Market, this signal forces **Total Exit (0% Stock)**.
*   **Performance:**
    *   CAGR Impact: +0.1% (Marginal)
    *   MDD Impact: ~1% (Reduces deep drawdowns)
    *   **Role:** Acts as a "Safety Valve" for black swan events.

### C. Bear Market Allocation: 30% vs 70%
*   **Standard Rule:** In a normal correction (Yield Curve Normal), keep **30% Stock**.
*   **Why 30%?**
    *   **Tax Deferral:** Selling 100% incurs immediate 22% tax on gains. Holding 30% defers this tax.
    *   **Sortino Ratio:** Our optimization showed that a **70% Cash / 30% Stock** ratio maximizes the Sortino Ratio (Downside-adjusted return).
    *   **FOMO Protection:** If the market rebounds suddenly, the 30% position captures initial upside.

---

## 3. Daily Workflow (Automated)

1.  **Run Check:** Every morning (or via Scheduler), run `scripts/daily_signal_discord.py`.
2.  **Fetch Data:** Get latest `VTI`, `^TNX`, `^IRX` from Yahoo Finance.
3.  **Compute:** Calculate `MA185`, `Buffer`, `Yield Gap`, `Regime`.
4.  **Notify:** Send Discord message with:
    *   Current Regime (Bull / Bear / Crisis)
    *   Target Allocation (e.g., "Stock 30% / Cash 70%")
    *   Buffer Distance ("2.5% to Sell Signal")
