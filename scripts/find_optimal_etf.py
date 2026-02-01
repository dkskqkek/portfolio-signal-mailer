"""
Analysis: Antigravity V4 ETF Tournament
Author: Antigravity
Date: 2026-02-01

Goal: Find the BEST ETF to replace VTI under V4 Strategy conditions.
Strategy: MA185 + 3% Buffer + Yield Logic (Bear=30%, Crisis=0%)

Candidates:
1. Broad: VTI(Base), SPY, IWM, RSP, DIA
2. Growth: QQQ, SCHG, XLK
3. Leverage: QLD (2x QQQ), SSO (2x SPY)
   * Note: Leverage ETFs have shorter history (2006~), but cover 2008 crisis.
"""

import yfinance as yf
import pandas as pd
import numpy as np


def run_etf_tournament():
    print("🚀 Starting Antigravity V4 ETF Tournament...")

    # 1. Candidates
    tickers = {
        "VTI": "Base (Total US)",
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "SCHG": "Large Growth",
        "XLK": "Tech Select",
        "RSP": "S&P 500 Equal",
        "IWM": "Russell 2000",
        "DIA": "Dow Jones",
        "QLD": "ProShares Ultra QQQ (2x)",
        "SSO": "ProShares Ultra S&P (2x)",
    }

    macro_tickers = ["^TNX", "^IRX"]
    all_tickers = list(tickers.keys()) + macro_tickers

    start_date = "1993-01-01"  # Start as early as possible
    end_date = "2025-12-31"

    print(f"📥 Downloading data for {len(tickers)} ETFs + Macro...")
    try:
        data = yf.download(
            all_tickers,
            start=start_date,
            end=end_date,
            progress=False,
            group_by="ticker",
        )
    except Exception as e:
        print(f"Error downloading: {e}")
        return

    # Prepare Shared Macro Data
    macro_df = pd.DataFrame()
    macro_df["TNX"] = data["^TNX"]["Close"]
    macro_df["IRX"] = data["^IRX"]["Close"]
    macro_df["Spread"] = macro_df["TNX"] - macro_df["IRX"]
    macro_df = macro_df.ffill().dropna()  # Yield curve is essential

    results = []

    for ticker, desc in tickers.items():
        print(f"⚔️ Testing {ticker} ({desc})...")

        # Get Price Data
        try:
            df = pd.DataFrame()
            if ticker in data.columns.levels[0]:  # Check if ticker exists at top level
                df["Close"] = data[ticker]["Close"]
            else:
                print(f"  ⚠️ No data for {ticker}")
                continue
        except KeyError:  # Fallback for single level if download structure differs
            # Sometimes yfinance returns flat cols if only 1 ticker, but we requested list.
            # With group_by='ticker', accessing data[ticker] should work.
            if ticker in data:
                df["Close"] = data[ticker]["Close"]
            else:
                print(f"  ⚠️ Key error for {ticker}")
                continue

        df = df.dropna()
        if df.empty:
            continue

        # Align with Macro
        # Since macro is 1993~, and ETF might be 2006~, we join.
        df = df.join(macro_df["Spread"], how="inner").dropna()

        # Start Date Check
        first_date = df.index[0].strftime("%Y-%m-%d")

        # Strategy Logic: MA185 + 3% Buffer
        df["MA185"] = df["Close"].rolling(window=185).mean()
        buffer = 0.03
        df["Upper"] = df["MA185"] * (1 + buffer)
        df["Lower"] = df["MA185"] * (1 - buffer)

        df = df.dropna()  # Drop initial MA NaN
        if df.empty:
            continue

        prices = df["Close"].values
        uppers = df["Upper"].values
        lowers = df["Lower"].values
        spreads = df["Spread"].values

        states = np.zeros(len(df))
        curr = 1 if prices[0] > uppers[0] else -1

        for i in range(len(df)):
            if prices[i] > uppers[i]:
                curr = 1
            elif prices[i] < lowers[i]:
                curr = -1
            states[i] = curr

        # Allocation Logic
        # Bull: 100%
        # Bear (Normal): 30% Stock (Assume 70% Cash has 0% return for simplicity/safety margin)
        # Bear (Crisis/Inverted): 0% Stock

        weights = np.zeros(len(df))
        buy_hold_weights = np.ones(len(df))  # Benchmark

        for i in range(len(df)):
            if states[i] == 1:  # Bull
                weights[i] = 1.0
            else:  # Bear
                if spreads[i] < 0:  # Crisis
                    weights[i] = 0.0
                else:  # Normal Bear
                    weights[i] = 0.3  # 30% Stock

        # Calculate Returns
        # Note: We simulate "Price Return" mainly. Dividends are included if 'Adj Close' used?
        # yfinance 'Close' is usually Adjusted if auto_adjust=True default? No, manual download usually Close is Raw, Adj Close is Adj.
        # Let's check yf.download default. Usually "Close" is split-adjusted but not dividend-adjusted unless auto_adjust=True.
        # For simplicity in tournament, we use "Close". Dividends would improve High Div ETFs (SCHD) more.
        # But here valid comparison is mostly Growth vs Broad.

        rets = df["Close"].pct_change().fillna(0)

        # Strategy Ret
        strat_rets = rets * pd.Series(weights, index=df.index).shift(1).fillna(0)

        # Buy & Hold Ret
        bh_rets = rets  # Always 100%

        # Metrics Calculation
        def get_metrics(r, start_val=100.0):
            cum = (1 + r).cumprod() * start_val
            final = cum.iloc[-1]
            years = len(r) / 252
            cagr = (final / start_val) ** (1 / years) - 1 if years > 0 else 0

            dd = (cum - cum.cummax()) / cum.cummax()
            mdd = dd.min()

            sharpe = (r.mean() / r.std()) * np.sqrt(252) if r.std() != 0 else 0

            # Sortino
            downside = r[r < 0].std()
            sortino = (r.mean() / downside) * np.sqrt(252) if downside != 0 else 0

            return cagr, mdd, sharpe, sortino, final

        s_cagr, s_mdd, s_sharpe, s_sortino, s_final = get_metrics(strat_rets)
        b_cagr, b_mdd, b_sharpe, b_sortino, b_final = get_metrics(bh_rets)

        results.append(
            {
                "Ticker": ticker,
                "Start": first_date,
                "V4_CAGR": s_cagr,
                "V4_MDD": s_mdd,
                "V4_Sharpe": s_sharpe,
                "V4_Sortino": s_sortino,
                "BH_CAGR": b_cagr,  # For reference
                "BH_MDD": b_mdd,
            }
        )

    # Convert to DF
    res_df = pd.DataFrame(results).sort_values("V4_Sortino", ascending=False)

    print("\n" + "=" * 100)
    print("🏆 Antigravity V4 ETF Tournament Results")
    print("=" * 100)
    print(
        res_df.to_string(
            index=False,
            formatters={
                "V4_CAGR": "{:.2%}".format,
                "V4_MDD": "{:.2%}".format,
                "V4_Sharpe": "{:.2f}".format,
                "V4_Sortino": "{:.2f}".format,
                "BH_CAGR": "{:.2%}".format,
                "BH_MDD": "{:.2%}".format,
            },
        )
    )

    res_df.to_csv("d:/gg/etf_tournament_results.csv", index=False)
    print(f"\n✅ Results saved to d:/gg/etf_tournament_results.csv")


if __name__ == "__main__":
    run_etf_tournament()
