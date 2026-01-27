# -*- coding: utf-8 -*-
"""
Antigravity V4.1 Verified Backtest (Trust Version)
==================================================
검증 목표: 속도보다 '정확도'와 '신뢰도'
검증 방법:
1. Look-Ahead Bias(미래 참조) 원천 차단:
   - 모든 신호는 '어제' 종가 기준으로 계산하여 '오늘' 시가/종가에 진입
2. 정확한 데이터:
   - 야후 파이낸스 Adjusted Close 사용 (배당/분할 반영)
   - Monthly Rebalancing 시점의 정확한 달력 계산
3. 거래 비용 반영:
   - 슬리피지 + 수수료 = 회전당 0.1% (보수적 적용)

Logic V4.1:
1. Market State:
   - IF QQQ > SMA(110) AND QQQ > SMA(250) -> BULL (Buy QLD)
   - ELSE -> DEFENSIVE (Buy Top 1 of [GLD, TLT, BIL] based on 8-month momentum)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
START_DATE = "2008-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
INIT_CAPITAL = 10000.0
TX_COST = 0.001

# Tickers
TICKERS = ["QQQ", "QLD", "GLD", "TLT", "BIL"]


def fetch_data():
    print(f"Downloading data for {TICKERS}...")
    data = yf.download(
        TICKERS,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=True,
        group_by="column",
    )

    df = pd.DataFrame()
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.levels[0]:
                df = data["Close"]
            elif "Close" in data.columns.levels[1]:
                df = data.xs("Close", level=1, axis=1)
            else:
                if "Adj Close" in data.columns.levels[0]:
                    df = data["Adj Close"]
        else:
            if "Close" in data.columns:
                df = data[["Close"]]
            elif "Adj Close" in data.columns:
                df = data[["Adj Close"]]
    except Exception as e:
        print(f"Extraction Error: {e}")

    if df.empty and not data.empty:
        return pd.DataFrame()

    df = df.ffill().dropna()
    print(f"Data Loaded: {len(df)} days ({df.index[0].date()} ~ {df.index[-1].date()})")

    for t in TICKERS:
        if t not in df.columns:
            print(f"⚠️ Warning: {t} missing from data!")
            df[t] = np.nan

    return df


def run_verified_backtest():
    df = fetch_data()
    if df.empty:
        print("❌ Data Fetch Failed.")
        return

    # ==========================================
    # 2. 지표 계산 (Indicators) with Shift
    # ==========================================
    qqq = df["QQQ"]
    sma110 = qqq.rolling(110).mean()
    sma250 = qqq.rolling(250).mean()

    is_bull = (qqq > sma110) & (qqq > sma250)
    is_bull_shifted = is_bull.shift(1).fillna(False)

    # Monthly Momentum
    df_m = df.resample("M").last()
    mom_8m = df_m.pct_change(8)

    def_candidates = ["GLD", "TLT", "BIL"]
    valid_candidates = [c for c in def_candidates if c in mom_8m.columns]

    best_defensive = mom_8m[valid_candidates].idxmax(axis=1)
    target_def_daily = best_defensive.reindex(df.index).ffill().shift(1)

    # ==========================================
    # 3. 시뮬레이션 함수 (Run Simulation)
    # ==========================================
    def run_sim(bull_ticker, label):
        capital = INIT_CAPITAL
        shares = 0
        current_ticker = "CASH"
        history = []

        print(f"\n>>> Simulating: Bull Mode = {bull_ticker} ({label})")

        for date, row in df.iterrows():
            bull_mode = is_bull_shifted.loc[date]
            def_target = target_def_daily.loc[date]
            if pd.isna(def_target):
                def_target = "BIL" if "BIL" in df.columns else "CASH"

            target_ticker = bull_ticker if bull_mode else def_target

            if target_ticker != "CASH" and pd.isna(row.get(target_ticker)):
                target_ticker = "CASH"

            portfolio_value = 0
            if current_ticker != "CASH":
                price = row[current_ticker]
                if pd.isna(price):
                    price = 0
                portfolio_value = shares * price
            else:
                portfolio_value = capital

            if current_ticker != target_ticker:
                # Sell
                if current_ticker != "CASH":
                    portfolio_value = portfolio_value * (1 - TX_COST)
                # Buy
                if target_ticker != "CASH":
                    portfolio_value = portfolio_value * (1 - TX_COST)
                    new_price = row[target_ticker]
                    shares = portfolio_value / new_price
                else:
                    shares = 0

                current_ticker = target_ticker
                capital = portfolio_value

            history.append(
                {
                    "Date": date,
                    "Equity": portfolio_value,
                    "Holding": current_ticker,
                }
            )
            if current_ticker == "CASH":
                capital = portfolio_value

        return pd.DataFrame(history).set_index("Date")

    # Run Loop
    res_qld = run_sim("QLD", "2x Leverage")
    res_qqq = run_sim("QQQ", "1x No-Leverage")

    # Save
    res_qld.to_csv("research/verified_backtest_qld.csv")
    res_qqq.to_csv("research/verified_backtest_qqq.csv")

    # Report
    def report(df, name):
        final = df["Equity"].iloc[-1]
        cagr = (final / INIT_CAPITAL) ** (365 / (df.index[-1] - df.index[0]).days) - 1
        mdd = (df["Equity"] / df["Equity"].cummax() - 1).min()
        print(
            f"[{name}] CAGR: {cagr * 100:.2f}% | MDD: {mdd * 100:.2f}% | Final: ${final:,.0f}"
        )

    print("\n" + "=" * 50)
    print(" ⚖️ LEVERAGE COMPARISON (Verified) ")
    print("=" * 50)
    report(res_qld, "2x QLD")
    report(res_qqq, "1x QQQ")
    print("=" * 50)


if __name__ == "__main__":
    run_verified_backtest()
