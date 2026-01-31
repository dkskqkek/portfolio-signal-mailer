"""
MAMA Lite v3 백테스팅 엔진 (v3.1 Critical Priority)
- Walk-Forward Testing (2004-2024)
- Transaction Cost 포함
- CAGR/MDD/Sharpe 등 다양한 지표 산출
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# MAMA Predictor 로드를 위한 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mama_lite_predictor import MAMAPredictor


class BacktestEngine:
    def __init__(self, start_date, end_date, initial_capital=100000):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Transaction Costs (한국투자증권 기준)
        self.commission_rate = 0.0025  # 0.25%
        self.slippage_bps = 5  # 5 basis points

        self.predictor = MAMAPredictor()
        self.portfolio_values = []
        self.trades = []
        self.daily_returns = []

    def calculate_transaction_cost(self, value):
        """거래 비용 계산"""
        commission = value * self.commission_rate
        slippage = value * (self.slippage_bps / 10000)
        return commission + slippage

    def rebalance(self, date, current_holdings, prices):
        """포트폴리오 리밸런싱"""
        # 1. MAMA 예측
        try:
            target_weights = self.predictor.predict_portfolio()
        except Exception as e:
            print(f"Warning: Prediction failed on {date}: {e}")
            return current_holdings

        if not target_weights:
            return current_holdings

        # 2. 현재 포트폴리오 가치
        portfolio_value = self.capital
        for ticker, qty in current_holdings.items():
            if ticker in prices and not pd.isna(prices[ticker]):
                portfolio_value += qty * prices[ticker]

        # 3. 목표 포지션 계산
        target_holdings = {}
        total_transaction_cost = 0

        for ticker, weight in target_weights.items():
            if ticker not in prices or pd.isna(prices[ticker]):
                continue

            target_value = portfolio_value * weight
            target_qty = int(target_value / prices[ticker])

            current_qty = current_holdings.get(ticker, 0)
            delta_qty = target_qty - current_qty

            if delta_qty != 0:
                # 거래 비용
                trade_value = abs(delta_qty * prices[ticker])
                cost = self.calculate_transaction_cost(trade_value)
                total_transaction_cost += cost

                # 거래 기록
                self.trades.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "side": "BUY" if delta_qty > 0 else "SELL",
                        "qty": abs(delta_qty),
                        "price": prices[ticker],
                        "cost": cost,
                    }
                )

            target_holdings[ticker] = target_qty

        # 4. 현금 업데이트
        self.capital = (
            portfolio_value
            - sum(
                qty * prices.get(ticker, 0)
                for ticker, qty in target_holdings.items()
                if ticker in prices and not pd.isna(prices.get(ticker))
            )
            - total_transaction_cost
        )

        return target_holdings

    def run(self, rebalance_freq="monthly", custom_tickers=None):
        """백테스트 실행"""
        print(
            f"🚀 Starting Backtest: {self.start_date.date()} to {self.end_date.date()}"
        )
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Rebalance Frequency: {rebalance_freq}")
        print(f"   Commission: {self.commission_rate * 100}%")
        print(f"   Slippage: {self.slippage_bps} bps\n")

        # 데이터 로드
        if custom_tickers:
            # Add macro/defensive tickers if not present
            all_tickers = list(
                set(custom_tickers + ["^VIX", "^TNX", "SPY", "BIL", "TLT"])
            )

            # Update predictor's universe temporarily
            self.predictor.gnn_tickers = custom_tickers

            # Adjacency Matrix Size Mismatch Fix
            # Custom Universe -> Use Identity Matrix (assume independent initially)
            import torch

            n = len(custom_tickers)
            self.predictor.adj_norm = torch.eye(n).to(self.predictor.device)
            print(f"⚠️ Using Identity Adjacency Matrix for custom universe ({n}x{n})")
        else:
            all_tickers = [
                "BIL",
                "TLT",
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "NVDA",
                "TSLA",
                "NFLX",
                "AVGO",
                "JNJ",
                "V",
                "SPY",
                "^VIX",
                "^TNX",
            ]

        print(f"📥 Downloading historical data for {len(all_tickers)} tickers...")
        data = yf.download(
            all_tickers,
            start=self.start_date - timedelta(days=10),
            end=self.end_date,
            progress=True,
        )["Close"]

        # 리밸런싱 날짜 생성
        if rebalance_freq == "monthly":
            rebalance_dates = pd.date_range(self.start_date, self.end_date, freq="MS")
        elif rebalance_freq == "weekly":
            rebalance_dates = pd.date_range(
                self.start_date, self.end_date, freq="W-MON"
            )
        else:
            rebalance_dates = data.index

        current_holdings = {}

        # 일별 시뮬레이션
        print(f"📊 Running simulation...")
        for date in data.index:
            if date < self.start_date:
                continue

            prices = data.loc[date].to_dict()

            # 일별 regime 추적 (v3.1 Smoothing) - 현재까지 데이터 전달
            data_until_today = data.loc[:date]
            self.predictor.update_regime_history(df=data_until_today)

            # 리밸런싱
            if date in rebalance_dates:
                current_holdings = self.rebalance(date, current_holdings, prices)

            # 포트폴리오 가치 계산
            holdings_value = sum(
                qty * prices.get(ticker, 0)
                for ticker, qty in current_holdings.items()
                if ticker in prices and not pd.isna(prices.get(ticker))
            )
            total_value = self.capital + holdings_value

            self.portfolio_values.append(
                {
                    "date": date,
                    "value": total_value,
                    "cash": self.capital,
                    "holdings": holdings_value,
                }
            )

            # 일별 수익률
            if len(self.portfolio_values) > 1:
                prev_value = self.portfolio_values[-2]["value"]
                daily_return = (
                    (total_value - prev_value) / prev_value if prev_value > 0 else 0
                )
                self.daily_returns.append(daily_return)

        print(f"✅ Simulation complete!\n")
        return self.calculate_metrics()

    def calculate_metrics(self):
        """성과 지표 계산"""
        df = pd.DataFrame(self.portfolio_values).set_index("date")

        # 기간 (연 단위)
        years = (df.index[-1] - df.index[0]).days / 365.25

        # CAGR
        total_return = df["value"].iloc[-1] / self.initial_capital
        cagr = (total_return ** (1 / years)) - 1 if years > 0 else 0

        # Returns
        returns = (
            pd.Series(self.daily_returns) if self.daily_returns else pd.Series([0])
        )

        # Sharpe Ratio
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        )

        # Max Drawdown
        cummax = df["value"].cummax()
        drawdown = (df["value"] - cummax) / cummax
        mdd = drawdown.min()

        # Win Rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # Calmar Ratio
        calmar = cagr / abs(mdd) if mdd != 0 else 0

        metrics = {
            "CAGR": cagr,
            "Total Return": total_return - 1,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": mdd,
            "Calmar Ratio": calmar,
            "Win Rate": win_rate,
            "Volatility": volatility,
            "Final Value": df["value"].iloc[-1],
            "Total Trades": len(self.trades),
            "Avg Trade Cost": np.mean([t["cost"] for t in self.trades])
            if self.trades
            else 0,
        }

        # CSV 저장 (사용자 요청)
        save_dir = r"d:\gg\data\backtest_results"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 일별 성과 (Daily Portfolio Value)
        metrics_file = os.path.join(save_dir, f"backtest_daily_{timestamp}.csv")
        df.to_csv(metrics_file)
        print(f"✅ Daily performance saved to: {metrics_file}")

        # 2. 거래 기록 (Trade Log)
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = os.path.join(save_dir, f"backtest_trades_{timestamp}.csv")
            trades_df.to_csv(trades_file, index=False)
            print(f"✅ Trade history saved to: {trades_file}")

        return metrics, df

    def print_report(self, metrics):
        """결과 리포트 출력"""
        print("\n" + "=" * 60)
        print("📊 BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${metrics['Final Value']:,.2f}")
        print("-" * 60)
        print(f"CAGR: {metrics['CAGR']:.2%}")
        print(f"Total Return: {metrics['Total Return']:.2%}")
        print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {metrics['Max Drawdown']:.2%}")
        print(f"Calmar Ratio: {metrics['Calmar Ratio']:.2f}")
        print(f"Win Rate: {metrics['Win Rate']:.2%}")
        print(f"Volatility (Ann.): {metrics['Volatility']:.2%}")
        print("-" * 60)
        print(f"Total Trades: {metrics['Total Trades']}")
        print(f"Avg Transaction Cost: ${metrics['Avg Trade Cost']:.2f}")
        print("=" * 60)

        # 목표 대비 평가
        print("\n📈 Performance Evaluation:")
        cagr_status = (
            "✅ EXCELLENT"
            if metrics["CAGR"] > 0.20
            else "✅ TARGET"
            if metrics["CAGR"] > 0.15
            else "⚠️  MINIMUM"
            if metrics["CAGR"] > 0.10
            else "❌ BELOW MIN"
        )
        sharpe_status = (
            "✅ EXCELLENT"
            if metrics["Sharpe Ratio"] > 1.3
            else "✅ TARGET"
            if metrics["Sharpe Ratio"] > 1.0
            else "⚠️  MINIMUM"
            if metrics["Sharpe Ratio"] > 0.7
            else "❌ BELOW MIN"
        )
        mdd_status = (
            "✅ EXCELLENT"
            if abs(metrics["Max Drawdown"]) < 0.15
            else "✅ TARGET"
            if abs(metrics["Max Drawdown"]) < 0.20
            else "⚠️  MINIMUM"
            if abs(metrics["Max Drawdown"]) < 0.30
            else "❌ BELOW MIN"
        )

        print(f"   CAGR: {cagr_status}")
        print(f"   Sharpe: {sharpe_status}")
        print(f"   MDD: {mdd_status}")
        print("=" * 60)


# 사용 예시
if __name__ == "__main__":
    # 5년 백테스트 (2021-2025)
    engine = BacktestEngine(
        start_date="2021-01-01", end_date="2025-12-31", initial_capital=100000
    )

    metrics, df = engine.run(rebalance_freq="monthly")
    engine.print_report(metrics)
