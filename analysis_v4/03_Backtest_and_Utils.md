# Antigravity v4.0 Backtest & Utilities Source Code

This document contains the backtesting engine, utility scripts, and configuration file (masked) for Antigravity v4.0.

---

## 1. backtest_v3_engine.py
**Path:** `signal_mailer/backtest_v3_engine.py`
**Description:** Backtesting engine supporting Walk-Forward Testing, transaction costs, and detailed performance metrics.

```python
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

    def run(self, rebalance_freq="monthly"):
        """백테스트 실행"""
        print(
            f"🚀 Starting Backtest: {self.start_date.date()} to {self.end_date.date()}"
        )
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Rebalance Frequency: {rebalance_freq}")
        print(f"   Commission: {self.commission_rate * 100}%")
        print(f"   Slippage: {self.slippage_bps} bps\\n")

        # 데이터 로드
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

        print("📥 Downloading historical data...")
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

        print(f"✅ Simulation complete!\\n")
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

        return metrics, df

    def print_report(self, metrics):
        """결과 리포트 출력"""
        print("\\n" + "=" * 60)
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
        print("\\n📈 Performance Evaluation:")
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
    # 5년 백테스트 (2019-2024)
    engine = BacktestEngine(
        start_date="2019-01-01", end_date="2024-12-31", initial_capital=100000
    )

    metrics, df = engine.run(rebalance_freq="monthly")
    engine.print_report(metrics)
```

---

## 2. save_backtest_results.py
**Path:** `signal_mailer/save_backtest_results.py`
**Description:** Utility to run the backtest and save results to a JSON file (includes NumpyEncoder).

```python
"""
백테스트 실행 및 결과를 JSON 파일로 저장
"""

import sys
import os
import json
import logging
import numpy as np

# 프로젝트 루트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from backtest_v3_engine import BacktestEngine


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


if __name__ == "__main__":
    print("Running backtest...")

    # 5년 백테스트 (v4.0 기간 확장 반영 가능)
    engine = BacktestEngine(
        start_date="2019-01-01", end_date="2024-12-31", initial_capital=100000
    )

    metrics, df = engine.run(rebalance_freq="monthly")

    # 결과를 dict로 변환 (JSON serializable)
    results = {
        "period": "2019-2024",
        "initial_capital": engine.initial_capital,
        "final_value": metrics["Final Value"],
        "cagr": metrics["CAGR"],
        "total_return": metrics["Total Return"],
        "sharpe_ratio": metrics["Sharpe Ratio"],
        "max_drawdown": metrics["Max Drawdown"],
        "calmar_ratio": metrics["Calmar Ratio"],
        "win_rate": metrics["Win Rate"],
        "volatility": metrics["Volatility"],
        "total_trades": metrics["Total Trades"],
        "avg_trade_cost": metrics["Avg Trade Cost"],
    }

    # JSON 파일로 저장
    output_path = r"d:\\gg\\backtest_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\\nResults saved to: {output_path}")
    print("\\nQuick Summary:")
    print(f"  CAGR: {metrics['CAGR']:.2%}")
    print(f"  Sharpe: {metrics['Sharpe Ratio']:.2f}")
    print(f"  MDD: {metrics['Max Drawdown']:.2%}")
```

---

## 3. scripts/collect_intraday_us.py
**Path:** `scripts/collect_intraday_us.py`
**Description:** Script to collect 1-minute intraday bars for US stocks via KIS API, resample to 5-minute, and save as Parquet. Includes logging and Discord notifications.

```python
"""
미국 주식 당일 1분봉 수집 스크립트 (KIS API)
실행 시각: 매일 06:30 KST (미국 장마감 후)

Usage:
    python scripts/collect_intraday_us.py
"""

import logging
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_mailer.kis_api_wrapper import KISAPIWrapper

# 로그 디렉토리 생성
log_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정 (Console + File)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "intraday_us.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# MAMA Lite 포트폴리오 종목
US_TICKERS = [
    # Core Holdings
    {"ticker": "QQQ", "exchange": "NAS"},
    {"ticker": "SPY", "exchange": "NYS"},
    # GNN - Big Tech
    {"ticker": "AAPL", "exchange": "NAS"},
    {"ticker": "MSFT", "exchange": "NAS"},
    {"ticker": "GOOGL", "exchange": "NAS"},
    {"ticker": "AMZN", "exchange": "NAS"},
    {"ticker": "META", "exchange": "NAS"},
    {"ticker": "NVDA", "exchange": "NAS"},
    {"ticker": "TSLA", "exchange": "NAS"},
    {"ticker": "NFLX", "exchange": "NAS"},
    {"ticker": "AVGO", "exchange": "NAS"},
    # Defensive Assets
    {"ticker": "BIL", "exchange": "NYS"},
    {"ticker": "TLT", "exchange": "NAS"},
    {"ticker": "GLD", "exchange": "NYS"},
    {"ticker": "UUP", "exchange": "NYS"},
    {"ticker": "BTAL", "exchange": "NYS"},
    {"ticker": "PFIX", "exchange": "NYS"},
    {"ticker": "DBMF", "exchange": "NYS"},
    {"ticker": "AGG", "exchange": "NYS"},
    {"ticker": "SHY", "exchange": "NAS"},
    # Value/Dividend
    {"ticker": "SCHD", "exchange": "NAS"},
    {"ticker": "VTI", "exchange": "NYS"},
]


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """1분봉 → 5분봉 변환"""
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    # Group by ticker and resample
    resampled_list = []
    for ticker, group in df.groupby("ticker"):
        group_resampled = (
            group.set_index("datetime")
            .resample("5min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        group_resampled["ticker"] = ticker
        group_resampled.reset_index(inplace=True)
        resampled_list.append(group_resampled)

    if resampled_list:
        return pd.concat(resampled_list, ignore_index=True)
    return pd.DataFrame()


def collect_us_intraday():
    """미국 주식 1분봉 수집 → 5분봉 변환"""
    # 1. Load Config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "signal_mailer", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # 2. Collect Data
    all_bars = []
    logger.info(f"Collecting 1min bars for {len(US_TICKERS)} US tickers...")

    for item in US_TICKERS:
        ticker = item["ticker"]
        exchange = item["exchange"]

        bars = kis.get_us_intraday_bars(ticker, exchange=exchange, period="1")
        if bars:
            for bar in bars:
                try:
                    # Parse time: "093000" → datetime
                    time_str = bar.get("xhms", "093000")
                    dt = datetime.strptime(
                        f"{datetime.now().strftime('%Y-%m-%d')} {time_str}",
                        "%Y-%m-%d %H%M%S",
                    )

                    all_bars.append(
                        {
                            "ticker": ticker,
                            "datetime": dt,
                            "open": float(bar.get("open", 0)),
                            "high": float(bar.get("high", 0)),
                            "low": float(bar.get("low", 0)),
                            "close": float(bar.get("last", 0)),
                            "volume": int(bar.get("evol", 0)),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error parsing bar for {ticker}: {e}")
                    continue

            logger.info(f"✓ {ticker}: {len(bars)} bars")
        else:
            logger.warning(f"✗ {ticker}: No data")

    # 3. Convert to DataFrame and Resample to 5min
    if not all_bars:
        logger.warning("No data collected")
        return

    df = pd.DataFrame(all_bars)
    logger.info(f"Collected {len(df)} 1min bars")

    df_5min = resample_to_5min(df)
    logger.info(f"Resampled to {len(df_5min)} 5min bars")

    # 4. Save to Parquet
    output_dir = Path("data/intraday/us")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{datetime.now().strftime('%Y-%m-%d')}.parquet"
    filepath = output_dir / filename

    df_5min.to_parquet(filepath, compression="snappy", index=False)
    logger.info(f"✅ Saved {len(df_5min)} records (5min bars) to {filepath}")
    logger.info(f"   Tickers: {df_5min['ticker'].nunique()}")
    logger.info(f"   File size: {filepath.stat().st_size / 1024:.1f} KB")

    # 5. Discord Notification
    try:
        from signal_mailer.notification.discord_webhook import send_discord_msg

        send_discord_msg(
            config,
            "📊 [Data] US Intraday Collection",
            f"수집 완료: {df_5min['ticker'].nunique()} 종목\\n1분봉: {len(df)} → 5분봉: {len(df_5min)}\\n파일: `{filename}`",
            color=0x00BFFF,
        )
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")


if __name__ == "__main__":
    collect_us_intraday()
```

---

## 4. config.yaml (Masked)
**Path:** `signal_mailer/config.yaml`
**Description:** Configuration file. **CRITICAL: Secrets have been masked.**

```yaml
scheduler:
  run_time: "09:00"
  timezone: Asia/Seoul

email:
  smtp_server: smtp.gmail.com
  smtp_port: 587
  sender_email: gamjatangjo@gmail.com
  sender_password: "****************"
  recipient_email: gamjatangjo@gmail.com
  subject_template: "[포트폴리오 신호] {status}"

strategy_info:
  description: "MAMA Lite v4.0 - Attention GNN + Diversified"
  signal_logic: "Multi-Head Attention GNN (4 heads) + SRL Regime"
  # Dynamic GNN Candidates (11 tickers)
  gnn_tickers:
    [
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
    ]
  # 9-ETF Universe (MAMA Pro)
  etf_universe: ["SPY", "QQQ", "IWM", "TLT", "IEF", "SHY", "GLD", "DBC", "BIL"]
  weights:
    tactical: 0.45
    kospi: 0.20
    spy: 0.20
    gold: 0.15

trading:
  slippage_bps: 5.0 # 1bp = 0.01%
  commission_rate: 0.0025 # 0.25% (Overseas average)
  max_daily_trades: 15

gemini:
  api_key: "***********************************"
  enabled: true

discord:
  webhook_url: "https://discord.com/api/webhooks/******************************************************************"
  token: "************************************************************************"
  enabled: true

history_file: d:/gg/data/signal_history.json
log_file: d:/gg/signal_mailer/mailer.log
debug_mode: false

kis:
  app_key: "************************************"
  app_secret: "********************************************************************************************************************************************************"
  account_no: "********"
  account_prod_code: "01"
  is_mock: true
  url_base: "https://openapivts.koreainvestment.com:29443"
```
