# Antigravity v4.0 Execution & API Source Code

This document contains the execution scripts and API wrappers for Antigravity v4.0.

---

## 1. execute_mama_lite.py
**Path:** `execute_mama_lite.py`
**Description:** Main execution script for MAMA Lite strategy. Handles initialization, prediction, portfolio rebalancing, and order execution.

```python
# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys
import time
from datetime import datetime

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor
from signal_mailer.mama_lite_predictor import MAMAPredictor
from signal_mailer.trade_limit_counter import TradeLimitCounter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MAMA_Execution")


def send_discord_msg(config, title, message, color=0x00FF00, fields=None):
    """Send enhanced Discord notification with optional structured fields.

    Args:
        config: Configuration dictionary with Discord webhook URL
        title: Notification title
        message: Main message body
        color: Embed color (default: green)
        fields: Optional list of {"name": str, "value": str, "inline": bool} dicts
    """
    import requests

    webhook_url = config.get("discord", {}).get("webhook_url")
    if not webhook_url:
        return

    embed = {
        "title": title,
        "description": message,
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if fields:
        embed["fields"] = fields

    payload = {"embeds": [embed]}

    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Discord notice failed: {e}")


def get_exchange_code(ticker):
    """
    Return exchange code for KIS US API.
    Updated for v3.0 9-ETF Universe.
    """
    # MAMA Pro Universe
    # SPY: ARCA(AMS), QQQ: NASDAQ(NAS), IWM: ARCA(AMS)
    # TLT/IEF/SHY: NASDAQ(NAS)
    # GLD: ARCA(AMS), DBC: ARCA(AMS), BIL: ARCA(AMS)
    nas_list = [
        "TLT",
        "QQQ",
        "IEF",
        "SHY",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "NFLX",
        "AVGO",
    ]
    if ticker in nas_list:
        return "NAS"
    return "AMS"


class TransactionCostModel:
    def __init__(self, slippage_bps=5.0, commission_rate=0.0025):
        self.slippage = slippage_bps / 10000.0
        self.comm = commission_rate

    def estimate_cost(self, amount_usd):
        return amount_usd * (self.slippage + self.comm)


def run_mama_lite_execution(dry_run=False):
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 1. Initialize Components
    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)
    predictor = MAMAPredictor(config_path=config_path)

    t_config = config.get("trading", {})
    trade_limiter = TradeLimitCounter(
        limits_file=os.path.join(current_dir, "data", "trade_limits.json"),
        max_daily_trades=t_config.get("max_daily_trades", 15),
    )
    cost_model = TransactionCostModel(
        slippage_bps=t_config.get("slippage_bps", 5.0),
        commission_rate=t_config.get("commission_rate", 0.0025),
    )

    print(
        f"\\n--- [MODE: {'MOCK' if kis.is_mock else 'REAL'}] Antigravity v3.0 Engine ---"
    )

    # Real-time FX
    exch_rate = kis.get_exchange_rate()
    print(f"💱 Real-time USD/KRW Rate: {exch_rate:,.2f}")

    if dry_run:
        print("💡 DRY RUN MODE: No orders will be executed.")

    # Check trade limit
    remaining_trades = trade_limiter.get_remaining("mama_lite")
    print(f"📊 Remaining trades today: {remaining_trades}")
    if remaining_trades <= 0:
        print("⚠️ DAILY TRADE LIMIT REACHED. Aborting.")
        return

    # 2. Get Predicted Weights
    target_weights = predictor.predict_portfolio()
    if not target_weights:
        print("❌ Prediction Failed. Aborting.")
        return

    # 3. Get Current Portfolio (US)
    holdings = executor.get_us_balance()
    current_holdings = {}
    current_total_val_usd = 0.0

    print("\\n📊 Current US Holdings:")
    for h in holdings:
        ticker = h.get("ovrs_pdno", "")
        qty = float(h.get("ovrs_cblc_qty", 0))
        val_usd = float(h.get("frcr_evlu_amt2", 0))
        if qty > 0:
            current_holdings[ticker] = qty
            current_total_val_usd += val_usd
            print(f"   - {ticker}: {qty} shares (${val_usd:,.2f})")

    # Capital Allocation Rules
    global_total_equity_krw = executor.get_total_equity()
    target_us_equity_krw = global_total_equity_krw * 0.5
    target_us_equity_usd = target_us_equity_krw / exch_rate

    print(f"\\n🌍 Net Worth: {global_total_equity_krw:,.0f}원")
    print(f"🎯 Target US Equity: ${target_us_equity_usd:,.2f}")

    total_equity_base = target_us_equity_usd

    # 4. Rebalancing Logic
    trades = []

    # Sell Loop
    for t, qty in current_holdings.items():
        target_w = target_weights.get(t, 0.0)
        curr_price = kis.get_us_current_price(t, exchange=get_exchange_code(t))
        if not curr_price:
            continue

        current_val = qty * curr_price
        target_val = total_equity_base * target_w
        diff_val = target_val - current_val

        if diff_val < -50:  # Threshold $50
            sell_qty = int(abs(diff_val) / curr_price)
            if sell_qty > 0:
                cost = cost_model.estimate_cost(sell_qty * curr_price)
                trades.append(
                    {
                        "ticker": t,
                        "side": "SELL",
                        "qty": sell_qty,
                        "price": curr_price,
                        "cost": cost,
                    }
                )

    # Buy Loop
    for t, w in target_weights.items():
        curr_price = kis.get_us_current_price(t, exchange=get_exchange_code(t))
        if not curr_price:
            continue

        current_qty = current_holdings.get(t, 0)
        current_val = current_qty * curr_price
        target_val = total_equity_base * w
        diff_val = target_val - current_val

        if diff_val > 50:
            buy_qty = int(diff_val / curr_price)
            if buy_qty > 0:
                cost = cost_model.estimate_cost(buy_qty * curr_price)
                # Risk limit: max 20%
                if (buy_qty * curr_price) > (total_equity_base * 0.20):
                    buy_qty = int((total_equity_base * 0.20) / curr_price)

                if buy_qty > 0:
                    trades.append(
                        {
                            "ticker": t,
                            "side": "BUY",
                            "qty": buy_qty,
                            "price": curr_price,
                            "cost": cost,
                        }
                    )

    # 5. Execution
    print(f"\\n📝 Generated {len(trades)} Trades:")
    for trade in trades:
        print(
            f"   {trade['side']} {trade['ticker']} {trade['qty']} shares @ ${trade['price']:.2f}"
        )

    if not dry_run:
        # execute
        for trade in trades:
            xc = get_exchange_code(trade["ticker"])

            # Check trade limit
            if not trade_limiter.check_and_increment("mama_lite"):
                print(
                    f"   ⚠️  Cannot execute {trade['side']} for {trade['ticker']}: Daily trade limit reached"
                )
                send_discord_msg(
                    config,
                    "⚠️ [MAMA Lite] Trade Limit Blocked",
                    f"{trade['ticker']} {trade['side']} 차단: 일일 거래 한도 도달",
                    color=0xFFA500,
                )
                continue

            res = executor.create_us_order(
                ticker=trade["ticker"],
                side=trade["side"],
                qty=trade["qty"],
                price=trade["price"],
                exchange=xc,
                ord_type="00",
            )

            if res:
                msg = f"{trade['side']} {trade['ticker']} ({trade['qty']}sh) Executed"
                send_discord_msg(config, "MAMA Lite Trade", msg)
                time.sleep(0.5)

    print("\\n[MAMA Lite Execution Finished]")


if __name__ == "__main__":
    # Default to DRY RUN for safety integration
    run_mama_lite_execution(dry_run=True)
```

---

## 2. kis_api_wrapper.py
**Path:** `signal_mailer/kis_api_wrapper.py`
**Description:** Wrapper class for Korea Investment & Securities (KIS) API. Handles authentication, quota management, and API calls.

```python
import logging
import time
import os
import requests
from collections import deque
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_calls: int, period: float = 1.0):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()

    def wait(self) -> None:
        while True:
            now = time.time()
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return
            time.sleep(0.01)


class KISAPIWrapper:
    """
    Modular wrapper for Korea Investment & Securities (KIS) Open API.
    Adapted from kis_bot_v7_4.py with standardized structure.
    """

    def __init__(self, config: Dict[str, Any]):
        self.app_key = config.get("app_key")
        self.app_secret = config.get("app_secret")
        self.cano = config.get("account_no")
        self.acnt_prdt_cd = config.get("account_prod_code", "01")
        self.is_mock = config.get("is_mock", False)
        self.base_url = config.get("url_base")

        self.access_token = ""
        self.headers = {}

        # Rate Limiters (Mock: 2 GET/sec, 1 POST/sec | Real: 20 GET/sec, 20 POST/sec)
        if self.is_mock:
            self.limiter_get = RateLimiter(2, 1.0)
            self.limiter_post = RateLimiter(1, 1.0)
        else:
            self.limiter_get = RateLimiter(20, 1.0)
            self.limiter_post = RateLimiter(20, 1.0)

        self._auth()

    def _auth(self) -> None:
        """Authenticate and retrieve access token with local caching."""
        import json  # Local import to avoid any issues

        # Use different cache files for mock vs real to avoid collisions
        env_suffix = "mock" if self.is_mock else "real"
        cache_path = os.path.join(os.getcwd(), f".kis_token_cache_{env_suffix}.json")

        # 1. Try Loading from Cache
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cache = json.load(f)
                    # Expiry check (23 hours)
                    if time.time() - cache.get("timestamp", 0) < 23 * 3600:
                        self.access_token = cache.get("access_token")
                        self.headers = {
                            "content-type": "application/json",
                            "authorization": f"Bearer {self.access_token}",
                            "appkey": self.app_key,
                            "appsecret": self.app_secret,
                            "custtype": "P",
                        }
                        logger.debug(f"[KIS] Loaded {env_suffix} token from cache")
                        return
            except Exception:
                pass

        # 2. Issue New Token
        url = f"{self.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        self.limiter_post.wait()
        try:
            r = requests.post(
                url, json=body, headers={"Content-Type": "application/json"}, timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                self.access_token = data.get("access_token")
                # Save to Cache
                with open(cache_path, "w") as f:
                    json.dump(
                        {"access_token": self.access_token, "timestamp": time.time()}, f
                    )

                self.headers = {
                    "content-type": "application/json",
                    "authorization": f"Bearer {self.access_token}",
                    "appkey": self.app_key,
                    "appsecret": self.app_secret,
                    "custtype": "P",
                }
                env_suffix = "mock" if self.is_mock else "real"
                logger.debug(f"[KIS] New {env_suffix} token issued and cached")
            else:
                logger.error(f"[KIS] Auth Failed Status {r.status_code}: {r.text}")
        except Exception as e:
            logger.error(f"[KIS] Auth Exception: {e}")

    def get_current_price(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Inquire current price for a domestic stock."""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self.headers.copy()
        headers["tr_id"] = "FHKST01010100"
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}
        self.limiter_get.wait()
        try:
            r = requests.get(url, headers=headers, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("rt_cd") != "0":
                    logger.error(f"[KIS] Price API Error: {data.get('msg1')}")
                return data.get("output")
            return None
        except Exception as e:
            logger.error(f"[KIS] Price Inquiry Error for {ticker}: {e}")
            return None

    def get_ohlcv_recent(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Inquire daily OHLCV for recent days (Domestic)."""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        headers = self.headers.copy()
        headers["tr_id"] = "FHKST01010400"
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "1",
        }
        self.limiter_get.wait()
        try:
            r = requests.get(url, headers=headers, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("rt_cd") != "0":
                    logger.error(
                        f"[KIS] OHLCV API Error for {ticker}: {data.get('msg1')}"
                    )
                return data
            return None
        except Exception as e:
            logger.error(f"[KIS] OHLCV Inquiry Error for {ticker}: {e}")
            return None

    def get_us_current_price(
        self, ticker: str, exchange: str = "NAS"
    ) -> Optional[float]:
        """
        Inquire current price for a US stock.
        exchange: NAS (Nasdaq), NYS (NYSE), AMS (Amex)
        """
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price"
        headers = self.headers.copy()
        headers["tr_id"] = (
            "HHDFS00000300"  # Real/Mock same for quotation usually, check docs if fails
        )

        # Mapping for safety, though user should pass correct code
        excd = exchange.upper()

        params = {
            "AUTH": "",
            "EXCD": excd,
            "SYMB": ticker,
        }
        self.limiter_get.wait()
        try:
            r = requests.get(url, headers=headers, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("rt_cd") != "0":
                    logger.error(
                        f"[KIS-US] Price API Error for {ticker}: {data.get('msg1')}"
                    )
                    return None

                output = data.get("output")
                if output:
                    last_str = str(output.get("last", "0"))
                    if not last_str.strip():
                        return None
                    return float(last_str)
            return None
        except Exception as e:
            logger.error(f"[KIS-US] Price Inquiry Error for {ticker}: {e}")
            return None

    def get_us_ohlcv_recent(
        self, ticker: str, exchange: str = "NAS"
    ) -> Optional[Dict[str, Any]]:
        """Inquire daily OHLCV for recent days (US)."""
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/dailyprice"
        headers = self.headers.copy()
        headers["tr_id"] = "HHDFS76240000"

        excd = exchange.upper()
        params = {
            "AUTH": "",
            "EXCD": excd,
            "SYMB": ticker,
            "GUBN": "0",  # 0: Daily, 1: Weekly, 2: Monthly
            "BYMD": "",  # Empty for most recent
            "MODP": "1",  # 1: Adjusted Price
        }
        self.limiter_get.wait()
        try:
            r = requests.get(url, headers=headers, params=params, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("rt_cd") != "0":
                    logger.error(
                        f"[KIS-US] OHLCV API Error for {ticker}: {data.get('msg1')}"
                    )
                return data
            return None
        except Exception as e:
            logger.error(f"[KIS-US] OHLCV Inquiry Error for {ticker}: {e}")
            return None

    def get_intraday_bars(
        self, ticker: str, period: str = "5"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        국내 주식 당일 분봉 조회

        TR_ID: FHKST03010200
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        headers = self.headers.copy()
        headers["tr_id"] = "FHKST03010200"

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_PERIOD_DIV_CODE": period,
            "FID_INPUT_HOUR_1": "090000",  # 조회 시작 시간
        }

        self.limiter_get.wait()
        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("rt_cd") == "0":
                    return data.get("output2", [])
                else:
                    logger.error(
                        f"[KIS] Intraday bars error for {ticker}: {data.get('msg1')}"
                    )
            return None
        except Exception as e:
            logger.error(f"[KIS] Intraday bars error for {ticker}: {e}")
            return None

    def get_us_intraday_bars(
        self, ticker: str, exchange: str = "NAS", period: str = "1"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        미국 주식 당일 분봉 조회
        TR_ID: HHDFS76950200
        """
        url = f"{self.base_url}/uapi/overseas-stock/v1/quotations/inquire-time-itemchartprice"
        headers = self.headers.copy()
        headers["tr_id"] = "HHDFS76950200"

        params = {
            "AUTH": "",
            "EXCD": exchange.upper(),
            "SYMB": ticker,
            "NMIN": period,
            "PINC": "0",
            "NEXT": "",
            "NREC": "120",  # 최대 조회 건수
        }

        self.limiter_get.wait()
        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("rt_cd") == "0":
                    return data.get("output2", [])
                else:
                    logger.error(
                        f"[KIS-US] Intraday bars error for {ticker}: {data.get('msg1')}"
                    )
            return None
        except Exception as e:
            logger.error(f"[KIS-US] Intraday bars error for {ticker}: {e}")
            return None

    def get_exchange_rate(self) -> float:
        """
        Fetch real-time USD/KRW exchange rate from KIS.
        Uses Overseas Balance Inquiry (TTTS3012R/VTTS3012R) result which includes applied FX rate.
        """
        tr_id = "TTTS3012R" if not self.is_mock else "VTTS3012R"
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"

        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "OVRS_EXCG_CD": "NAS",  # Any US exchange works
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }

        self.limiter_get.wait()
        try:
            r = requests.get(
                url,
                headers=self.headers.copy() | {"tr_id": tr_id},
                params=params,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                # output2 contains summary info including 'frcr_evlu_amt2' (USD total) and 'tot_evlu_amt' (KRW total)
                summary = data.get("output2", {})
                if isinstance(summary, list):
                    summary = summary[0] if summary else {}

                # Fallback to a common key if directly available
                fx_rt = summary.get("fx_rt") or summary.get("frcr_buy_amt_smtl1")
                if fx_rt and float(fx_rt) > 500:
                    return float(fx_rt)

                # Global fallback calculation
                usd_val = float(summary.get("frcr_evlu_amt2", 0))
                krw_val = float(summary.get("tot_evlu_amt", 0))
                if usd_val > 0 and krw_val > 0:
                    calculated_fx = krw_val / usd_val
                    return round(calculated_fx, 2)

            logger.warning(
                "[KIS] Exchange rate API failed or returned 0. Using fallback 1400.0"
            )
            return 1400.0
        except Exception as e:
            logger.error(f"[KIS] Exchange rate exception: {e}")
            return 1400.0

    def call_get(
        self, url: str, headers: Dict[str, str], params: Dict[str, Any]
    ) -> requests.Response:
        """Rate-limited GET request."""
        self.limiter_get.wait()
        return requests.get(url, headers=headers, params=params, timeout=10)

    def call_post(
        self, url: str, headers: Dict[str, str], json: Dict[str, Any]
    ) -> requests.Response:
        """Rate-limited POST request."""
        self.limiter_post.wait()
        return requests.post(url, headers=headers, json=json, timeout=10)
```

---

## 3. update_adjacency_matrix.py
**Path:** `signal_mailer/update_adjacency_matrix.py`
**Description:** Script to dynamically update the adjacency matrix based on asset correlations.

```python
"""
Adjacency Matrix 동적 업데이트 (v3.2)
상관계수 기반 분기별 자동 업데이트
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

GNN_DATA_DIR = r"d:\\gg\\data\\gnn"
ADJ_FILE = os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv")
ADJ_BACKUP = os.path.join(GNN_DATA_DIR, "adjacency_matrix_backup.csv")

# v4.0: Expanded tickers with Healthcare (JNJ) and Financials (V)
GNN_TICKERS = [
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


def calculate_correlation_matrix(lookback_days=252):
    """Calculate correlation matrix from recent data."""
    print(f"📥 Downloading {lookback_days} days of data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 50)

    data = yf.download(GNN_TICKERS, start=start_date, end=end_date, progress=False)[
        "Close"
    ]
    data = data.ffill().dropna()

    # Daily returns correlation
    returns = data.pct_change().dropna()
    corr_matrix = returns.corr()

    return corr_matrix


def correlation_to_adjacency(corr_matrix, threshold=0.5, include_self=True):
    """Convert correlation matrix to binary adjacency matrix.

    Args:
        corr_matrix: Correlation matrix
        threshold: Minimum correlation to establish connection
        include_self: Include self-loops (diagonal = 1)

    Returns:
        Binary adjacency matrix
    """
    adj = (corr_matrix.abs() >= threshold).astype(int)

    if include_self:
        np.fill_diagonal(adj.values, 1)
    else:
        np.fill_diagonal(adj.values, 0)

    return adj


def update_adjacency_matrix(threshold=0.5, lookback_days=252):
    """Update adjacency matrix based on current correlations."""
    print("=" * 60)
    print("Adjacency Matrix 동적 업데이트 (v3.2)")
    print("=" * 60)

    # 1. Backup current matrix
    if os.path.exists(ADJ_FILE):
        import shutil

        shutil.copy(ADJ_FILE, ADJ_BACKUP)
        print(f"\\n[1] 기존 행렬 백업: {ADJ_BACKUP}")

    # 2. Calculate current correlations
    print(f"\\n[2] 상관계수 계산 (최근 {lookback_days}일)...")
    corr_matrix = calculate_correlation_matrix(lookback_days)

    # 3. Convert to adjacency
    print(f"\\n[3] Adjacency 변환 (threshold: {threshold})...")
    new_adj = correlation_to_adjacency(corr_matrix, threshold)

    # 4. Compare with old matrix (only if same shape)
    if os.path.exists(ADJ_BACKUP):
        old_adj = pd.read_csv(ADJ_BACKUP, index_col=0)
        if old_adj.shape == new_adj.shape:
            changes = (new_adj.values != old_adj.values).sum()
            print(f"\\n[4] 변경된 연결 수: {changes}")

            if changes > 0:
                print("\\n변경 상세:")
                for i, ticker1 in enumerate(GNN_TICKERS):
                    for j, ticker2 in enumerate(GNN_TICKERS):
                        if i < j and new_adj.iloc[i, j] != old_adj.iloc[i, j]:
                            old_val = old_adj.iloc[i, j]
                            new_val = new_adj.iloc[i, j]
                            action = "추가" if new_val == 1 else "제거"
                            corr_val = corr_matrix.iloc[i, j]
                            print(
                                f"  {ticker1}-{ticker2}: {action} (상관계수: {corr_val:.3f})"
                            )
        else:
            print(
                f"\\n[4] 티커 수 변경: {old_adj.shape[0]} -> {new_adj.shape[0]} (비교 생략)"
            )

    # 5. Save new matrix
    new_adj.to_csv(ADJ_FILE)
    print(f"\\n[5] 새 행렬 저장: {ADJ_FILE}")

    # 6. Statistics
    total_connections = (new_adj.sum().sum() - len(GNN_TICKERS)) / 2
    possible_connections = len(GNN_TICKERS) * (len(GNN_TICKERS) - 1) / 2
    density = total_connections / possible_connections

    print(f"\\n📊 통계:")
    print(f"  연결 수: {int(total_connections)}/{int(possible_connections)}")
    print(f"  밀도: {density:.1%}")

    # Show connection details per ticker
    print(f"\\n📈 티커별 연결:")
    for ticker in GNN_TICKERS:
        connections = new_adj.loc[ticker].sum() - 1  # Exclude self
        connected_to = [
            t for t in GNN_TICKERS if t != ticker and new_adj.loc[ticker, t] == 1
        ]
        print(f"  {ticker}: {int(connections)}개 ({', '.join(connected_to)})")

    print("\\n" + "=" * 60)
    print("Adjacency Matrix 업데이트 완료!")
    print("=" * 60)

    return new_adj, corr_matrix


if __name__ == "__main__":
    new_adj, corr_matrix = update_adjacency_matrix(threshold=0.5, lookback_days=252)

    # Save correlation matrix for reference
    corr_file = os.path.join(GNN_DATA_DIR, "correlation_matrix.csv")
    corr_matrix.to_csv(corr_file)
    print(f"\\n상관계수 행렬 저장: {corr_file}")
```
