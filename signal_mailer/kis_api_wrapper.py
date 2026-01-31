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

        Args:
            ticker: 종목코드 (예: "005930")
            period: 분봉 주기 ("1", "3", "5", "10", "15", "30")

        Returns:
            [
                {
                    "stck_bsop_date": "20260130",
                    "stck_cntg_hour": "0905",
                    "stck_prpr": "60200",
                    "stck_oprc": "60000",
                    "stck_hgpr": "60500",
                    "stck_lwpr": "59800",
                    "cntg_vol": "12345"
                },
                ...
            ]
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

        Args:
            ticker: 종목코드 (예: "AAPL")
            exchange: 거래소 ("NAS", "NYS", "AMS")
            period: 분봉 주기 (KIS는 "1"만 제공)

        Returns:
            [
                {
                    "xymd": "20260129",
                    "xhms": "093000",
                    "open": "150.25",
                    "high": "150.50",
                    "low": "150.10",
                    "last": "150.30",
                    "evol": "12345"
                },
                ...
            ]
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

                # Formula: KRW Total / USD Total (approximate FX used by KIS)
                # Or specific key: 'fx_rt' or 'frcr_buy_amt_smtl' etc.
                # In TTTS3012R, output2 often has 'fx_rt' or 'frcr_evlu_amt2' vs 'tot_evlu_amt'

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
