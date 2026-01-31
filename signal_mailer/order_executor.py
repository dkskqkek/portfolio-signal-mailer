# -*- coding: utf-8 -*-
import logging
import time
from typing import Dict, Any, List, Optional
from signal_mailer.kis_api_wrapper import KISAPIWrapper

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Handles order execution for Korean stocks via KIS API.
    Designed for the 'Hybrid Alpha' strategy.
    """

    def __init__(self, kis: KISAPIWrapper):
        self.kis = kis
        self.base_url = kis.base_url
        self.cano = kis.cano
        self.acnt_prdt_cd = kis.acnt_prdt_cd

    def _get_order_headers(self, tr_id: str) -> Dict[str, str]:
        """Prepare headers for order-related TR IDs."""
        headers = self.kis.headers.copy()
        headers["tr_id"] = tr_id
        return headers

    def create_order(
        self, ticker: str, side: str, qty: int, price: int = 0, ord_type: str = "01"
    ) -> Optional[Dict[str, Any]]:
        """
        Create a localized market or limit order.
        side: 'BUY' or 'SELL'
        ord_type: '01' (Market), '00' (Limit)
        """
        tr_id = "TTTC0802U" if side == "BUY" else "TTTC0801U"
        if self.kis.is_mock:
            tr_id = "VTTC0802U" if side == "BUY" else "VTTC0801U"

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"

        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": ticker,
            "ORD_DVSN": ord_type,
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price) if ord_type == "00" else "0",
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = self.kis.call_post(
                    url, headers=self._get_order_headers(tr_id), json=body
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("rt_cd") == "0":
                        logger.info(f"[ORDER] {side} {ticker} Success: {qty} shares")
                        return data
                    else:
                        logger.error(
                            f"[ORDER] {side} {ticker} Failed: {data.get('msg1')}"
                        )
                        return data
                else:
                    logger.error(
                        f"[ORDER] {side} {ticker} API Error: {r.status_code} {r.text}"
                    )
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) * 0.5  # 0.5s, 1s, 2s
                        logger.info(
                            f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    return None
            except Exception as e:
                logger.error(f"[ORDER] Execution error for {ticker}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) * 0.5
                    logger.info(
                        f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                return None
        return None

    def get_balance(self) -> List[Dict[str, Any]]:
        """Fetch current stock holdings."""
        tr_id = "TTTC8434R"  # Real
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLG": "N",
            "OFR_FLG": "N",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FRLI_GVOFF_SYS_DVSN": "00",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        if self.kis.is_mock:
            tr_id = "VTTC8434R"  # Mock
            params = {
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "N",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            }

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"

        try:
            headers = self.kis.headers.copy()
            headers["tr_id"] = tr_id
            r = self.kis.call_get(url, headers=headers, params=params)
            if r.status_code == 200:
                data = r.json()
                return data.get("output1", [])
            return []
        except Exception as e:
            logger.error(f"[BALANCE] Inquiry error: {e}")
            return []

    def get_cash(self) -> int:
        """Get available cash for trading."""
        tr_id = "TTTC8434R"  # Real
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLG": "N",
            "OFR_FLG": "N",
            "INQR_DVSN": "01",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FRLI_GVOFF_SYS_DVSN": "00",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        if self.kis.is_mock:
            tr_id = "VTTC8434R"  # Mock
            params = {
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "N",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            }

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        try:
            headers = self.kis.headers.copy()
            headers["tr_id"] = tr_id
            logger.info(f"[CASH-Probing] URL: {url} | Params: {params} | TR: {tr_id}")
            r = self.kis.call_get(url, headers=headers, params=params)
            if r.status_code == 200:
                data = r.json()
                # output2 contains summary data including cash
                summary_list = data.get("output2", [])
                if not summary_list:
                    logger.warning(f"[CASH] output2 is empty. Data: {data}")
                    return 0
                summary = summary_list[0]
                # Try multiple keys for cash in mock/real summary
                v1 = summary.get("dnca_tot_amt")
                v2 = summary.get("tot_evlu_amt")
                v3 = summary.get("nass_amt")
                logger.debug(f"[CASH] Raw keys: dnca={v1}, tot={v2}, nass={v3}")
                try:
                    cash = int(float(v1 or v2 or v3 or 0))
                except (ValueError, TypeError):
                    cash = 0
                return cash
            else:
                logger.error(f"[CASH] API Error: {r.status_code} {r.text}")
                return 0
        except Exception as e:
            logger.error(f"[CASH] Inquiry error: {e}")
            return 0

    def create_us_order(
        self,
        ticker: str,
        exchange: str,
        side: str,
        qty: int,
        price: float = 0,
        ord_type: str = "00",
    ) -> Optional[Dict[str, Any]]:
        """
        Create a US stock market or limit order.
        side: 'BUY' or 'SELL'
        ord_type: '00' (Limit - Specify Price), 'LOO', 'LOC', 'MOO', 'MOC' etc.
        NOTE: KIS Overseas API does NOT support simple '01' (Market) for US stocks easily.
              Usually requires '00' with price.
              However, for 'Market' equivalent, we might need to use specific codes or price logic.
              For now, we will stick to LIMIT orders (00) as default for safety.
        """
        tr_id = (
            "JTTT1002U" if side == "BUY" else "JTTT1006U"
        )  # Real (US specific TR may vary, check JTTT1002U=Buy)
        # Actually standard: JTTT1002U (Buy), JTTT1006U (Sell)

        if self.kis.is_mock:
            tr_id = "VTTT1002U" if side == "BUY" else "VTTT1006U"

        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"

        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "OVRS_EXCG_CD": exchange.upper(),
            "PDNO": ticker,
            "ORD_DVSN": ord_type,  # 00: Limit
            "ORD_QTY": str(qty),
            "OVRS_ORD_UNPR": str(price),
            "ORD_SVR_DVSN_CD": "0",  # 0: Normal
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = self.kis.call_post(
                    url, headers=self._get_order_headers(tr_id), json=body
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("rt_cd") == "0":
                        logger.info(
                            f"[US-ORDER] {side} {ticker} Success: {qty} shares @ {price}"
                        )
                        return data
                    else:
                        logger.error(
                            f"[US-ORDER] {side} {ticker} Failed: {data.get('msg1')}"
                        )
                        return data
                else:
                    logger.error(
                        f"[US-ORDER] {side} {ticker} API Error: {r.status_code} {r.text}"
                    )
                    if attempt < max_retries - 1:
                        wait_time = (2**attempt) * 0.5  # 0.5s, 1s, 2s
                        logger.info(
                            f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    return None
            except Exception as e:
                logger.error(f"[US-ORDER] Execution error for {ticker}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) * 0.5
                    logger.info(
                        f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                return None
        return None

    def get_us_balance(self) -> List[Dict[str, Any]]:
        """Fetch US stock holdings by probing multiple exchanges."""
        tr_id = "TTTS3012R"
        if self.kis.is_mock:
            tr_id = "VTTS3012R"

        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
        exchanges = ["AMS", "NAS", "NYS"]
        all_holdings = []

        for ex in exchanges:
            params = {
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "OVRS_EXCG_CD": ex,
                "TR_CRCY_CD": "USD",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": "",
            }
            try:
                headers = self.kis.headers.copy()
                headers["tr_id"] = tr_id
                r = self.kis.call_get(url, headers=headers, params=params)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("rt_cd") == "0":
                        all_holdings.extend(data.get("output1", []))
            except Exception as e:
                logger.error(f"[US-BALANCE] Error for {ex}: {e}")

        return all_holdings

    def get_us_cash(self) -> float:
        """Get available US cash (Order Possible Amount)."""
        # TR for US Buy Possible: TTTS3007R (Real), VTTS3007R (Mock)
        tr_id = "TTTS3007R"
        if self.kis.is_mock:
            tr_id = "VTTS3007R"

        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order"

        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "OVRS_EXCG_CD": "NAS",
            "OVRS_ORD_UNPR": "0",
            "ITEM_CD": "AAPL",
        }

        try:
            headers = self.kis.headers.copy()
            headers["tr_id"] = tr_id
            r = self.kis.call_get(url, headers=headers, params=params)
            if r.status_code == 200:
                data = r.json()
                output = data.get("output")
                if output:
                    # 'frcr_ord_psbl_amt1' is usually the orderable amount in foreign currency
                    val = output.get("frcr_ord_psbl_amt1") or output.get(
                        "ovrs_reva_mny1", 0
                    )
                    return float(val)
            return 0.0
        except Exception as e:
            logger.error(f"[US-CASH] Inquiry error: {e}")
            return 0.0

    def get_total_equity(self) -> float:
        """Calculate total equity across Korea and US assets in KRW."""
        # 1. Domestic (KRW)
        # Use tot_evlu_amt from output2 which includes (Stock Value + Cash)
        tr_id = "TTTC8434R"
        if self.kis.is_mock:
            tr_id = "VTTC8434R"

        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "N",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01" if not self.kis.is_mock else "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        dom_total_krw = 0.0
        try:
            headers = self.kis.headers.copy()
            headers["tr_id"] = tr_id
            r = self.kis.call_get(url, headers=headers, params=params)
            if r.status_code == 200:
                data = r.json()
                summary_list = data.get("output2", [])
                if summary_list:
                    dom_total_krw = float(summary_list[0].get("tot_evlu_amt", 0))
        except Exception as e:
            logger.error(f"[TOTAL-EQUITY-KR] Error: {e}")

        # 2. Overseas (in USD)
        us_cash_usd = float(self.get_us_cash())
        us_holdings = self.get_us_balance()
        us_val_usd = sum(float(h.get("frcr_evlu_amt2", 0)) for h in us_holdings)

        # 3. Sum up (Approx exchange rate 1400)
        exch_rate = 1400.0
        total_krw = dom_total_krw + (us_cash_usd + us_val_usd) * exch_rate
        return total_krw
