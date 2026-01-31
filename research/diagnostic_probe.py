import yaml
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def diagnostic_probe():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # 1. Try Inquire Balance (Known path)
    url_bal = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
    headers_bal = {**kis.headers, "tr_id": "VTTS3012R"}
    params_bal = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "OVRS_EXCG_CD": "NAS",
        "TR_CRCY_CD": "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": "",
    }
    r_bal = kis.call_get(url_bal, headers=headers_bal, params=params_bal)
    print(f"Balance Status: {r_bal.status_code}")
    print(f"Balance Response: {r_bal.text}")

    # 2. Try PSBL Order with different path variant
    url_psbl = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order"
    headers_psbl = {**kis.headers, "tr_id": "VTTS3007R"}
    params_psbl = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "OVRS_EXCG_CD": "NAS",
        "OVRS_ORD_UNPR": "0",
        "ITEM_CD": "AAPL",
    }
    r_psbl = kis.call_get(url_psbl, headers=headers_psbl, params=params_psbl)
    print(f"PSBL Status: {r_psbl.status_code}")

    # 3. Try Price (Verification that base_url is OK)
    url_price = f"{kis.base_url}/uapi/overseas-price/v1/quotations/price"
    headers_price = {**kis.headers, "tr_id": "HHDFS00000300"}
    params_price = {"AUTH": "", "EXCD": "NAS", "SYMB": "AAPL"}
    r_price = kis.call_get(url_price, headers=headers_price, params=params_price)
    print(f"Price Status: {r_price.status_code}")


if __name__ == "__main__":
    diagnostic_probe()
