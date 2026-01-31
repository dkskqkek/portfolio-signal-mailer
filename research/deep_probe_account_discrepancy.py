import yaml
import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def probe_all_history():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])
    today = datetime.now().strftime("%Y%m%d")

    report = {"us_history": [], "kr_history": [], "us_balance": [], "kr_balance": []}

    # 1. US Order History (VTTS3035R)
    # Trying different combinations for US history
    url_us_hist = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-nccs-order"
    headers_us = {**kis.headers, "tr_id": "VTTS3035R"}
    params_us = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "ORD_STRT_DT": today,
        "ORD_END_DT": today,
        "SLL_BUY_DVSN_CD": "00",
        "OVRS_EXCG_CD": "ANY",
        "PDNO": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    r = kis.call_get(url_us_hist, headers=headers_us, params=params_us)
    if r.status_code == 200:
        report["us_history"] = r.json().get("output", [])

    # 2. KR Order History (VTTC8001R)
    url_kr_hist = f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    headers_kr = {**kis.headers, "tr_id": "VTTC8001R"}
    params_kr = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "INQR_STRT_DT": today,
        "INQR_END_DT": today,
        "SLL_BUY_DVSN_CD": "00",
        "INQR_DVSN": "00",
        "PRCS_DVSN": "00",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    r = kis.call_get(url_kr_hist, headers=headers_kr, params=params_kr)
    if r.status_code == 200:
        report["kr_history"] = r.json().get("output1", [])

    # 3. Check US Balance (VTTS3012R) - Double check with different exchange
    url_us_bal = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
    for ex in ["AMS", "NAS", "NYS", "USA"]:
        headers = {**kis.headers, "tr_id": "VTTS3012R"}
        params = {
            "CANO": kis.cano,
            "ACNT_PRDT_CD": kis.acnt_prdt_cd,
            "OVRS_EXCG_CD": ex,
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        r = kis.call_get(url_us_bal, headers=headers, params=params)
        if r.status_code == 200:
            data = r.json()
            if data.get("rt_cd") == "0":
                report["us_balance"].append({ex: data.get("output1", [])})

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    probe_all_history()
