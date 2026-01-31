import requests
import json
import yaml
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def diagnostic_us_verify():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    print(f"--- DIAGNOSTIC VERIFY for {kis.cano} ---")

    # 1. Check KRW Balance (Domestic - VTTC8434R)
    # This should show if cash was frozen for integrated funds
    url_kr = f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
    params_kr = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": "01",
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "N",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",  # 01: Include previous day? No.
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    headers_kr = {**kis.headers, "tr_id": "VTTC8434R"}
    r_kr = requests.get(url_kr, headers=headers_kr, params=params_kr)
    print("\nDomestic (KR) Balance Status:")
    try:
        data = r_kr.json()
        out2 = data.get("output2", [{}])[0]
        print(f"  Total Cash: ₩{out2.get('dnca_tot_amt')}")
        print(
            f"  Frozen/Buy Possible: {out2.get('prvs_rcdl_exrt')} (Check for changes)"
        )
        # Look for 'asst_icdc_amt' or similar for "Integrated Fund" usage
        print(f"  Summary Item: {out2}")
    except:
        print(f"  Error: {r_kr.text[:100]}")

    # 2. Probe various US History paths
    paths = [
        "/uapi/overseas-stock/v1/trading/inquire-ccld",
        "/uapi/overseas-stock/v1/trading/inquire-psbl-order",  # Buy Possible
        "/uapi/overseas-stock/v1/trading/inquire-balance",  # Balance
    ]

    print("\nProbing US Mock Paths:")
    for path in paths:
        full_url = f"{kis.base_url}{path}"
        # Test with VTTS3035R for history, VTTS3007R for PSBL, VTTS3012R for Balance
        tr_id = (
            "VTTS3035R"
            if "ccld" in path
            else ("VTTS3007R" if "psbl" in path else "VTTS3012R")
        )
        headers = {**kis.headers, "tr_id": tr_id}

        # Generic params
        params = {
            "CANO": kis.cano,
            "ACNT_PRDT_CD": "01",
            "OVRS_EXCG_CD": "NAS",
            "PDNO": "%",
            "ORD_STRT_DT": "20260130",
            "ORD_END_DT": "20260131",
            "SHTN_PDNO": "",
            "ORD_DVSN": "00",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
            "ITEM_CD": "AAPL",
            "OVRS_ORD_UNPR": "0",
            "TR_CRCY_CD": "USD",
        }

        r = requests.get(full_url, headers=headers, params=params)
        print(f"  {path} ({tr_id}) -> Status: {r.status_code}")
        if r.status_code == 200:
            print(f"    SUCCESS! Data: {r.text[:200]}")


if __name__ == "__main__":
    diagnostic_us_verify()
