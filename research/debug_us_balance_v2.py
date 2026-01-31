import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def debug_us_balance_v2():
    # Load Config
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # Try multiple variants of the Balance TR
    tr_id = "VTTS3012R"
    url = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"

    # KIS Mock often requires specific combinations
    variants = [
        {"OVRS_EXCG_CD": "NAS", "TR_CRCY_CD": "USD"},
        {"OVRS_EXCG_CD": "NYS", "TR_CRCY_CD": "USD"},
        {"OVRS_EXCG_CD": "AMS", "TR_CRCY_CD": "USD"},
    ]

    for v in variants:
        params = {
            "CANO": kis.cano,
            "ACNT_PRDT_CD": kis.acnt_prdt_cd,
            "OVRS_EXCG_CD": v["OVRS_EXCG_CD"],
            "TR_CRCY_CD": v["TR_CRCY_CD"],
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        res = kis.call_get(url, headers={**kis.headers, "tr_id": tr_id}, params=params)
        print(f"--- Probe {v['OVRS_EXCG_CD']} ---")
        if res.status_code == 200:
            data = res.json()
            print(f"rt_cd: {data.get('rt_cd')}, msg1: {data.get('msg1')}")
            for h in data.get("output1", []):
                print(
                    f"  HOLDING: {h.get('ovrs_pdno')} | Qty: {h.get('ovrs_cblc_qty')}"
                )
        else:
            print(f"Error: {res.status_code}")


if __name__ == "__main__":
    debug_us_balance_v2()
