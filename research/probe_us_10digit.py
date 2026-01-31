import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def probe_us_10digit():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # Try US Balance (VTTS3012R) with 10-digit CANO
    url = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"

    # We set account_no to 5016124801 in config.yaml already
    params = {
        "CANO": kis.cano,  # 5016124801
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,  # 01
        "OVRS_EXCG_CD": "NAS",
        "TR_CRCY_CD": "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": "",
    }

    headers = {**kis.headers, "tr_id": "VTTS3012R"}
    r = kis.call_get(url, headers=headers, params=params)

    print(f"US Balance (10-digit) Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    probe_us_10digit()
