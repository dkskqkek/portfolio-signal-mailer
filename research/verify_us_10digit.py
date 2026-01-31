import requests
import json
import yaml
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def verify_us_10digit():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # Try 10-digit CANO for US Balance
    url = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
    cano_10 = "5016124801"

    params = {
        "CANO": "50161248",
        "ACNT_PRDT_CD": "01",
        "OVRS_EXCG_CD": "NAS",
        "TR_CRCY_CD": "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": "",
    }

    headers = {**kis.headers, "tr_id": "VTTS3012R"}
    r = requests.get(url, headers=headers, params=params)
    print(f"US Balance (8-digit) Status: {r.status_code}")
    print(f"Response: {r.text[:200]}")

    # Try 10-digit
    # KIS Mock often uses the 10-digit number as the CANO itself in some TRs
    params["CANO"] = "5016124801"
    r2 = requests.get(url, headers=headers, params=params)
    print(f"\nUS Balance (10-digit) Status: {r2.status_code}")
    print(f"Response: {r2.text[:200]}")


if __name__ == "__main__":
    verify_us_10digit()
