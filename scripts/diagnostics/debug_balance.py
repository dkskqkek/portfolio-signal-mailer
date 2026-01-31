# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper

logging.basicConfig(level=logging.INFO)


def debug_balance():
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    kis = KISAPIWrapper(config["kis"])

    tr_id = "VTTC8434R"  # Mock balance
    url = f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
    params = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
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

    headers = kis.headers.copy()
    headers["tr_id"] = tr_id

    import requests

    r = requests.get(url, headers=headers, params=params, timeout=10)
    print(f"Status Code: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print("\n--- Output2 (Summary) ---")
        print(json.dumps(data.get("output2", []), indent=2, ensure_ascii=False))

        print("\n--- Output1 (Holdings) ---")
        print(json.dumps(data.get("output1", []), indent=2, ensure_ascii=False))

        if data.get("rt_cd") != "0":
            print(f"Error: {data.get('msg1')}")
    else:
        print(f"Response: {r.text}")


if __name__ == "__main__":
    import json

    debug_balance()
