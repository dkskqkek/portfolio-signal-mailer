import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def debug_us_balance():
    # Load Config
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # Check US Balance (US Mock: VTTS3012R)
    url_us = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
    headers_us = kis.headers.copy()
    headers_us["tr_id"] = "VTTS3012R"

    params_us = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "OVRS_EXCG_CD": "NAS",
        "TR_CRCY_CD": "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": "",
    }

    res_us = kis.call_get(url_us, headers=headers_us, params=params_us)
    if res_us.status_code == 200:
        data = res_us.json()
        print(f"rt_cd: {data.get('rt_cd')}")
        print(f"msg1: {data.get('msg1')}")

        output1 = data.get("output1", [])
        if output1:
            print("Holdings in output1:")
            print(json.dumps(output1, indent=2))
        else:
            print("output1 is empty (No holdings).")

        output2 = data.get("output2", [])
        if output2:
            print("Summary in output2:")
            print(
                json.dumps(
                    output2[0] if isinstance(output2, list) else output2, indent=2
                )
            )
    else:
        print(f"API Error: {res_us.status_code}")


if __name__ == "__main__":
    debug_us_balance()
