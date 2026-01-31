import yaml
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def raw_us_probe():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # VTTS3007R is for US Buy Possible
    url = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order"
    params = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "OVRS_EXCG_CD": "NAS",
        "OVRS_ORD_UNPR": "0",
        "ITEM_CD": "AAPL",
    }
    r = kis.call_get(url, headers={**kis.headers, "tr_id": "VTTS3007R"}, params=params)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")


if __name__ == "__main__":
    raw_us_probe()
