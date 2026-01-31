import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def probe_us_capability():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # 1. Price check (should work)
    price = kis.get_us_current_price("BIL", "AMS")
    print(f"Price of BIL: {price}")

    # 2. US Buy Possible (VTTS3007R)
    url = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order"
    params = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "OVRS_EXCG_CD": "NAS",
        "OVRS_ORD_UNPR": "0",
        "ITEM_CD": "AAPL",
    }
    r = kis.call_get(url, headers={**kis.headers, "tr_id": "VTTS3007R"}, params=params)
    print("US PSBL Order Response:")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    probe_us_capability()
