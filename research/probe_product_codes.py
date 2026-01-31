import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def probe_product_codes():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    account_no = "50161248"
    results = []

    # Try common product codes
    for prdt_cd in ["01", "02", "03", "04", "05", "06"]:
        tr_id_bal = "VTTS3007R"  # US Buy Possible (often more lenient with ACNO check)
        params = {
            "CANO": account_no,
            "ACNT_PRDT_CD": prdt_cd,
            "OVRS_EXCG_CD": "NAS",
            "OVRS_ORD_UNPR": "0",
            "ITEM_CD": "AAPL",
        }
        r = kis.call_get(
            f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order",
            headers={**kis.headers, "tr_id": tr_id_bal},
            params=params,
        )
        data = r.json()
        results.append(
            {"prdt_cd": prdt_cd, "rt_cd": data.get("rt_cd"), "msg1": data.get("msg1")}
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    probe_product_codes()
