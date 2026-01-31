import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def probe_integrated_funds():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # Try Integrated Funds inquiry (Real/Mock might share this or have variants)
    # Using the new account number
    account_no = "50161248"
    tr_id = "CTRP6010R"

    url = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order"  # Use psbl-order as a fallback probe

    # Actually, CTRP6010R is for Real. For Mock, it's often VTTS3007R for USD buy power.
    # Let's try VTTS3007R which we used successfully before.

    params = {
        "CANO": account_no,
        "ACNT_PRDT_CD": "01",
        "OVRS_EXCG_CD": "NAS",
        "OVRS_ORD_UNPR": "0",
        "ITEM_CD": "AAPL",
    }

    headers = {**kis.headers, "tr_id": "VTTS3007R"}
    r = kis.call_get(
        f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order",
        headers=headers,
        params=params,
    )

    print(f"Probe Result for {account_no}-01:")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    probe_integrated_funds()
