import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def dual_probe():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    accounts = ["50161248", "50160736"]
    results = {}

    for acct in accounts:
        tr_id = "VTTC8434R"  # KR Balance
        params = {
            "CANO": acct,
            "ACNT_PRDT_CD": "01",
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
        r = kis.call_get(
            f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-balance",
            headers={**kis.headers, "tr_id": tr_id},
            params=params,
        )

        try:
            results[acct] = r.json()
        except:
            results[acct] = {"error": "Non-JSON", "text": r.text[:100]}

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    dual_probe()
