import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def probe_account_format():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # Try 8-digit and 10-digit CANO
    account_variants = [
        {"CANO": "50161248", "ACNT_PRDT_CD": "01"},
        {"CANO": "5016124801", "ACNT_PRDT_CD": "01"},
    ]

    tr_id_kr = "VTTC8434R"
    url_kr = f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"

    results = []

    for variant in account_variants:
        params = {
            "CANO": variant["CANO"],
            "ACNT_PRDT_CD": variant["ACNT_PRDT_CD"],
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
            url_kr, headers={**kis.headers, "tr_id": tr_id_kr}, params=params
        )
        data = r.json()
        results.append(
            {
                "variant": variant,
                "rt_cd": data.get("rt_cd"),
                "msg1": data.get("msg1"),
                "cash": data.get("output2", [{}])[0].get("dnca_tot_amt")
                if data.get("output2")
                else 0,
            }
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    probe_account_format()
