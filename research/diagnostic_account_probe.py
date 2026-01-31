import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def diagnostic_account_probe():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # Try multiple CANO formats for the new account
    cano_variants = ["50161248", "5016124801"]
    results = []

    for cano in cano_variants:
        # KR Balance
        tr_id_bal = "VTTC8434R"
        params = {
            "CANO": cano[:8],
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
        if len(cano) == 10:
            params["CANO"] = cano[:8]
            params["ACNT_PRDT_CD"] = cano[8:]

        r = kis.call_get(
            f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-balance",
            headers={**kis.headers, "tr_id": tr_id_bal},
            params=params,
        )

        try:
            data = r.json()
            results.append(
                {
                    "cano": cano,
                    "type": "KR_BAL",
                    "rt_cd": data.get("rt_cd"),
                    "msg1": data.get("msg1"),
                    "cash": data.get("output2", [{}])[0].get("dnca_tot_amt")
                    if data.get("output2")
                    else 0,
                }
            )
        except:
            results.append(
                {
                    "cano": cano,
                    "type": "KR_BAL",
                    "error": "Non-JSON",
                    "text": r.text[:100],
                }
            )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    diagnostic_account_probe()
