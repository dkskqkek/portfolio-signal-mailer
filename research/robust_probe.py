import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def robust_probe():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    account_no = "50161248"
    results = []

    # Try common product codes
    for prdt_cd in ["01", "02"]:
        tr_id_bal = "VTTC8434R"  # KR Balance
        params = {
            "CANO": account_no,
            "ACNT_PRDT_CD": prdt_cd,
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
            headers={**kis.headers, "tr_id": tr_id_bal},
            params=params,
        )

        try:
            data = r.json()
            results.append({"prdt_cd": prdt_cd, "status": r.status_code, "data": data})
        except:
            results.append(
                {
                    "prdt_cd": prdt_cd,
                    "status": r.status_code,
                    "error": "Non-JSON response",
                    "text": r.text[:100],
                }
            )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    robust_probe()
