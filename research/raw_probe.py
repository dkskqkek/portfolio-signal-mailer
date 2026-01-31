import yaml
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def raw_probe():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    account_no = "50161248"
    tr_id = "VTTC8434R"  # KR Balance

    for prdt_cd in ["01", "02"]:
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
            headers={**kis.headers, "tr_id": tr_id},
            params=params,
        )

        print(f"--- Probe Result for {account_no}-{prdt_cd} ---")
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text}")


if __name__ == "__main__":
    raw_probe()
