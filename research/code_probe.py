import yaml
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def code_probe():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    account_no = "50161248"
    tr_id = "VTTC8434R"  # KR Balance

    for i in range(1, 10):
        prdt_cd = f"{i:02d}"
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

        try:
            data = r.json()
            if data.get("rt_cd") == "0":
                print(f"!!! SUCCESS for {account_no}-{prdt_cd} !!!")
                print(json.dumps(data, indent=2, ensure_ascii=False))
                return
            else:
                print(f"Code {prdt_cd}: {data.get('msg1')}")
        except:
            print(f"Code {prdt_cd}: Non-JSON or error {r.status_code}")


if __name__ == "__main__":
    import json

    code_probe()
