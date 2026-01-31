import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def debug_kr_balance():
    # Load Config
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # Check KR Cash (KR Mock: VTTC8434R)
    url_kr = f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers_kr = kis.headers.copy()
    headers_kr["tr_id"] = "VTTC8434R"

    params_kr = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
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

    res_kr = kis.call_get(url_kr, headers=headers_kr, params=params_kr)
    if res_kr.status_code == 200:
        data = res_kr.json()
        print(f"rt_cd: {data.get('rt_cd')}")
        print(f"msg1: {data.get('msg1')}")

        output2 = data.get("output2", [])
        if output2:
            print("First item in output2:")
            for k, v in output2[0].items():
                print(f"  {k}: {v}")
        else:
            print("output2 is empty.")
            print(f"Full Data: {json.dumps(data, indent=2)}")
    else:
        print(f"API Error: {res_kr.status_code}")


if __name__ == "__main__":
    debug_kr_balance()
