import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def get_holdings():
    # Load Config
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # KIS TR ID for US Portfolio Balance/Holdings: HCTRT2001R
    url = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-present-balance"
    headers = kis.headers.copy()
    headers["tr_id"] = "HCTRT2001R"

    params = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "WCRC_FRCR_DVSN_CD": "02",  # 02 for USD/Foreign
        "NATN_CD": "840",  # 840 for US
        "TR_P_CTR_CD": "",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": "",
    }

    response = kis.call_get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    get_holdings()
