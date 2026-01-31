import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def detailed_probe():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Refresh token
    if os.path.exists(".kis_token_cache_mock.json"):
        os.remove(".kis_token_cache_mock.json")

    kis = KISAPIWrapper(config["kis"])

    test_cases = [
        {
            "desc": "KR Bal - 8 digit",
            "tr": "VTTC8434R",
            "url": "/uapi/domestic-stock/v1/trading/inquire-balance",
            "params": {
                "CANO": "50161248",
                "ACNT_PRDT_CD": "01",
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "N",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
            },
        },
        {
            "desc": "US Bal - 8 digit",
            "tr": "VTTS3012R",
            "url": "/uapi/overseas-stock/v1/trading/inquire-balance",
            "params": {
                "CANO": "50161248",
                "ACNT_PRDT_CD": "01",
                "OVRS_EXCG_CD": "NAS",
                "TR_CRCY_CD": "USD",
            },
        },
        {
            "desc": "US Bal - 02 prod",
            "tr": "VTTS3012R",
            "url": "/uapi/overseas-stock/v1/trading/inquire-balance",
            "params": {
                "CANO": "50161248",
                "ACNT_PRDT_CD": "02",
                "OVRS_EXCG_CD": "NAS",
                "TR_CRCY_CD": "USD",
            },
        },
        {
            "desc": "US Bal - 03 prod",
            "tr": "VTTS3012R",
            "url": "/uapi/overseas-stock/v1/trading/inquire-balance",
            "params": {
                "CANO": "50161248",
                "ACNT_PRDT_CD": "03",
                "OVRS_EXCG_CD": "NAS",
                "TR_CRCY_CD": "USD",
            },
        },
    ]

    for tc in test_cases:
        print(f"\n--- {tc['desc']} ---")
        p = tc["params"].copy()
        # Add required empty fields if not present
        if tc["tr"] == "VTTC8434R":
            p.update({"CTX_AREA_FK100": "", "CTX_AREA_NK100": ""})
        else:
            p.update({"CTX_AREA_FK200": "", "CTX_AREA_NK200": ""})

        print(f"Params: {p}")
        r = kis.call_get(
            f"{kis.base_url}{tc['url']}",
            headers={**kis.headers, "tr_id": tc["tr"]},
            params=p,
        )
        print(f"Status: {r.status_code}")
        try:
            data = r.json()
            print(f"Result: rt_cd={data.get('rt_cd')}, msg1={data.get('msg1')}")
            if data.get("rt_cd") == "0":
                print(
                    f"Success! Data: {json.dumps(data.get('output2', data.get('output1', [])), indent=2, ensure_ascii=False)}"
                )
        except:
            print(f"Raw: {r.text[:200]}")


if __name__ == "__main__":
    detailed_probe()
