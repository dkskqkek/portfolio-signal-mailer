import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def get_actual_mock_status():
    # Load Config
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # Check US Balance (US Mock: VTTS3012R)
    url_us = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
    headers_us = kis.headers.copy()
    headers_us["tr_id"] = "VTTS3012R"

    params_us = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "OVRS_EXCG_CD": "NAS",
        "TR_CRCY_CD": "USD",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": "",
    }

    print("--- [US Mock Account Balance] ---")
    res_us = kis.call_get(url_us, headers=headers_us, params=params_us)
    if res_us.status_code == 200:
        data = res_us.json()
        holdings = data.get("output1", [])
        summary = data.get("output2", {})
        if not holdings:
            print("Status: No overseas holdings found.")
        else:
            for h in holdings:
                print(
                    f"  {h['ovrs_pdno']} ({h['ovrs_item_name']}): {h['ovrs_cblc_qty']} shares | Val: ${h['frcr_evlu_amt2']}"
                )

        # Summary
        if isinstance(summary, list) and summary:
            summary = summary[0]
        print(f"Total Valuation: ${summary.get('tot_evlu_pamt', '0')}")
    else:
        print(f"API Error: {res_us.status_code}")

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

    print("\n--- [KR Mock Account Balance] ---")
    res_kr = kis.call_get(url_kr, headers=headers_kr, params=params_kr)
    if res_kr.status_code == 200:
        data = res_kr.json()
        summary = data.get("output2", [])
        if summary:
            s = summary[0]
            print(f"Cash (KRW): ₩{int(s.get('dnca_tot_amt', 0)):,}")
            print(f"Total Asset: ₩{int(s.get('tot_evlu_amt', 0)):,}")
        else:
            print("Status: No summary data found.")


if __name__ == "__main__":
    get_actual_mock_status()
