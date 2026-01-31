import yaml
import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def get_final_report():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # 1. Check US Balance Probing
    res_report = {"us_holdings": [], "cash_usd": 0, "cash_krw": 0, "history": []}

    # Probing US Balance
    tr_id_bal = "VTTS3012R"
    exchanges = ["AMS", "NAS", "NYS"]
    for ex in exchanges:
        params = {
            "CANO": kis.cano,
            "ACNT_PRDT_CD": kis.acnt_prdt_cd,
            "OVRS_EXCG_CD": ex,
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        r = kis.call_get(
            f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-balance",
            headers={**kis.headers, "tr_id": tr_id_bal},
            params=params,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("rt_cd") == "0":
                res_report["us_holdings"].extend(data.get("output1", []))

    # 2. Check US Cash (Buy Possible)
    tr_id_cash = "VTTS3007R"
    params_cash = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "OVRS_EXCG_CD": "NAS",
        "OVRS_ORD_UNPR": "0",
        "ITEM_CD": "AAPL",
    }
    r = kis.call_get(
        f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order",
        headers={**kis.headers, "tr_id": tr_id_cash},
        params=params_cash,
    )
    if r.status_code == 200:
        data = r.json()
        output = data.get("output")
        if output:
            res_report["cash_usd"] = float(output.get("frcr_ord_psbl_amt1") or 0)

    # 3. Check KR Cash
    tr_id_kr = "VTTC8434R"
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
    r = kis.call_get(
        f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-balance",
        headers={**kis.headers, "tr_id": tr_id_kr},
        params=params_kr,
    )
    if r.status_code == 200:
        data = r.json()
        print(
            f"KR Balance Response: rt_cd={data.get('rt_cd')}, msg1={data.get('msg1')}"
        )
        summary = data.get("output2", [])
        if summary:
            res_report["cash_krw"] = int(summary[0].get("dnca_tot_amt") or 0)
    else:
        print(f"KR Balance HTTP Error: {r.status_code}")

    # 4. Check Order History (US)
    tr_id_hist = "VTTS3035R"
    today = datetime.now().strftime("%Y%m%d")
    params_hist = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "ORD_STRT_DT": today,
        "ORD_END_DT": today,
        "SLL_BUY_DVSN_CD": "00",
        "OVRS_EXCG_CD": "ANY",
        "PDNO": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    r = kis.call_get(
        f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-nccs-order",
        headers={**kis.headers, "tr_id": tr_id_hist},
        params=params_hist,
    )
    if r.status_code == 200:
        data = r.json()
        print(
            f"US History Response: rt_cd={data.get('rt_cd')}, msg1={data.get('msg1')}"
        )
        res_report["history"] = data.get("output", [])

    print(json.dumps(res_report, indent=2))


if __name__ == "__main__":
    get_final_report()
