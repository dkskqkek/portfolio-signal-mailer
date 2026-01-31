import requests
import json
import yaml
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def audit_us_orders():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    print(f"--- DETAILED US ORDER AUDIT for {kis.cano} ---")

    # Try multiple CANO formats for history
    cano_formats = [kis.cano, kis.cano + "01"]

    for cano in cano_formats:
        print(f"\n>> Testing CANO: {cano}")

        # 1. US Order History (VTTS3035R)
        url = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-ccld"
        params = {
            "CANO": cano[:8],
            "ACNT_PRDT_CD": cano[8:] if len(cano) > 8 else "01",
            "PDNO": "%",
            "ORD_STRT_DT": "20260130",
            "ORD_END_DT": "20260131",
            "SHTN_PDNO": "",
            "ORD_DVSN": "00",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": "",
        }
        headers = {**kis.headers, "tr_id": "VTTS3035R"}
        r = requests.get(url, headers=headers, params=params)
        print(f"History Result (Status {r.status_code}):")
        try:
            data = r.json()
            history = data.get("output1", [])
            print(f"  Found {len(history)} orders.")
            for o in history[:10]:
                print(
                    f"    - {o.get('pdno')} | {o.get('sll_buy_dvsn_cd_name')} | Qty: {o.get('ft_ord_qty')} | CCLD: {o.get('ft_ccld_qty')} | Stat: {o.get('prcs_stat_name')}"
                )
        except:
            print(f"  Raw: {r.text[:200]}")

        # 2. Check US Balance/Cash (VTTS3007R - PSBL ORDER)
        print(f"\n>> Checking US Buy Possible (VTTS3007R) for {cano}")
        url_psbl = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order"
        psbl_params = {
            "CANO": cano[:8],
            "ACNT_PRDT_CD": cano[8:] if len(cano) > 8 else "01",
            "OVRS_EXCG_CD": "NAS",
            "OVRS_ORD_UNPR": "0",
            "ITEM_CD": "AAPL",
        }
        headers["tr_id"] = "VTTS3007R"
        r_psbl = requests.get(url_psbl, headers=headers, params=psbl_params)
        try:
            psbl_data = r_psbl.json()
            output = psbl_data.get("output", {})
            print(f"  Available USD: ${output.get('frcr_ord_psbl_amt1')}")
            print(
                f"  Frozen USD? (Check fields like 'ovrs_reva_mny1'): {output.get('ovrs_reva_mny1')}"
            )
        except:
            print(f"  Error: {r_psbl.text[:100]}")


if __name__ == "__main__":
    audit_us_orders()
