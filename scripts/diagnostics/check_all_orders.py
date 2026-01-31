# -*- coding: utf-8 -*-
import yaml
import os
import sys
import json
import requests

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def check_all_orders():
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    kis = KISAPIWrapper(config["kis"])

    url = f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    tr_id = "VTTC8001R"

    headers = kis.headers.copy()
    headers["tr_id"] = tr_id

    params = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "INQR_STRT_DT": "20260129",
        "INQR_END_DT": "20260129",
        "SLL_BUY_DVSN_CD": "00",
        "INQR_DVSN": "00",  # All
        "PDNO": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code == 200:
        data = r.json()
        print(f"Return Code: {data.get('rt_cd')}")
        print(f"Message: {data.get('msg1')}")

        output = data.get("output1", [])
        print(f"\n--- 전체 주문 내역 ({len(output)}건) ---")
        for i, item in enumerate(output, 1):
            print(
                f"{i}. [{item.get('pdno')}] {item.get('prdt_name')} | 주문수량: {item.get('ord_qty')} | 체결수량: {item.get('tot_ccld_qty')} | 상태: {item.get('ord_unpr')} / {item.get('ord_stat_name')}"
            )
            # Check for rejection or specific messages
            if item.get("ord_stat_name") or item.get("can_mod_stat_name"):
                print(
                    f"   ㄴ 상세: {item.get('ord_stat_name')} | {item.get('can_mod_stat_name')}"
                )


if __name__ == "__main__":
    check_all_orders()
