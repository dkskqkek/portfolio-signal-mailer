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


def check_all_orders_fix():
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    kis = KISAPIWrapper(config["kis"])

    url = f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    tr_id = "VTTC0081R"  # Mock Inner 3 Months

    headers = kis.headers.copy()
    headers["tr_id"] = tr_id

    params = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "INQR_STRT_DT": "20260129",
        "INQR_END_DT": "20260129",
        "SLL_BUY_DVSN_CD": "00",
        "PDNO": "",
        "CCLD_DVSN": "00",  # 전체
        "INQR_DVSN": "00",  # 역순
        "INQR_DVSN_3": "00",  # 전체
        "ORD_GNO_BRNO": "",
        "ODNO": "",
        "INQR_DVSN_1": "",
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
            name = item.get("prdt_name", "Unknown")
            ticker = item.get("pdno", "000000")
            qty = item.get("ord_qty")
            ccld_qty = item.get("tot_ccld_qty")
            stat = item.get("ord_stat_name")
            print(
                f"{i}. [{ticker}] {name} | 주문: {qty} | 체결: {ccld_qty} | 상태: {stat}"
            )
            if int(ccld_qty or 0) == 0:
                print(f"   ㄴ 상세 상태: {item.get('can_mod_stat_name')}")


if __name__ == "__main__":
    check_all_orders_fix()
