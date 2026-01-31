# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys
import json

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper

logging.basicConfig(level=logging.INFO)


def check_order_history():
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    kis = KISAPIWrapper(config["kis"])

    # Inquire Daily Conclusion (Order History)
    url = f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    tr_id = "VTTC8001R"  # Mock

    headers = kis.headers.copy()
    headers["tr_id"] = tr_id

    params = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "INQR_STRT_DT": "20260129",
        "INQR_END_DT": "20260129",
        "SLL_BUY_DVSN_CD": "00",
        "INQR_DVSN": "00",
        "PDNO": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    import requests

    r = requests.get(url, headers=headers, params=params, timeout=10)
    print(f"Status Code: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        output = data.get("output1", [])
        print(f"\n--- 오늘의 주문 내역 (총 {len(output)}건) ---")

        # '테스' (095610) 필터링 또는 전체 출력
        for item in output:
            ticker = item.get("pdno")
            name = item.get("prdt_name")
            status = item.get("rmnd_qty")  # 잔량 (0이면 전량 체결)
            msg = item.get("ord_stat_name")  # 취소/거부 등 상태
            print(
                f"[{ticker}] {name} | 수량: {item.get('ord_qty')} | 체결: {item.get('tot_ccld_qty')} | 상태: {msg}"
            )

        if not output:
            print("주문 내역이 없습니다.")
    else:
        print(f"Response: {r.text}")


if __name__ == "__main__":
    check_order_history()
