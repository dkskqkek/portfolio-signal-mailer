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


def diagnose_tess():
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    kis = KISAPIWrapper(config["kis"])
    ticker = "095610"

    # 1. Check Current Status
    print(f"\n--- [{ticker}] 테스 현재 상태 ---")
    data = kis.get_current_price(ticker)
    if data:
        print(f"종목명: {data.get('rprs_mrkt_kor_name')}")
        print(f"현재가: {data.get('stck_prpr')}")
        print(f"종목상태코드: {data.get('iscd_stat_cls_code')}")
        print(f"증거금률: {data.get('marg_rate')}")
        print(f"거래정지여부: {data.get('tr_stop_yn')}")
        print(f"관리종목여부: {data.get('mang_issu_cls_code')}")

    # 2. Check Order History (Alternative TR_ID if any)
    # let's try inquire-psbl-order (주문가능조회) to see if it's restricted
    print(f"\n--- [{ticker}] 주문 가능 여부 조회 ---")
    url = f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
    tr_id = "TTTC8908R" if not kis.is_mock else "VTTC8908R"
    headers = kis.headers.copy()
    headers["tr_id"] = tr_id
    params = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "PDNO": ticker,
        "ORD_UNPR": "21750",  # current price approx
        "ORD_DVSN": "01",  # Market
        "CMA_EVLU_AMT_ICLD_YN": "N",
        "OVRS_ICLD_YN": "N",
    }

    import requests

    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code == 200:
        res = r.json()
        print(f"주문 가능 금액: {res.get('output', {}).get('nrcy_buy_psbl_amt')}")
        print(f"주문 가능 수량: {res.get('output', {}).get('nrcy_buy_psbl_qty')}")
        if res.get("rt_cd") != "0":
            print(f"비고: {res.get('msg1')}")

    # 3. Last attempt: check for TR_ID VTTC8001R with different params
    print(f"\n--- 전종목 주문 내역 재조회 ---")
    url_ccld = f"{kis.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    h_ccld = kis.headers.copy()
    h_ccld["tr_id"] = "VTTC8001R"
    p_ccld = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "INQR_STRT_DT": "20260129",
        "INQR_END_DT": "20260129",
        "SLL_BUY_DVSN_CD": "00",
        "INQR_DVSN": "01",  # Executed
        "PDNO": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    r2 = requests.get(url_ccld, headers=h_ccld, params=p_ccld, timeout=10)
    if r2.status_code == 200:
        out1 = r2.json().get("output1", [])
        print(f"체결 내역: {len(out1)}건")
        for x in out1:
            print(f" - {x.get('prdt_name')} ({x.get('pdno')}): {x.get('ord_qty')}주")

    p_ccld["INQR_DVSN"] = "02"  # Unexecuted
    r3 = requests.get(url_ccld, headers=h_ccld, params=p_ccld, timeout=10)
    if r3.status_code == 200:
        out2 = r3.json().get("output1", [])
        print(f"미체결 내역: {len(out2)}건")
        for x in out2:
            print(
                f" - {x.get('prdt_name')} ({x.get('pdno')}): {x.get('ord_qty')}주 | 사유: {x.get('ord_stat_name')}"
            )


if __name__ == "__main__":
    diagnose_tess()
