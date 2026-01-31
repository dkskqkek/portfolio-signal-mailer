import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor


def verify_rebalance_results():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)

    print("--- POST-REBALANCE VERIFICATION ---")

    # 1. Check US Holdings
    holdings = executor.get_us_balance()
    print(f"\nCurrent US Holdings: {len(holdings)} items")
    for h in holdings:
        print(
            f"  - {h.get('ovrs_pdno')}: {h.get('ovrs_cblc_qty')} shares (Val: ${h.get('frcr_evlu_amt2')})"
        )

    # 2. Check Order History (US)
    # TR: TTTS3035R (Real), VTTS3035R (Mock)
    url = f"{kis.base_url}/uapi/overseas-stock/v1/trading/inquire-ccld"
    params = {
        "CANO": kis.cano,
        "ACNT_PRDT_CD": kis.acnt_prdt_cd,
        "PDNO": "%",
        "ORD_STRT_DT": "20260130",  # Assuming today/yesterday
        "ORD_END_DT": "20260131",
        "SHTN_PDNO": "",
        "ORD_DVSN": "00",
        "CTX_AREA_FK200": "",
        "CTX_AREA_NK200": "",
    }
    headers = {**kis.headers, "tr_id": "VTTS3035R"}
    r = kis.call_get(url, headers=headers, params=params)
    print("\nOrder History (Mock US):")
    try:
        history = r.json().get("output1", [])
        print(f"Total Orders Found: {len(history)}")
        for order in history[:5]:
            print(
                f"  - {order.get('pdno')} | {order.get('sll_buy_dvsn_cd_name')} | Qty: {order.get('ft_ord_qty')} | CCLD: {order.get('ft_ccld_qty')} | Status: {order.get('prcs_stat_name')}"
            )
    except:
        print(f"Error fetching history: {r.text[:200]}")


if __name__ == "__main__":
    verify_rebalance_results()
