import requests
import json
import time


def probe_all_products(app_key, app_secret, cano):
    # 1. Get Token
    auth_url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"
    auth_body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret,
    }
    r_auth = requests.post(
        auth_url, json=auth_body, headers={"Content-Type": "application/json"}
    )
    token = r_auth.json().get("access_token")

    results = []
    for i in range(1, 7):
        prdt_cd = f"{i:02d}"
        url = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/trading/inquire-balance"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": app_key,
            "appsecret": app_secret,
            "tr_id": "VTTC8434R",
            "custtype": "P",
        }
        params = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt_cd,
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
        r = requests.get(url, headers=headers, params=params)
        data = r.json()
        results.append(
            {"prdt_cd": prdt_cd, "rt_cd": data.get("rt_cd"), "msg1": data.get("msg1")}
        )
        print(f"Product {prdt_cd}: {data.get('msg1')}")

    return results


if __name__ == "__main__":
    key = "PSKqKzfa2FVwftOotO3WfI2CuBtQ3WtExJX1"
    sec = "gK57cT3fjgdzo7dvIyL6qsOffNTuAS/L6FrYdhzM6dz8glNSR5PdBg4uMeTjBcqN3lbDZamGWKobSsekw6P6i2CmQxm4NcPCCSbLqDZLYPvdd6ShlAuDYI4j8nt5g9XtmQLnQ3GJ6fUQpFcAGxGvMd1ThXPLDmTv/z5C4r7KziQ+Hawkjv8="
    cano = "50161248"
    probe_all_products(key, sec, cano)
