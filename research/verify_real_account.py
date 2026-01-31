import requests
import json


def get_real_balance(app_key, app_secret, cano):
    # 1. Get Token
    auth_url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"
    auth_body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret,
    }
    r_auth = requests.post(
        auth_url, json=auth_body, headers={"Content-Type": "application/json"}
    )
    token = r_auth.json().get("access_token")

    # 2. Get Balance (Real TR: TTTC8434R)
    url = "https://openapi.koreainvestment.com:9443/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": app_key,
        "appsecret": app_secret,
        "tr_id": "TTTC8434R",
        "custtype": "P",
    }
    params = {
        "CANO": cano,
        "ACNT_PRDT_CD": "01",
        "AFHR_FLG": "N",
        "OFR_FLG": "N",
        "INQR_DVSN": "01",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FRLI_GVOFF_SYS_DVSN": "00",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    r = requests.get(url, headers=headers, params=params)
    return r.status_code, r.text


key = "PSKqKzfa2FVwftOotO3WfI2CuBtQ3WtExJX1"
sec = "gK57cT3fjgdzo7dvIyL6qsOffNTuAS/L6FrYdhzM6dz8glNSR5PdBg4uMeTjBcqN3lbDZamGWKobSsekw6P6i2CmQxm4NcPCCSbLqDZLYPvdd6ShlAuDYI4j8nt5g9XtmQLnQ3GJ6fUQpFcAGxGvMd1ThXPLDmTv/z5C4r7KziQ+Hawkjv8="
cano = "50161248"

print(f"Checking Real Balance for {cano}:")
st, txt = get_real_balance(key, sec, cano)
print(f"Status: {st}, Response: {txt}")
