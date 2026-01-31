import requests
import json
import os


def test_combination(env_name, url, app_key, app_secret, tests):
    # 1. Get Token
    auth_url = f"{url}/oauth2/tokenP"
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
    for cano, prdt_cd, tr_id, endpoint in tests:
        full_url = f"{url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": app_key,
            "appsecret": app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        params = {"CANO": cano, "ACNT_PRDT_CD": prdt_cd}
        # Add required dummy/extra fields for KR balance
        if "domestic" in endpoint:
            params.update(
                {
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
            )
        else:
            params.update(
                {
                    "OVRS_EXCG_CD": "NAS",
                    "TR_CRCY_CD": "USD",
                    "CTX_AREA_FK200": "",
                    "CTX_AREA_NK200": "",
                }
            )

        r = requests.get(full_url, headers=headers, params=params)
        try:
            data = r.json()
            results.append(
                {
                    "cano": cano,
                    "prdt": prdt_cd,
                    "tr": tr_id,
                    "rt_cd": data.get("rt_cd"),
                    "msg": data.get("msg1"),
                    "cash": data.get("output2", [{}])[0].get("dnca_tot_amt")
                    if data.get("output2")
                    else "N/A",
                }
            )
        except:
            results.append(
                {
                    "cano": cano,
                    "prdt": prdt_cd,
                    "tr": tr_id,
                    "err": "Non-JSON",
                    "txt": r.text[:50],
                }
            )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    key = "PSKqKzfa2FVwftOotO3WfI2CuBtQ3WtExJX1"
    sec = "gK57cT3fjgdzo7dvIyL6qsOffNTuAS/L6FrYdhzM6dz8glNSR5PdBg4uMeTjBcqN3lbDZamGWKobSsekw6P6i2CmQxm4NcPCCSbLqDZLYPvdd6ShlAuDYI4j8nt5g9XtmQLnQ3GJ6fUQpFcAGxGvMd1ThXPLDmTv/z5C4r7KziQ+Hawkjv8="

    url = "https://openapivts.koreainvestment.com:29443"
    tests = [
        # Domestic (KR) Balance
        (
            "50161248",
            "01",
            "VTTC8434R",
            "/uapi/domestic-stock/v1/trading/inquire-balance",
        ),
        (
            "5016124801",
            "01",
            "VTTC8434R",
            "/uapi/domestic-stock/v1/trading/inquire-balance",
        ),
        (
            "50161248",
            "02",
            "VTTC8434R",
            "/uapi/domestic-stock/v1/trading/inquire-balance",
        ),
        # Overseas (US) Balance
        (
            "50161248",
            "01",
            "VTTS3012R",
            "/uapi/overseas-stock/v1/trading/inquire-balance",
        ),
        (
            "5016124801",
            "01",
            "VTTS3012R",
            "/uapi/overseas-stock/v1/trading/inquire-balance",
        ),
    ]

    test_combination("MOCK", url, key, sec, tests)
