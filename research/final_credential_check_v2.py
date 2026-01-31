import requests
import json
import time


def test_auth_and_balance(app_key, app_secret, cano):
    url_base = "https://openapivts.koreainvestment.com:29443"

    # 1. Auth Test
    auth_url = f"{url_base}/oauth2/tokenP"
    auth_body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret,
    }

    print(f"Testing Auth on {url_base}...")
    r_auth = requests.post(
        auth_url, json=auth_body, headers={"Content-Type": "application/json"}
    )
    print(f"Auth Status: {r_auth.status_code}")
    auth_data = r_auth.json()
    print(f"Auth Response: {json.dumps(auth_data, indent=2, ensure_ascii=False)}")

    token = auth_data.get("access_token")
    if not token:
        print("Failed to obtain token.")
        return

    # 2. Balance Test (Domestic)
    print("\nTesting Balance (Domestic)...")
    bal_url = f"{url_base}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": app_key,
        "appsecret": app_secret,
        "tr_id": "VTTC8434R",
        "custtype": "P",
    }
    # Test both 8-digit and 10-digit for certainty
    for test_cano in [cano, cano + "01"]:
        params = {
            "CANO": test_cano[:8],
            "ACNT_PRDT_CD": test_cano[8:] if len(test_cano) > 8 else "01",
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
        r_bal = requests.get(bal_url, headers=headers, params=params)
        print(f"CANO {test_cano} Balance Status: {r_bal.status_code}")
        try:
            bal_data = r_bal.json()
            print(f"Response: {bal_data.get('rt_cd')} - {bal_data.get('msg1')}")
            if bal_data.get("rt_cd") == "0":
                print(f"Success! Output2: {bal_data.get('output2')}")
        except:
            print(f"Raw: {r_bal.text[:100]}")


if __name__ == "__main__":
    # Current config values
    app_key = "PSKqKzfa2FVwftOotO3WfI2CuBtQ3WtExJX1"
    app_secret = "gK57cT3fjgdzo7dvIyL6qsOffNTuAS/L6FrYdhzM6dz8glNSR5PdBg4uMeTjBcqN3lbDZamGWKobSsekw6P6i2CmQxm4NcPCCSbLqDZLYPvdd6ShlAuDYI4j8nt5g9XtmQLnQ3GJ6fUQpFcAGxGvMd1ThXPLDmTv/z5C4r7KziQ+Hawkjv8="
    cano = "50161248"

    test_auth_and_balance(app_key, app_secret, cano)
