import requests
import json
import os


def test_combination(env_name, url, app_key, app_secret, cano, prdt_cd="01"):
    print(f"\n--- Testing Environment: {env_name} ---")
    print(f"URL: {url}")

    # 1. Get Token
    auth_url = f"{url}/oauth2/tokenP"
    auth_body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret,
    }
    try:
        r_auth = requests.post(
            auth_url,
            json=auth_body,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        print(f"Auth Status: {r_auth.status_code}")
        if r_auth.status_code != 200:
            print(f"Auth failed: {r_auth.text[:200]}")
            return False

        token = r_auth.json().get("access_token")
        if not token:
            print("No token in response")
            return False
        print(f"Token obtained (length: {len(token)})")

        # 2. Try simple price inquiry (No account needed)
        price_url = f"{url}/uapi/overseas-price/v1/quotations/price"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": app_key,
            "appsecret": app_secret,
            "tr_id": "HHDFS00000300",
            "custtype": "P",
        }
        params = {"AUTH": "", "EXCD": "NAS", "SYMB": "AAPL"}
        r_price = requests.get(price_url, headers=headers, params=params, timeout=10)
        print(f"Price Inquiry Status: {r_price.status_code}")
        if r_price.status_code == 200:
            print(
                f"Price Result: {r_price.json().get('rt_cd')} - {r_price.json().get('msg1')}"
            )
        else:
            print(f"Price Inquiry Failed: {r_price.text[:200]}")

        # 3. Try Balance inquiry
        bal_url = f"{url}/uapi/domestic-stock/v1/trading/inquire-balance"
        tr_id = "VTTC8434R" if "vts" in url else "TTTC8434R"
        headers["tr_id"] = tr_id

        # Balance params (Domestic)
        bal_params = {
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
        # Adjust for Real if needed
        if "vts" not in url:
            bal_params = {
                "CANO": cano,
                "ACNT_PRDT_CD": prdt_cd,
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

        r_bal = requests.get(bal_url, headers=headers, params=bal_params, timeout=10)
        print(f"Balance Status: {r_bal.status_code}")
        if r_bal.status_code == 200:
            print(
                f"Balance Result: {r_bal.json().get('rt_cd')} - {r_bal.json().get('msg1')}"
            )
        else:
            print(f"Balance Inquiry Failed: {r_bal.text[:200]}")

    except Exception as e:
        print(f"Error testing {env_name}: {e}")

    return True


if __name__ == "__main__":
    key = "PSKqKzfa2FVwftOotO3WfI2CuBtQ3WtExJX1"
    sec = "gK57cT3fjgdzo7dvIyL6qsOffNTuAS/L6FrYdhzM6dz8glNSR5PdBg4uMeTjBcqN3lbDZamGWKobSsekw6P6i2CmQxm4NcPCCSbLqDZLYPvdd6ShlAuDYI4j8nt5g9XtmQLnQ3GJ6fUQpFcAGxGvMd1ThXPLDmTv/z5C4r7KziQ+Hawkjv8="
    cano = "50161248"

    urls = [
        ("MOCK (29443)", "https://openapivts.koreainvestment.com:29443"),
        ("REAL (9443)", "https://openapi.koreainvestment.com:9443"),
    ]

    for name, url in urls:
        test_combination(name, url, key, sec, cano)
