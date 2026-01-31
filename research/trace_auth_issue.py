import requests
import json
import os


def trace_auth_issue():
    key = "PSKqKzfa2FVwftOotO3WfI2CuBtQ3WtExJX1"
    sec = "gK57cT3fjgdzo7dvIyL6qsOffNTuAS/L6FrYdhzM6dz8glNSR5PdBg4uMeTjBcqN3lbDZamGWKobSsekw6P6i2CmQxm4NcPCCSbLqDZLYPvdd6ShlAuDYI4j8nt5g9XtmQLnQ3GJ6fUQpFcAGxGvMd1ThXPLDmTv/z5C4r7KziQ+Hawkjv8="

    url_base = "https://openapivts.koreainvestment.com:29443"
    auth_url = f"{url_base}/oauth2/tokenP"

    # 1. Auth Call
    body = {"grant_type": "client_credentials", "appkey": key, "appsecret": sec}
    r_auth = requests.post(
        auth_url, json=body, headers={"Content-Type": "application/json"}
    )
    print(f"Auth Status: {r_auth.status_code}")
    auth_data = r_auth.json()
    token = auth_data.get("access_token")

    if not token:
        print("Auth Failed - No token.")
        return

    print(f"Token (First 20 chars): {token[:20]}...")

    # 2. Test simple GET with this token
    price_url = f"{url_base}/uapi/overseas-price/v1/quotations/price"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": key,
        "appsecret": sec,
        "tr_id": "HHDFS00000300",
    }
    params = {"AUTH": "", "EXCD": "NAS", "SYMB": "AAPL"}
    r_test = requests.get(price_url, headers=headers, params=params)
    print(f"Test Status: {r_test.status_code}")
    print(f"Test Response: {r_test.text[:200]}")


if __name__ == "__main__":
    trace_auth_issue()
