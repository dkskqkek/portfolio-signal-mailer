import requests
import json


def test_real_auth():
    key = "PSKqKzfa2FVwftOotO3WfI2CuBtQ3WtExJX1"
    sec = "gK57cT3fjgdzo7dvIyL6qsOffNTuAS/L6FrYdhzM6dz8glNSR5PdBg4uMeTjBcqN3lbDZamGWKobSsekw6P6i2CmQxm4NcPCCSbLqDZLYPvdd6ShlAuDYI4j8nt5g9XtmQLnQ3GJ6fUQpFcAGxGvMd1ThXPLDmTv/z5C4r7KziQ+Hawkjv8="

    url_real = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"
    body = {"grant_type": "client_credentials", "appkey": key, "appsecret": sec}
    r_auth = requests.post(
        url_real, json=body, headers={"Content-Type": "application/json"}
    )
    print(f"Real Auth Status: {r_auth.status_code}")
    print(f"Real Auth Response: {r_auth.text[:200]}")


if __name__ == "__main__":
    test_real_auth()
