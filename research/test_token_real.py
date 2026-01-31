import requests
import json


def test_token_real(app_key, app_secret):
    url = "https://openapi.koreainvestment.com:9443/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret,
    }
    r = requests.post(url, json=body, headers={"Content-Type": "application/json"})
    return r.status_code, r.text


key = "PSKqKzfa2FVwftOotO3WfI2CuBtQ3WtExJX1"
sec = "gK57cT3fjgdzo7dvIyL6qsOffNTuAS/L6FrYdhzM6dz8glNSR5PdBg4uMeTjBcqN3lbDZamGWKobSsekw6P6i2CmQxm4NcPCCSbLqDZLYPvdd6ShlAuDYI4j8nt5g9XtmQLnQ3GJ6fUQpFcAGxGvMd1ThXPLDmTv/z5C4r7KziQ+Hawkjv8="

print("Testing Real API URL:")
st, txt = test_token_real(key, sec)
print(f"Status: {st}, Response: {txt}")
