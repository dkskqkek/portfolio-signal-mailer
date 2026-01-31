import requests
import json


def test_token(app_key, app_secret):
    url = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret,
    }
    r = requests.post(url, json=body, headers={"Content-Type": "application/json"})
    return r.status_code, r.text


# Variant 1: No dot
key1 = "PSKqKzfa2FVwftOotO3WfI2CuBtQ3WtExJX1"
sec1 = "gK57cT3fjgdzo7dvIyL6qsOffNTuAS/L6FrYdhzM6dz8glNSR5PdBg4uMeTjBcqN3lbDZamGWKobSsekw6P6i2CmQxm4NcPCCSbLqDZLYPvdd6ShlAuDYI4j8nt5g9XtmQLnQ3GJ6fUQpFcAGxGvMd1ThXPLDmTv/z5C4r7KziQ+Hawkjv8="

# Variant 2: With dot
key2 = "PSKqKzfa2FVwftOotO3WfI2CuBtQ3WtExJX1."

print(f"Testing Variant 1 (No dot):")
st1, txt1 = test_token(key1, sec1)
print(f"Status: {st1}, Response: {txt1}")

print(f"\nTesting Variant 2 (With dot):")
st2, txt2 = test_token(key2, sec1)
print(f"Status: {st2}, Response: {txt2}")
