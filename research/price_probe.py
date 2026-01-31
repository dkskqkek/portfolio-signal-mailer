import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def price_probe():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    # TR for US Stock Price (Real/Mock often same)
    # Using kis.get_us_current_price logic
    ticker = "AAPL"
    exchange = "NAS"

    url = f"{kis.base_url}/uapi/overseas-price/v1/quotations/price"
    params = {
        "AUTH": "",
        "EXCD": exchange,
        "SYMB": ticker,
    }
    headers = {**kis.headers, "tr_id": "HHDFS00000300"}

    r = kis.call_get(url, headers=headers, params=params)
    print(f"--- Price Probe for {ticker} ---")
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    price_probe()
