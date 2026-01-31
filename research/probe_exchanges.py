import yaml
import json
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper


def probe_exchanges(ticker):
    # Load Config
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])

    exchanges = ["NAS", "NYS", "AMS"]
    for ex in exchanges:
        price = kis.get_us_current_price(ticker, exchange=ex)
        if price:
            print(f"Ticker {ticker} found on {ex} at ${price}")
            return ex
        else:
            print(f"Ticker {ticker} NOT found on {ex}")
    return None


if __name__ == "__main__":
    probe_exchanges("BIL")
    probe_exchanges("TLT")
