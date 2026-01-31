import yaml
import json
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)


def test_single_order():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)

    # Try buying 1 share of BIL
    ticker = "BIL"
    exchange = "AMS"
    price = kis.get_us_current_price(ticker, exchange=exchange)
    print(f"Current price for {ticker}: ${price}")

    if not price:
        print("Could not get price.")
        return

    limit_price = round(price * 1.01, 2)  # 1% higher for guaranteed fill

    print(f"Attempting to BUY 1 share of {ticker} @ ${limit_price}...")
    res = executor.create_us_order(
        ticker=ticker,
        exchange=exchange,
        side="BUY",
        qty=1,
        price=limit_price,
        ord_type="00",
    )

    print("\nFull Response:")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_single_order()
