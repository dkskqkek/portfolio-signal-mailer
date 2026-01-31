import yaml
import sys
import os
import time

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor
from signal_mailer.mama_lite_predictor import MAMAPredictor


def full_rebalance_test():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)
    predictor = MAMAPredictor()

    print("1. Predicting...")
    target_weights = predictor.predict_portfolio()
    print(f"   Target: {target_weights}")

    print("2. Checking Cash...")
    usd_cash = executor.get_us_cash()
    print(f"   USD Cash: {usd_cash}")

    if usd_cash < 10:
        print("   Checking KRW Integrated Cash...")
        krw_cash = executor.get_cash()
        print(f"   KRW Cash: {krw_cash}")
        usd_cash = krw_cash / 1410.0  # Use slightly higher rate for safety
        print(f"   Virtual USD: {usd_cash}")

    print("3. Executing Order for BIL...")
    ticker = "BIL"
    exchange = "AMS"
    price = kis.get_us_current_price(ticker, exchange=exchange)
    print(f"   Current Price of {ticker}: {price}")

    weight = target_weights.get(ticker, 0.5)
    allocation = usd_cash * weight * 0.90  # Use 90% for safety
    qty = int(allocation / price)
    print(f"   Planned: {qty} shares @ {price * 1.001}")

    res = executor.create_us_order(
        ticker=ticker,
        exchange=exchange,
        side="BUY",
        qty=qty,
        price=round(price * 1.001, 2),
        ord_type="00",
    )
    print(f"   Order Result: {json.dumps(res, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    import json

    full_rebalance_test()
