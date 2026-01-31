import yaml
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor

# Setup logging to see executor logs
logging.basicConfig(level=logging.INFO)


def test_executor_cash():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)

    print(f"Testing OrderExecutor for Account: {executor.cano}")
    cash = executor.get_cash()
    print(f"\nFinal Cash Result: ₩{cash}")


if __name__ == "__main__":
    test_executor_cash()
