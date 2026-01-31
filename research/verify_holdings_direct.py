import yaml
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)


def verify_holdings_direct():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)

    print(f"Verifying Holdings for Account: {executor.cano}")

    # Check US Holdings
    holdings = executor.get_us_balance()
    print(f"\nFinal US Holdings Count: {len(holdings)}")
    for h in holdings:
        print(f"  - {h.get('ovrs_pdno')}: {h.get('ovrs_cblc_qty')} shares")


if __name__ == "__main__":
    verify_holdings_direct()
