import logging
import yaml
import sys
import os
import time

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor
from signal_mailer.mama_lite_predictor import MAMAPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MAMA-Debug")


def debug_rebalance():
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Using Account: {config['kis']['account_no']}")
    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)

    logger.info("Checking US Cash...")
    usd_cash = executor.get_us_cash()
    logger.info(f"USD Cash: {usd_cash}")

    if usd_cash < 10:
        logger.warning("USD Cash is low. Checking KRW cash...")
        krw_cash = executor.get_cash()
        logger.info(f"KRW Cash: {krw_cash}")


if __name__ == "__main__":
    debug_rebalance()
