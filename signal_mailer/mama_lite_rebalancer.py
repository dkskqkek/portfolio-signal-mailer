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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MAMA-Rebalancer")


def run_mama_rebalance():
    logger.info("Starting MAMA Lite Rebalancing Process...")

    # 1. Load Config
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)
    predictor = MAMAPredictor()

    # 2. Get Target Weights from MAMA Lite
    # Current MAMA Lite signal is calculated using daily data via yfinance
    logger.info("Predicting target weights via MAMA Lite...")
    target_weights = predictor.predict_portfolio()
    # Example: {'BIL': 0.5, 'TLT': 0.5}

    if not target_weights:
        logger.error("Failed to get target weights from MAMA Lite.")
        return

    logger.info(f"Target Weights: {target_weights}")

    # 3. Get Current Portfolio Status
    logger.info("Fetching current US holdings and cash...")
    current_holdings = executor.get_us_balance()  # List[Dict]
    # Current holdings output1 fields: 'ovrs_pdno' (ticker), 'ovrs_cblc_qty' (qty), etc.

    # KIS Mock US Balance summary often found in output2 or separate cash call
    # Let's use get_us_cash which uses inquire-psbl-order
    usd_cash = executor.get_us_cash()
    logger.info(f"Available USD Cash: ${usd_cash:,.2f}")

    # If USD cash is 0, check if we can use KRW (Integrated Funds)
    if usd_cash < 10:  # Threshold for "empty"
        logger.warning(
            "USD Cash is low. Checking KRW cash for Integrated Funds usage..."
        )
        krw_cash = executor.get_cash()
        logger.info(f"OrderExecutor returned KRW Cash: ₩{krw_cash}")
        if krw_cash > 100000:
            # Approx USD conversion for calculation (not official rate)
            usd_cash = krw_cash / 1400.0
            logger.info(f"Using estimated Integrated USD: ${usd_cash:,.2f}")
        else:
            logger.error("Insufficient funds (Both USD and KRW) in mock account.")
            return

    # 4. Calculate Orders
    # Filter out tickers already held (if any)
    # Using a set for comparison
    held_tickers_set = {
        h.get("ovrs_pdno")
        for h in current_holdings
        if h.get("ovrs_cblc_qty", "0") != "0"
    }
    logger.info(f"Current US Holdings: {held_tickers_set}")
    tickers_to_buy = {
        t: w for t, w in target_weights.items() if t not in held_tickers_set and w > 0
    }

    if not tickers_to_buy:
        logger.info("No new tickers to buy. Already in position or weights are 0.")
        return

    # Calculate total weight of tickers we are about to buy
    buying_weight_sum = sum(tickers_to_buy.values())

    # Calculate quantity for each ticker
    for ticker, weight in tickers_to_buy.items():
        time.sleep(1.0)  # Small delay to avoid hammering

        # Try to infer exchange
        exchange = "NAS"
        if ticker in ["BIL", "GLD", "SPY", "VTI", "COWZ", "BTAL", "PFIX"]:
            exchange = "AMS" if ticker in ["BIL", "GLD", "BTAL", "PFIX"] else "NYS"

        logger.info(f"Fetching price for {ticker} on {exchange}...")
        price = kis.get_us_current_price(ticker, exchange=exchange)
        if not price or price <= 0:
            logger.error(f"Could not fetch price for {ticker}. Skipping.")
            continue

        # Allocate from the CURRENTly available cash based on relative weight among remaining items
        # Use a slightly larger buffer (5%) for fees/slippage just in case
        portion = weight / buying_weight_sum
        allocation_usd = usd_cash * portion * 0.92
        qty = int(allocation_usd / price)

        if qty <= 0:
            logger.warning(
                f"Quantity for {ticker} is 0. (Alloc: ${allocation_usd:.2f} / Price: ${price:.2f})"
            )
            continue

        logger.info(
            f"Planned Order: BUY {ticker} | {qty} shares @ ~${price:.2f} (Total: ${price * qty:.2f})"
        )

        # 5. Execute Order
        # US Order requires limit price. We use current price + 0.1% for Buy orders for high fill probability.
        limit_price = round(price * 1.001, 2)

        res = executor.create_us_order(
            ticker=ticker,
            exchange=exchange,
            side="BUY",
            qty=qty,
            price=limit_price,
            ord_type="00",  # Limit
        )

        if res and res.get("rt_cd") == "0":
            logger.info(f"Successfully placed order for {ticker}.")
        else:
            logger.error(
                f"Failed to place order for {ticker}: {res.get('msg1') if res else 'Unknown error'}"
            )


if __name__ == "__main__":
    run_mama_rebalance()
