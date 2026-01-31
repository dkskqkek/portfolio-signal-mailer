import logging
import yaml
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor
from signal_mailer.mama_lite_predictor import MAMAPredictor
from signal_mailer.mama_lite_rebalancer import run_mama_rebalance

# Force INFO logging for all
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("--- STARTING INSTRUMENTED REBALANCE DEBUG ---")
    try:
        run_mama_rebalance()
    except Exception as e:
        print(f"CRITICAL ERROR in rebalancer: {e}")
    print("--- DEBUG COMPLETE ---")
