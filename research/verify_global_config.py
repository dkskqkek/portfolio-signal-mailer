import yaml
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from signal_mailer.kis_api_wrapper import KISAPIWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Global-Config-Check")


def verify_global_config():
    # 1. Load config
    with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    kis_config = config["kis"]
    print(f"Checking keys in config.yaml...")
    print(f"AppKey (first 5): {kis_config['app_key'][:5]}...")
    print(f"Account: {kis_config['account_no']}")

    # 2. Test Authentication via KISAPIWrapper
    print("\nAttempting authentication via Wrapper...")
    try:
        # Clear cache to force real auth check
        for suffix in ["mock", "real"]:
            cache_file = f".kis_token_cache_{suffix}.json"
            if os.path.exists(cache_file):
                os.remove(cache_file)

        kis = KISAPIWrapper(kis_config)
        if kis.access_token:
            print("✅ Authentication SUCCESSFUL via global wrapper.")
            print(f"Access Token length: {len(kis.access_token)}")
        else:
            print("❌ Authentication FAILED via global wrapper.")
    except Exception as e:
        print(f"❌ Error during auth test: {e}")


if __name__ == "__main__":
    verify_global_config()
