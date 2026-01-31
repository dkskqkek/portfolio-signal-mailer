# -*- coding: utf-8 -*-
import yaml
import os
import sys

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor


def test_tess_buy():
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)

    ticker = "095610"  # 테스
    print(f"🚀 [{ticker}] 테스 1주 시장가 매수 테스트 시도...")

    # In mock environment, side="BUY", ord_type="01" (Market)
    result = executor.create_order(ticker, side="BUY", qty=1, ord_type="01")

    print("\n--- KIS API 응답 결과 ---")
    import json

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("rt_cd") != "0":
        print(f"\n❌ 실패 사유: {result.get('msg1')}")
    else:
        print(f"\n✅ 성공: {result.get('msg1')}")


if __name__ == "__main__":
    test_tess_buy()
