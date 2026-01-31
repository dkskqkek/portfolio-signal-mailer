# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys
import json

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper

logging.basicConfig(level=logging.INFO)


def diagnose_scan():
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    kis = KISAPIWrapper(config["kis"])

    # 삼성전자 (005930) 테스트
    ticker = "005930"
    print(f"\n--- {ticker} (삼성전자) 데이터 진단 ---")
    res = kis.get_ohlcv_recent(ticker)

    if not res:
        print("❌ KIS API 응답이 없습니다.")
        return

    output = res.get("output")
    if not output:
        print(f"❌ output 데이터가 없습니다. 원본: {res}")
        return

    print(json.dumps(output[:2], indent=2, ensure_ascii=False))

    # 로직 재현
    curr_price = float(output[0]["stck_clpr"])
    prev_price = float(output[1]["stck_clpr"])
    last_5 = [float(x["stck_clpr"]) for x in output[:5]]
    sma_5 = sum(last_5) / 5

    print(f"\n현재가: {curr_price}")
    print(f"전일가: {prev_price}")
    print(f"5일 평균: {sma_5}")
    print(
        f"결과: Close > SMA_5 ({curr_price > sma_5}), ROC_1 > 0 ({curr_price > prev_price})"
    )


if __name__ == "__main__":
    diagnose_scan()
