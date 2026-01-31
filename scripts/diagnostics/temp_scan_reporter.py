# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.kr_stock_scanner import KRStockScanner

logging.basicConfig(level=logging.INFO)


def run_scan():
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    kis = KISAPIWrapper(config["kis"])
    scanner = KRStockScanner(kis)

    print("\n🔍 현 시각 KR 마켓 하이브리드 알파 스캔 시작 (Top 100)...")
    candidates = scanner.scan_full_market(limit=100)

    if not candidates:
        print("🔍 조건에 부합하는 종목이 없습니다.")
        return

    print(f"✅ 총 {len(candidates)}개의 종목이 탐지되었습니다.\n")
    print("| 순위 | 종목명 (코드) | 현재가 | 1일 수익률 | 5일선 이격 |")
    print("| :--- | :--- | :--- | :--- | :--- |")

    for i, c in enumerate(candidates, 1):
        print(
            f"| {i:2d} | {c['name']} ({c['ticker']}) | {c['price']:,}원 | {c['roc_1'] * 100:+.2f}% | {c['dist_sma'] * 100:+.2f}% |"
        )


if __name__ == "__main__":
    run_scan()
