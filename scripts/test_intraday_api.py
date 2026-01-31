"""
간단 테스트: KIS API 분봉 조회 기능 검증
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from signal_mailer.kis_api_wrapper import KISAPIWrapper
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config
with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

kis = KISAPIWrapper(config["kis"])

# Test 1: 한국 주식 분봉 (삼성전자)
print("\n=== Test 1: 한국 주식 5분봉 (삼성전자 005930) ===")
try:
    kr_bars = kis.get_intraday_bars("005930", period="5")
    if kr_bars and len(kr_bars) > 0:
        print(f"✅ 수집 성공: {len(kr_bars)} bars")
        print(f"첫 번째 bar 샘플: {kr_bars[0]}")
        print(f"마지막 bar 샘플: {kr_bars[-1]}")
    else:
        print(f"⚠️  빈 응답: {kr_bars}")
        print("※ 주의: 장시간 외에는 분봉 데이터가 제공되지 않을 수 있습니다")
except Exception as e:
    print(f"❌ 에러: {e}")
    import traceback

    traceback.print_exc()

# Test 2: 미국 주식 분봉 (AAPL)
print("\n=== Test 2: 미국 주식 1분봉 (AAPL) ===")
try:
    us_bars = kis.get_us_intraday_bars("AAPL", exchange="NAS", period="1")
    if us_bars and len(us_bars) > 0:
        print(f"✅ 수집 성공: {len(us_bars)} bars")
        print(f"첫 번째 bar 샘플: {us_bars[0]}")
        print(f"마지막 bar 샘플: {us_bars[-1]}")
    else:
        print(f"⚠️  빈 응답: {us_bars}")
        print("※ 주의: 미국 시장 개장 시간 외에는 데이터가 제공되지 않을 수 있습니다")
except Exception as e:
    print(f"❌ 에러: {e}")
    import traceback

    traceback.print_exc()

print("\n=== 테스트 완료 ===")
print("\n※ 참고: 실제 데이터 수집은 장마감 후에 실행해야 합니다")
print("  - 한국: 매일 15:40 (KST)")
print("  - 미국: 매일 06:30 (KST)")
