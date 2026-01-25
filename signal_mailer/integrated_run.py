# -*- coding: utf-8 -*-
"""
신규 통합 신호 발송 스크립트 (integrated_run.py)
1. 기존 단순 시그널 (MA/Volatility)
2. 고급 레짐 감지 시그널 (Kalman + HMM)
두 결과를 하나로 합쳐 데일리 리포트(Email + Markdown)를 발송 및 생성합니다.
* 디자인: 순수 텍스트와 기호만 사용
* 전략배분: QLD(45%), KOSPI(20%), SPY(20%), GOLD(15%)
* 방어모드: Top-3 Defensive Ensemble (23종 순수 1배물)
* 서버리스: GitHub Actions 환경 변수 지원 추가
"""

import sys
import os
import datetime
import yaml
import pandas as pd
from pathlib import Path

# 경로 설정 (GitHub Actions 환경 대응)
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "signal_mailer"))

from signal_detector import SignalDetector
from mailer_service import MailerService


def load_config():
    """설정 로드 (환경 변수 우선, 없으면 config.yaml)"""
    config_path = BASE_DIR / "signal_mailer" / "config.yaml"
    config = {}

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    # 환경 변수 덮어쓰기 (GitHub Actions용)
    if os.environ.get("SENDER_EMAIL"):
        if "email" not in config:
            config["email"] = {}
        config["email"]["sender_email"] = os.environ.get("SENDER_EMAIL")
        config["email"]["sender_password"] = os.environ.get("SENDER_PASSWORD")
        config["email"]["recipient_email"] = os.environ.get("RECIPIENT_EMAIL")

    return config


def main():
    config = load_config()

    print("\n[최적화 하이브리드 엔진 가동 중]")
    print("  - 엔진 로직: Dual SMA (110/250) + Top-3 Defensive Ensemble")

    # 신호 탐지 실행
    start_time = datetime.datetime.now()
    detector = SignalDetector()
    signal_info = detector.detect()
    signal_info["execution_time"] = start_time

    # 리포트 생성 (통일된 포맷 사용)
    report = SignalDetector.format_signal_report(signal_info)

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    subject = f"[시장 신호 리포트] {today_str} : {report['title']}"

    print("\n[이메일 발송 중...]")
    mailer = MailerService(config)
    # 리포트 딕셔너리에서 생성된 HTML 본문을 추출하여 전송
    result = mailer.send_email(
        subject, report["body"], html_body=report.get("html_body")
    )

    if result["success"]:
        print(f"✓ {result['message']}")
        # 로컬 환경에서만 히스토리 저장
        if not os.environ.get("GITHUB_ACTIONS"):
            mailer.save_history(report["status"], signal_info)

        # GitHub Actions 환경에서 리포트 파일 저장 (워크플로우에서 커밋용)
        report_path = BASE_DIR / "latest_report.md"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report["body"])
            print(f"✓ 리포트 저장 완료: {report_path}")
        except Exception as e:
            print(f"리포트 저장 실패: {e}")
    else:
        print(f"이메일 발송 실패: {result.get('message')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
