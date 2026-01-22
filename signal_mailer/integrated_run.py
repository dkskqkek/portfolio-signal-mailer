# -*- coding: utf-8 -*-
"""
신규 통합 신호 발송 스크립트 (integrated_run.py)
1. 기존 단순 시그널 (MA/Volatility)
2. 고급 레짐 감지 시그널 (Kalman + HMM)
두 결과를 하나로 합쳐 데일리 리포트(Email + Markdown)를 발송 및 생성합니다.
* 디자인: 순수 텍스트와 기호만 사용
* 전략배분: SCHD(38%), QQQ or JEPI(38%), KS200(19%), GOLD(5%)
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
sys.path.insert(0, str(BASE_DIR / 'signal_mailer'))
sys.path.insert(0, str(BASE_DIR / 'crash_detection_system' / 'src'))

from signal_detector import SignalDetector
from mailer import MailerService
from main import CrashDetectionPipeline

def load_config():
    """설정 로드 (환경 변수 우선, 없으면 config.yaml)"""
    config_path = BASE_DIR / 'signal_mailer' / 'config.yaml'
    config = {}
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 환경 변수 덮어쓰기 (GitHub Actions용)
    if os.environ.get('SENDER_EMAIL'):
        if 'email' not in config: config['email'] = {}
        config['email']['sender_email'] = os.environ.get('SENDER_EMAIL')
        config['email']['sender_password'] = os.environ.get('SENDER_PASSWORD')
        config['email']['recipient_email'] = os.environ.get('RECIPIENT_EMAIL')
    
    if os.environ.get('GEMINI_API_KEY'):
        config['gemini_api_key'] = os.environ.get('GEMINI_API_KEY')
        
    return config

def get_hmm_signal_disabled():
    """HMM 엔진 비활성화 (무결성 검증 결과 기본 시그널이 우세함)"""
    return {'success': False, 'is_danger': False, 'error': "HMM 엔진 비활성화됨"}

def generate_reports(today_str, status_title, is_danger, signal_info):
    """최적화된 기본 시그널 전용 리포트 생성 (Email & Markdown 공용)"""
    line = "=" * 60
    
    report = f"""{line}
📅 {today_str} DAILY MARKET INTELLIGENCE (Optimized Basic)
{line}

[종합 시장 신호] : {status_title}
[권장 스탠스]     : {'방어적 리밸런싱 (JEPI 전환)' if is_danger else '공격적 자산 운용 (QQQ 유지)'}

{line}
1. 최적화된 시클리컬 엔진 분석 결과 (15d MA / 30d Vol)
{line}

(1) 시그널 판정
    - 상태: {'[🚨 DANGER (위험)]' if is_danger else '[✅ NORMAL (정상)]'}
    - 근거: {signal_info.get('reason', '정상 범위 내 동작 중')}

(2) 엔진 최적 파라미터 (무결성 검증 완료)
    - MA Window: 15일 (추세 감지 강화)
    - Vol Window: 30일 (변동성 측정 안정화)
    - Threshold: MA 25th / Vol 65th Percentile
    - 기대 성과: CAGR 13.1% | Sharpe 1.04 | MDD -20.2% (SCHD 기반)
"""

    growth_weight = " 0%" if is_danger else "38%"
    defense_weight = "38%" if is_danger else " 0%"
    
    report += f"""
{line}
2. 전략적 자산 배분 제안 (SCHD Core Portfolio)
{line}

(Ticker) | (기본 비중) | (권장 비중) | (Action)
------------------------------------------------------------
 SCHD    |    38%     |    38%     |   HOLD
 QQQ     |    38%     |   {growth_weight}     |   {'SELL' if is_danger else 'HOLD'}
 JEPI    |     0%     |   {defense_weight}     |   {'BUY ' if is_danger else ' -  '}
 KS200   |    19%     |    19%     |   HOLD
 GOLD    |     5%     |     5%     |   HOLD
------------------------------------------------------------

{line}
3. 투자 핵심 가이드
{line}
"""

    if is_danger:
        report += "!!! [🚨] 위험 신호 감지: 방어 자산 전환 !!!\n"
        report += "- 15일 이동평균 또는 30일 변동성이 임계치를 벗어났습니다.\n"
        report += "- QQQ 비중을 전량(38%) 매도하고 JEPI(38%)로 교체하세요.\n"
        report += "- 시장 변동성이 안정될 때까지 배당 수익 위주로 대응합니다.\n"
    else:
        report += "!!! [✅] 상태 정상: 공격적 포지션 유지 !!!\n"
        report += "- 추세와 변동성이 우호적인 영역에 머물러 있습니다.\n"
        report += "- 성장주(QQQ) 비중을 유지하며 복리 수익을 극대화하세요.\n"

    report += f"""
{line}
본 리포트는 ANTIGRAVITY INTELLIGENCE (Optimized Basic)에 의해 자동 생성되었습니다.
전략: 15d MA / 30d Vol Percentile (CAGR 13.1%, MDD -20.2%)
작성일: {today_str}
{line}
"""
    
    report_path = BASE_DIR / "latest_report.md"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✓ Markdown 리포트 생성 완료: {report_path}")
    except Exception as e:
        print(f"Markdown 저장 실패 (권한 등): {e}")
    
    return report

def main():
    config = load_config()
    
    print("\n[최적화된 기본 시그널 엔진 가동 중]")
    print("  - 엔진 로직: 15d MA / 30d Vol Percentile")
    print("  - 코어 자산: SCHD (38%)")
    
    # 기본 시그널 탐지 실행
    detector = SignalDetector()
    signal_info = detector.detect()
    is_danger = signal_info.get('is_danger', False)
    
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    status_title = "🚨 위험 (방어 전환)" if is_danger else "✅ 정상 (QQQ 유지)"
    
    text_report = generate_reports(today_str, status_title, is_danger, signal_info)
    
    subject = f"[시장 신호 리포트] {today_str} : {status_title}"
    
    print("\n[이메일 발송 중...]")
    mailer = MailerService(config)
    result = mailer.send_email(subject, text_report)
    
    if result['success']:
        print(f"✓ {result['message']}")
        # 로컬 환경에서만 히스토리 저장
        if not os.environ.get('GITHUB_ACTIONS'):
            mailer.save_history('DANGER' if is_danger else 'NORMAL', signal_info)
    else:
        print(f"이메일 발송 실패: {result.get('message')}")
        sys.exit(1)

if __name__ == '__main__':
    main()
