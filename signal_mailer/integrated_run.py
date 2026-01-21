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

def get_advanced_signal():
    """고급 레짐 감지 시스템 실행 및 최신 결과 추출"""
    print("\n[고급 시그널 엔진 가동 중...]")
    try:
        pipeline = CrashDetectionPipeline(
            ticker='SPY',
            start_date=(datetime.datetime.now() - datetime.timedelta(days=365*5)).strftime('%Y-%m-%d'),
            cache_dir=str(BASE_DIR / 'crash_detection_system' / 'data')
        )
        results = pipeline.run_full_pipeline()
        
        if results['status'] == 'SUCCESS':
            signal_val = pipeline.signals['signal'].iloc[-1]
            reason = pipeline.signals['signal_reason'].iloc[-1]
            regime = pipeline.indicators['HMM_Regime'].iloc[-1]
            
            regime_map = {0: 'Bull (상승)', 1: 'Correction (조정)', 2: 'Crisis (위기)'}
            regime_name = regime_map.get(int(regime), "Unknown")
            
            signal_map = {2: 'STRONG BUY', 1: 'BUY', 0: 'NEUTRAL (중립)', -1: 'SELL (매도)', -2: 'STRONG SELL (강력 매도)'}
            signal_name = signal_map.get(int(signal_val), "Unknown")
            
            return {
                'success': True,
                'signal': signal_name,
                'regime': regime_name,
                'reason': reason,
                'indicators': {
                    'RSI': pipeline.indicators['RSI'].iloc[-1],
                    'ADX': pipeline.indicators['ADX'].iloc[-1],
                    'VIX': pipeline.indicators['VIX'].iloc[-1]
                }
            }
    except Exception as e:
        print(f"고급 시그널 실행 중 오류 발생: {e}")
    return {'success': False, 'error': "고급 엔진 실행 실패"}

def generate_reports(today_str, status_title, is_overall_danger, is_simple_danger, is_adv_sell, simple_info, adv_info):
    """순수 텍스트 리포트 생성 (Email & Markdown 공용)"""
    line = "=" * 60
    
    report = f"""{line}
📅 {today_str} DAILY MARKET INTELLIGENCE
{line}

[종합 시장 신호] : {status_title}
[권장 스탠스]     : {'방어적 리밸런싱 (JEPI 전환)' if is_overall_danger else '공격적 자산 운용 (QQQ 유지)'}

{line}
1. 멀티-팩터 엔진 분석 결과
{line}

(1) 시클리컬 엔진 (MA/Vol)
    - 판정: {'[🚨 DANGER]' if is_simple_danger else '[✅ NORMAL]'}
    - 근거: {simple_info.get('reason', '지표 정상')}

"""

    if adv_info['success']:
        report += f"""(2) AI 인텔리전스 (HMM)
    - 판정: {'[🚨 ' + adv_info['signal'] + ']' if is_adv_sell else '[💎 ' + adv_info['signal'] + ']'}
    - 레짐: {adv_info['regime']}
    - 근거: {adv_info['reason'].strip() if adv_info['reason'] else '정상'}
    - 지표: RSI({adv_info['indicators']['RSI']:.1f}) | ADX({adv_info['indicators']['ADX']:.1f}) | VIX({adv_info['indicators']['VIX']:.1f})
"""
    else:
        report += "(2) AI 인텔리전스 (HMM)\n    - 판정: [❌ ENGINE ERROR]\n"

    growth_weight = " 0%" if is_overall_danger else "38%"
    defense_weight = "38%" if is_overall_danger else " 0%"
    
    report += f"""
{line}
2. 전략적 자산 배분 제안
{line}

(Ticker) | (기본 비중) | (권장 비중) | (Action)
------------------------------------------------------------
 SCHD    |    38%     |    38%     |   HOLD
 QQQ     |    38%     |   {growth_weight}     |   {'SELL' if is_overall_danger else 'HOLD'}
 JEPI    |     0%     |   {defense_weight}     |   {'BUY ' if is_overall_danger else ' -  '}
 KS200   |    19%     |    19%     |   HOLD
 GOLD    |     5%     |     5%     |   HOLD
------------------------------------------------------------

{line}
3. 투자 핵심 가이드
{line}
"""

    if is_simple_danger and is_adv_sell:
        report += "!!! [🚨] 강력 경고: 이중 매도 신호 발생 !!!\n"
        report += "- 모든 엔진에서 위기 신호가 포착되었습니다.\n"
        report += "- QQQ 비중을 전량(38%) 매도하고 JEPI(38%)로 교체하세요.\n"
    elif is_overall_danger:
        report += "!!! [⚠️] 주의: 부분적 위험 신호 감지 !!!\n"
        report += f"- {'일반' if is_simple_danger else '고급'} 엔진 경합: QQQ -> JEPI 교체 준비\n"
        report += "- 안정성을 위해 성장주 비중을 축소하고 배당 방어주로 전환을 권장합니다.\n"
    else:
        report += "!!! [✅] 상태 정상: 포지션 유지 !!!\n"
        report += "- 모든 지표가 우상향을 지지합니다. 성장주(QQQ) 비중을 유지하세요.\n"

    report += f"""
{line}
본 리포트는 ANTIGRAVITY INTELLIGENCE에 의해 자동 생성되었습니다.
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
    
    print("\n[엔진 가동 중...]")
    detector = SignalDetector()
    simple_info = detector.detect()
    adv_info = get_advanced_signal()
    
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    is_simple_danger = simple_info.get('is_danger', False)
    is_adv_sell = adv_info['success'] and "SELL" in adv_info['signal']
    is_overall_danger = is_simple_danger or is_adv_sell
    
    status_title = "정상"
    if is_simple_danger and is_adv_sell: status_title = "🚨 위험 (매도)"
    elif is_overall_danger: status_title = "⚠️ 주의 (조정)"
    
    text_report = generate_reports(today_str, status_title, is_overall_danger, is_simple_danger, is_adv_sell, simple_info, adv_info)
    
    subject = f"[신호 통합 리포트] {today_str} : {status_title}"
    
    print("\n[이메일 발송 중...]")
    mailer = MailerService(config)
    result = mailer.send_email(subject, text_report)
    
    if result['success']:
        print(f"✓ {result['message']}")
        # 로컬 환경에서만 히스토리 저장 (GitHub Actions은 휘발성)
        if not os.environ.get('GITHUB_ACTIONS'):
            mailer.save_history('DANGER' if is_simple_danger else 'NORMAL', simple_info)
    else:
        # GitHub Actions 로그에서 실패를 명확히 알리기 위해 에러 출력
        print(f"이메일 발송 실패: {result.get('message')}")
        sys.exit(1)

if __name__ == '__main__':
    main()
