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
    """
    최적화된 HMM 전략 실행
    
    최적 파라미터 (백테스트 결과):
    - Regime Threshold: 1.0
    - RSI Crisis: 45
    - RSI Normal: 40
    - ADX Min: 15
    - VIX High: 25
    
    성과 (SPY 기준):
    - CAGR: 21.83%
    - Sharpe: 1.43
    - MDD: -20.00%
    - Danger 비율: 16.4%
    """
    print("\n[최적화된 HMM 전략 엔진 가동 중...]")
    try:
        pipeline = CrashDetectionPipeline(
            ticker='SPY',
            start_date=(datetime.datetime.now() - datetime.timedelta(days=365*5)).strftime('%Y-%m-%d'),
            cache_dir=str(BASE_DIR / 'crash_detection_system' / 'data')
        )
        results = pipeline.run_full_pipeline()
        
        if results['status'] == 'SUCCESS':
            # HMM 레짐 및 지표 추출
            regime = pipeline.indicators['HMM_Regime'].iloc[-1]
            rsi = pipeline.indicators['RSI'].iloc[-1]
            adx = pipeline.indicators['ADX'].iloc[-1]
            vix = pipeline.indicators['VIX'].iloc[-1]
            
            # 최적화된 파라미터로 시그널 판정
            regime_threshold = 1.0
            rsi_crisis = 45
            rsi_normal = 40
            adx_min = 15
            vix_high = 25
            
            is_danger = False
            reason = ""
            
            # ADX 필터
            if adx < adx_min:
                is_danger = False
                reason = f"추세 약함 (ADX={adx:.1f} < {adx_min})"
            # Crisis 레짐
            elif regime >= 2:
                if rsi < rsi_crisis:
                    is_danger = True
                    reason = f"Crisis 레짐 + RSI 과매도 (RSI={rsi:.1f} < {rsi_crisis})"
                else:
                    is_danger = True
                    reason = f"Crisis 레짐 감지 (RSI={rsi:.1f})"
            # Correction 레짐
            elif regime >= regime_threshold:
                if rsi < rsi_normal or vix > vix_high:
                    is_danger = True
                    reason = f"Correction 레짐 + 위험 지표 (RSI={rsi:.1f}, VIX={vix:.1f})"
                else:
                    reason = f"Correction 레짐이나 지표 정상 (RSI={rsi:.1f}, VIX={vix:.1f})"
            else:
                reason = f"Bull 레짐 - 정상 (RSI={rsi:.1f}, VIX={vix:.1f})"
            
            regime_map = {0: 'Bull (상승)', 1: 'Correction (조정)', 2: 'Crisis (위기)'}
            regime_name = regime_map.get(int(regime), "Unknown")
            
            signal_name = 'DANGER (위험)' if is_danger else 'NORMAL (정상)'
            
            return {
                'success': True,
                'is_danger': is_danger,
                'signal': signal_name,
                'regime': regime_name,
                'reason': reason,
                'indicators': {
                    'RSI': rsi,
                    'ADX': adx,
                    'VIX': vix
                }
            }
    except Exception as e:
        print(f"HMM 전략 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    return {'success': False, 'is_danger': False, 'error': "HMM 엔진 실행 실패"}

def generate_reports(today_str, status_title, is_overall_danger, is_simple_danger, is_adv_danger, simple_info, adv_info):
    """HMM 전략 전용 리포트 생성 (Email & Markdown 공용)"""
    line = "=" * 60
    
    report = f"""{line}
📅 {today_str} DAILY MARKET INTELLIGENCE (HMM 전략)
{line}

[종합 시장 신호] : {status_title}
[권장 스탠스]     : {'방어적 리밸런싱 (JEPI 전환)' if is_overall_danger else '공격적 자산 운용 (QQQ 유지)'}

{line}
1. 최적화된 HMM 전략 분석 결과
{line}

"""

    if adv_info['success']:
        report += f"""(1) HMM 전략 엔진 (최적 파라미터)
    - 판정: {'[🚨 ' + adv_info['signal'] + ']' if is_adv_danger else '[✅ ' + adv_info['signal'] + ']'}
    - 레짐: {adv_info['regime']}
    - 근거: {adv_info['reason'].strip() if adv_info['reason'] else '정상'}
    - 지표: RSI({adv_info['indicators']['RSI']:.1f}) | ADX({adv_info['indicators']['ADX']:.1f}) | VIX({adv_info['indicators']['VIX']:.1f})

(2) 최적 파라미터 (백테스트 검증)
    - Regime Threshold: 1.0 (Correction부터 위험 인식)
    - RSI Crisis: 45 / RSI Normal: 40
    - ADX Min: 15 / VIX High: 25
    - 성과: CAGR 21.83% | Sharpe 1.43 | MDD -20.00%
"""
    else:
        report += "(1) HMM 전략 엔진\n    - 판정: [❌ ENGINE ERROR]\n"

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

    if is_adv_danger:
        report += "!!! [🚨] HMM 전략 위험 신호 발생 !!!\n"
        report += "- HMM 레짐 분석 결과 위험 구간 진입\n"
        report += "- QQQ 비중을 전량(38%) 매도하고 JEPI(38%)로 교체하세요.\n"
        report += "- 안정적인 배당 수익으로 하락장을 방어하세요.\n"
    else:
        report += "!!! [✅] HMM 전략 정상 신호 !!!\n"
        report += "- 시장 레짐이 안정적입니다. 성장주(QQQ) 비중을 유지하세요.\n"
        report += "- 최적화된 HMM 전략이 상승장 지속을 지지합니다.\n"

    report += f"""
{line}
본 리포트는 ANTIGRAVITY INTELLIGENCE (HMM 전략)에 의해 자동 생성되었습니다.
전략: 최적화된 HMM 레짐 감지 (CAGR 21.83%, Sharpe 1.43)
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
    
    print("\n[최적화된 HMM 전략 엔진만 사용]")
    print("  - 기본 시그널: 비활성화")
    print("  - HMM 전략: 활성화 (최적 파라미터)")
    
    # 기본 시그널 비활성화 - HMM만 사용
    adv_info = get_advanced_signal()
    
    # 더미 simple_info (사용하지 않음)
    simple_info = {
        'is_danger': False,
        'reason': '기본 시그널 비활성화 (HMM 전략만 사용)',
        'error': False
    }
    
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # HMM 시그널만 사용
    is_simple_danger = False  # 기본 시그널 비활성화
    is_adv_danger = adv_info.get('is_danger', False) if adv_info['success'] else False
    is_overall_danger = is_adv_danger  # HMM 시그널만 사용
    
    status_title = "정상"
    if is_adv_danger:
        status_title = "🚨 위험 (HMM 전략)"
    
    text_report = generate_reports(today_str, status_title, is_overall_danger, is_simple_danger, is_adv_danger, simple_info, adv_info)
    
    subject = f"[HMM 전략 리포트] {today_str} : {status_title}"
    
    print("\n[이메일 발송 중...]")
    mailer = MailerService(config)
    result = mailer.send_email(subject, text_report)
    
    if result['success']:
        print(f"✓ {result['message']}")
        # 로컬 환경에서만 히스토리 저장 (GitHub Actions은 휘발성)
        if not os.environ.get('GITHUB_ACTIONS'):
            mailer.save_history('DANGER' if is_adv_danger else 'NORMAL', adv_info if adv_info['success'] else simple_info)
    else:
        # GitHub Actions 로그에서 실패를 명확히 알리기 위해 에러 출력
        print(f"이메일 발송 실패: {result.get('message')}")
        sys.exit(1)

if __name__ == '__main__':
    main()
