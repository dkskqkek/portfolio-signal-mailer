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

from signal_detector import SignalDetector
from mailer import MailerService

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
    
    return config

def generate_reports(today_str, status_title, is_danger, signal_info):
    """최적화 하이브리드(Fusion) 모델 리포트 생성 (메일용/로컬용 분리)"""
    mf_score = signal_info.get('mf_score', 50.0)
    m1_danger = signal_info.get('m1_danger', False)
    
    # 1. 메일용 텍스트 포맷 (기존의 텍스트+기호 방식)
    line = "=" * 60
    bar_len = 20
    filled = int(mf_score / 100 * bar_len)
    bar = "■" * filled + "□" * (bar_len - filled)
    
    mail_content = f"""{line}
📅 {today_str} DAILY MARKET INTELLIGENCE (Optimized Fusion)
{line}

[종합 시장 신호] : {status_title}
[권장 스탠스]     : {'방어적 리밸런싱 (JEPI 전환)' if is_danger else '공격적 자산 운용 (QQQ 유지)'}

{line}
1. 최적화 하이브리드 엔진 분석 (Sentinel + Validator)
{line}

(1) 시그널 판정
    - 최종 상태: {'[🚨 DANGER (위험)]' if is_danger else '[✅ NORMAL (정상)]'}
    - 판정 근거: {signal_info.get('reason', '정상 범위 내 동작 중')}

(2) 세부 데이터 분석
    - 기술적 위기 감지 (Sentinel): {'[ON]' if m1_danger else '[OFF]'}
    - 멀티팩터 심리 점수 (Validator): {mf_score:.1f}점
      [Fear 0 {bar} 100 Greed]

(3) 엔진 스펙 (Optimized Hybrid)
    - 로직: 기술지표(15d MA/30d Vol) + 통계적 멀티팩터 CDF 융합
    - 성과: CAGR 13.01% | Sharpe 0.92 | MDD -25.5%
"""

    growth_weight = " 0%" if is_danger else "38%"
    defense_weight = "38%" if is_danger else " 0%"
    
    mail_content += f"""
{line}
2. 전략적 자산 배분 제안
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
        mail_content += "!!! [🚨] 이중 확정 위험: 방어 자산 전환 !!!\n- 기술지표와 시장 심리가 모두 약세장 진입에 동의했습니다.\n- QQQ 비중을 전량(38%) 매도하고 JEPI(38%)로 교체하세요.\n"
    elif m1_danger:
        mail_content += "!!! [⚖️] 주의: 기술지표 약세이나 심리 지수가 방어 중 !!!\n- 일시적 노이즈일 가능성이 높습니다. 포지션을 유지하며 관망하세요.\n"
    else:
        mail_content += "!!! [✅] 상태 평온: 공격적 포지션 유지 !!!\n- 시장의 추세와 심리가 모두 우호적인 영역에 있습니다.\n"

    # 2. 로컬 저장용 프리미엄 마크다운 포맷
    md_report = f"""# 🚀 실전 투자 지표 리포트 ({today_str})

## 📊 종합 시장 신호: **{status_title}**

> **권장 스탠스**: {'🛡️ 방어적 리밸런싱 (JEPI 전환)' if is_danger else '🔥 공격적 자산 운용 (QQQ 유지)'}

---

## 1. 하이브리드 엔진 정밀 분석

### 🔍 시그널 판정
- **최종 상태**: {'🚨 **DANGER (위험)**' if is_danger else '✅ **NORMAL (정상)**'}
- **판정 근거**: {signal_info.get('reason', '정상 범위 내 동작 중')}

### 📈 데이터 디테일
- **기술적 위기 감지 (Sentinel)**: `{'ON' if m1_danger else 'OFF'}`
- **멀티팩터 심리 점수 (Validator)**: **{mf_score:.1f}** / 100
  - `[Fear 0 {bar} 100 Greed]`

---

## 2. 전략적 자산 배분 제안

| Ticker | 역할 | 기본 비중 | **권장 비중** | 액션 |
| :--- | :--- | :---: | :---: | :--- |
| **SCHD** | 배당 코어 | 38% | 38% | **HOLD** |
| **QQQ** | 성장 엔진 | 38% | **{growth_weight}** | {'🚨 SELL' if is_danger else '✅ HOLD'} |
| **JEPI** | 하락 방어 | 0% | **{defense_weight}** | {'🚀 BUY' if is_danger else '-'} |
| **KS200** | 국내 시장 | 19% | 19% | HOLD |
| **GOLD** | 안전 자산 | 5% | 5% | HOLD |

---

## 💡 투자 가이드
"""
    if is_danger:
        md_report += "> [!CAUTION]\n> **이중 확정 위험: 방어 자산 전환**\n> 기술지표와 시장 심리가 모두 약세장 진입에 동의했습니다. QQQ 전량을 JEPI로 교체하십시오.\n"
    elif m1_danger:
        md_report += "> [!IMPORTANT]\n> **주의: 기술지표 약세이나 심리 지수가 유효**\n> 일시적 노이즈일 가능성이 높습니다. 포지션을 유지하며 관망하십시오.\n"
    else:
        md_report += "> [!NOTE]\n> **상태 평온: 공격적 포지션 유지**\n> 시장의 추세와 심리가 모두 우호적입니다. 성장을 온전히 누리시기 바랍니다.\n"

    md_report += f"\n---\n*본 리포트는 ANTIGRAVITY HYBRID 엔진에 의해 자동 생성되었습니다. ({today_str})*"
    
    # 로컬 파일 저장
    report_path = BASE_DIR / "latest_report.md"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"✓ Markdown 리포트 생성 완료: {report_path}")
    except Exception as e:
        print(f"Markdown 저장 실패: {e}")
    
    return mail_content

def main():
    config = load_config()
    
    print("\n[최적화 하이브리드 엔진 가동 중]")
    print("  - 엔진 로직: Optimized Basic + Multifactor CDF Fusion")
    
    # 융합 시그널 탐지 실행
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
