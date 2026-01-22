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

def generate_reports(today_str, status_title, is_danger, signal_info, config):
    """최적화 하이브리드(Fusion) 모델 리포트 생성 (메일용/로컬용 분리)"""
    mf_score = signal_info.get('mf_score', 50.0)
    m1_danger = signal_info.get('m1_danger', False)
    
    # 0. 설정에서 자산 관리 정보 추출
    portfolio_cfg = config.get('portfolio', {})
    mode_cfg = portfolio_cfg.get('danger_mode' if is_danger else 'normal_mode', {})
    
    # Tickers and Weights from config
    schd_w = f"{mode_cfg.get('schd_weight', 0)*100:.0f}%"
    qqq_w = f"{mode_cfg.get('qqq_weight', 0)*100:.0f}%"
    jepi_w = f"{mode_cfg.get('jepi_weight', 0)*100:.0f}%"
    ks200_w = f"{mode_cfg.get('ks200_weight', 0)*100:.0f}%"
    gold_w = f"{mode_cfg.get('gold_weight', 0)*100:.0f}%"
    
    # 1. 메일용 텍스트 포맷 (기존의 텍스트+기호 방식)
    line = "=" * 60
    bar_len = 20
    filled = int(mf_score / 100 * bar_len)
    bar = "■" * filled + "□" * (bar_len - filled)

    # SMA 150 Detail
    current_px = signal_info.get('current_price', 0)
    ma_val = signal_info.get('ma_value', 0)
    ma_status = "상회 (정상)" if current_px > ma_val else "하회 (🚨위험)"
    
    mail_content = f"""{line}
📅 {today_str} DAILY MARKET INTELLIGENCE (Pure SMA 150)
{line}

[종합 시장 신호] : {status_title}
[권장 스탠스]     : {'방어적 리밸런싱 (JEPI 전환)' if is_danger else '공격적 자산 운용 (QQQ 유지)'}

{line}
1. 시장 지표 분석 (Price vs SMA 150)
{line}

(1) 시그널 판정
    - 최종 상태: {'[🚨 DANGER (위험)]' if is_danger else '[✅ NORMAL (정상)]'}
    - 판정 근거: {signal_info.get('reason', '정상 범위 내 동작 중')}

(2) 세부 데이터 분석
    - QQQ 현재가 : ${current_px:.2f}
    - SMA 150선 : ${ma_val:.2f}
    - 이평선 상태: {ma_status}

(3) 전략 엔진 (SMA 150 Only)
    - 로직: QQQ 가격이 150일 단순 이동평균선(SMA) 위에 있으면 유지, 아래면 매도.
    - 성과: CAGR 12.1% | Sharpe 1.13 | MDD -15.6% (2020.06~현재)

{line}
2. 전략적 자산 배분 제안
{line}

(Ticker) | (기본 비중) | (권장 비중) | (Action)
------------------------------------------------------------
 SCHD    |    38%     |    {schd_w}     |   HOLD
 QQQ     |    38%     |    {qqq_w}     |   {'SELL' if is_danger else 'HOLD'}
 JEPI    |     0%     |    {jepi_w}     |   {'BUY ' if is_danger else ' -  '}
 KS200   |    19%     |    {ks200_w}     |   HOLD
 GLD     |     5%     |    {gold_w}     |   HOLD
------------------------------------------------------------

{line}
3. 투자 핵심 가이드
{line}
"""
    if is_danger:
        mail_content += f"!!! [🚨] 이중 확정 위험: 방어 자산 전환 !!!\n- 기술지표와 시장 심리가 모두 약세장 진입에 동의했습니다.\n- QQQ 비중을 전량 매도하고 JEPI({jepi_w})로 교체하세요.\n"
    elif m1_danger:
        mail_content += "!!! [⚖️] 주의: 기술지표 약세이나 심리 지수가 방어 중 !!!\n- 일시적 노이즈일 가능성이 높습니다. 포지션을 유지하며 관망하세요.\n"
    else:
        mail_content += "!!! [✅] 상태 평온: 공격적 포지션 유지 !!!\n- 시장의 추세와 심리가 모두 우호적인 영역에 있습니다.\n"

    # 2. 로컬 저장용 프리미엄 마크다운 포맷
    md_report = f"""# 🚀 실전 투자 지표 리포트 ({today_str})

## 📊 종합 시장 신호: **{status_title}**

> **권장 스탠스**: {'🛡️ 방어적 리밸런싱 (JEPI 전환)' if is_danger else '🔥 공격적 자산 운용 (QQQ 유지)'}

---

## 1. 시장 지표 분석 (SMA 150)

### 🔍 시그널 판정
- **최종 상태**: {'🚨 **DANGER (위험)**' if is_danger else '✅ **NORMAL (정상)**'}
- **판정 근거**: {signal_info.get('reason', '정상 범위 내 동작 중')}

### 📈 데이터 디테일
- **QQQ 현재가**: `${current_px:.2f}`
- **SMA 150선**: `${ma_val:.2f}`
- **이평선 상태**: **{ma_status}**

---

## 2. 전략적 자산 배분 제안

| Ticker | 역할 | 기본 비중 | **권장 비중** | 액션 |
| :--- | :--- | :---: | :---: | :--- |
| **SCHD** | 배당 코어 | 38% | {schd_w} | **HOLD** |
| **QQQ** | 성장 엔진 | 38% | **{qqq_w}** | {'🚨 SELL' if is_danger else '✅ HOLD'} |
| **JEPI** | 하락 방어 | 0% | **{jepi_w}** | {'🚀 BUY' if is_danger else '-'} |
| **KS200** | 국내 시장 | 19% | {ks200_w} | HOLD |
| **GLD** | 안전 자산 | 5% | {gold_w} | HOLD |

---

## 💡 투자 가이드
"""
    if is_danger:
        md_report += f"> [!CAUTION]\n> **추세 이탈: 위험 자산 매도**\n> QQQ 가격이 150일 이평선을 하회했습니다. 자산을 JEPI({jepi_w})로 교체하십시오.\n"
    else:
        md_report += "> [!NOTE]\n> **상세 평온: 상승 추세 지속**\n> QQQ 가격이 150일 이평선 위에서 안정적으로 움직이고 있습니다. 공격적 포지션을 유지하십시오.\n"

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
    
    text_report = generate_reports(today_str, status_title, is_danger, signal_info, config)
    
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
