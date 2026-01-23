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

    # SMA 110/250 Detail
    current_px = signal_info.get('current_price', 0)
    ma110 = signal_info.get('ma110', 0)
    ma250 = signal_info.get('ma250', 0)
    def_asset = signal_info.get('defensive_asset', 'BIL')
    
    mail_content = f"""{line}
📅 {today_str} DAILY MARKET INTELLIGENCE (Golden Combo 110/250)
{line}

[종합 시장 신호] : {status_title}
[권장 스탠스]     : {f'🛡️ 방어적 자산 전환 ({def_asset})' if is_danger else '🔥 공격적 자산 운용 (QQQ 유지)'}

{line}
1. 시장 지표 분석 (Dual SMA 110 & 250)
{line}

(1) 시그널 판정
    - 최종 상태: {'[🚨 DANGER (위험)]' if is_danger else '[✅ NORMAL (정상)]'}
    - 판정 근거: QQQ 가격 vs {ma110:.1f}(중기) & {ma250:.1f}(장기) 이평선 확정 신호

(2) 세부 데이터 분석
    - QQQ 현재가 : ${current_px:.2f}
    - SMA 110선  : ${ma110:.2f}
    - SMA 250선  : ${ma250:.2f}

(3) 전략 엔진 (Golden Combo)
    - 로직: 110일선과 250일선을 동시에 넘어야 상태 전환 (Hysteresis 적용)
    - 특징: 매매 횟수 32% 감소 및 하락장 방어력 극대화

{line}
2. 전략적 자산 배분 제안
{line}

| 자산명 | 기본 비중 | 권장 비중 | 실전 대응 |
|--------|-----------|-----------|-----------|
| 전략자산 |    55%    |    55%    | {f"🛡️ {def_asset} 매수" if is_danger else "✅ QQQ/KOSPI 유지"} |
| SPY    |    35%    |    35%    | 코어 포지션 유지 |
| GOLD   |    10%    |    10%    | 안전 자산 유지 |

{line}
3. 투자 핵심 가이드
{line}
"""
    if is_danger:
        mail_content += f"!!! [🚨] 하락 추세 확정: 방어 자산 전환 !!!\n- 시장이 장기 하락 트렌드로 진입했습니다.\n- 전략 자산(55%)을 최적 방어 자산인 {def_asset}로 교체하세요.\n"
    else:
        mail_content += "!!! [✅] 상승 추세 지속: 공격적 포지션 유지 !!!\n- 시장의 중장기 추세가 모두 우호적인 영역에 있습니다.\n- QQQ와 국내 대형주 비중을 유지하며 수익을 극대화하세요.\n"

    # 2. 로컬 저장용 프리미엄 마크다운 포맷
    md_report = f"""# 🚀 실전 투자 지표 리포트 ({today_str})

## 📊 종합 시장 신호: **{status_title}**

> **권장 스탠스**: {f'🛡️ 방어적 자산 전환 ({def_asset})' if is_danger else '🔥 공격적 자산 운용 (QQQ 유지)'}

---

## 1. 시장 지표 분석 (Dual SMA 110/250)

### 🔍 시그널 판정
- **최종 상태**: {'🚨 **DANGER (위험)**' if is_danger else '✅ **NORMAL (정상)**'}
- **판정 근거**: 110일(중기) 및 250일(장기) 이평선 동시 상회/하회 기반 확정 신호

### 📈 데이터 디테일
- **QQQ 현재가**: `${current_px:.2f}`
- **SMA 110선 (중기)**: `${ma110:.2f}`
- **SMA 250선 (장기)**: `${ma250:.2f}`

---

## 2. 전략적 자산 배분 제안

| 자산명 | 역할 | 기본 비중 | **권장 비중** | 액션 |
| :--- | :--- | :---: | :---: | :--- |
| **전략 자산** | 수익 엔진 | 55% | **55%** | {f'🛡️ BUY {def_asset}' if is_danger else '✅ HOLD QQQ/KOSPI'} |
| **SPY** | 시장 코어 | 35% | **35%** | **HOLD** |
| **GOLD** | 안전 자산 | 10% | **10%** | **HOLD** |

---

## 💡 투자 가이드
"""
    if is_danger:
        md_report += f"> [!CAUTION]\n> **추세 이탈: 방어 자산 매수**\n> QQQ 가격이 주요 이평선을 모두 하회했습니다. 전략 소매(55%)를 {def_asset}로 전량 교체하십시오.\n"
    else:
        md_report += "> [!NOTE]\n> **상태 평온: 상승 추세 지속**\n> QQQ 가격이 110/250일 이평선 위에서 안정적으로 움직이고 있습니다. 공격적 포지션을 유지하십시오.\n"

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
