# -*- coding: utf-8 -*-
"""
매일 실행되는 신호 발송 스크립트
1. 신호를 감지하고
2. 상태 변화 여부와 상관없이 항상 메일을 발송합니다 (데일리 리포트)
3. 메일 제목에 날짜를 포함합니다.
"""
import sys
import os
import datetime

# 현재 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_detector import SignalDetector
from mailer import MailerService
import yaml

def main():
    """데일리 리포트 실행"""
    
    # config.yaml 로드
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    if not os.path.exists(config_path):
        print(f"✗ 설정 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("데일리 포트폴리오 리포트 생성 중...")
    print("="*60)
    
    # 신호 감지
    detector = SignalDetector()
    signal_info = detector.detect()
    
    # 이전 상태 조회
    mailer = MailerService(config)
    previous_status = mailer.get_previous_status()
    
    # 신호 리포트 생성
    report = SignalDetector.format_signal_report(signal_info, previous_status)
    
    print(f"\n신호 상태: {report['status']}")
    
    # 신호 이력 저장 (이력은 계속 남김)
    mailer.save_history(report['status'], signal_info)
    print("✓ 신호 이력 업데이트 완료")
    
    # 무조건 메일 발송
    print("\n데일리 리포트 메일 발송 중...")
    
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    email_config = config.get('email', {})
    
    # 제목 커스터마이징
    subject = f"[데일리 리포트] {today_str} 포트폴리오 신호: {report['title']}"
    
    # 본문에 설명 추가
    body_header = f"""
<h2>📅 {today_str} 데일리 리포트</h2>
<p>이 메일은 자동화 설정에 의해 매일 아침 발송됩니다.</p>
<hr>
"""
    full_body = body_header + report['body']
    
    result = mailer.send_email(subject, full_body)
    
    if result['success']:
        print(f"✓ {result['message']}")
    else:
        print(f"✗ {result['message']}")
        sys.exit(1) # 실패 시 에러 코드 반환
    
    print("\n" + "="*60)
    print("데일리 리포트 완료")
    print("="*60)

if __name__ == '__main__':
    main()
