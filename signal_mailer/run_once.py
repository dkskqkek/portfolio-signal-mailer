# -*- coding: utf-8 -*-
"""
한 번만 실행하는 신호 감지 스크립트 (GitHub Actions용)
"""
import sys
import os

# 현재 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_detector import SignalDetector
from mailer import MailerService
import yaml

def main():
    """신호 감지 및 메일 발송 실행"""
    
    # config.yaml 로드
    config_path = 'signal_mailer/config.yaml'
    if not os.path.exists(config_path):
        print(f"✗ 설정 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("포트폴리오 신호 감지 (GitHub Actions)")
    print("="*60)
    
    # 설정 로드 및 서비스 초기화
    mailer = MailerService(config)
    previous_status = mailer.get_previous_status()
    
    # 신호 감지 (이전 상태를 반영하여 Hysteresis 적용)
    detector = SignalDetector()
    signal_info = detector.detect(previous_status=previous_status)
    
    # 신호 리포트 생성
    report = SignalDetector.format_signal_report(signal_info, previous_status)
    
    print(f"\n신호 상태: {report['status']}")
    print(f"이전 상태: {previous_status or 'None'}")
    print(f"상태 변화: {report['status_changed']}")
    
    # 신호 이력 저장
    mailer.save_history(report['status'], signal_info)
    print("✓ 신호 이력 저장 완료")
    
    # 상태 변화가 있을 때만 메일 발송
    if report['status_changed']:
        print("\n상태 변화 감지! 메일 발송 중...")
        
        email_config = config.get('email', {})
        subject_template = email_config.get('subject_template', "[신호] {status}")
        subject = subject_template.format(status=report['title'])
        
        result = mailer.send_email(subject, report['body'])
        
        if result['success']:
            print(f"✓ {result['message']}")
        else:
            print(f"✗ {result['message']}")
    else:
        print("✓ 상태 변화 없음. 메일 발송 스킵.")
    
    print("\n" + "="*60)
    print("신호 감지 완료")
    print("="*60)

if __name__ == '__main__':
    main()
