# -*- coding: utf-8 -*-
"""
포트폴리오 신호 자동 감지 및 메일 발송 시스템

매일 아침 설정된 시간에 자동으로 실행되어:
1. SPY 기반 위험신호 감지
2. 신호 발생 시 QQQ->XLP 전환 권장
3. 상태 변화가 있을 때만 메일 발송
4. 모든 신호 이력 기록

사용법:
    python main.py
    
설정:
    config.yaml에서 다음을 수정하세요:
    - email.sender_email: 발송 이메일 주소
    - email.sender_password: 앱 비밀번호 (Gmail의 경우)
    - email.recipient_email: 수신 이메일 주소
"""

import yaml
import logging
import os
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from signal_detector import SignalDetector
from mailer import MailerService

class SignalMailerSystem:
    """신호 메일 발송 시스템"""
    
    def __init__(self, config_path='d:/gg/signal_mailer/config.yaml'):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.detector = SignalDetector()
        self.mailer = MailerService(self.config)
        self.scheduler = None
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path):
        """YAML 설정 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"설정 파일 로드 오류: {e}")
            return {}
    
    def _setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger('SignalMailerSystem')
        logger.setLevel(logging.INFO)
        
        log_file = self.config.get('log_file', 'd:/gg/signal_mailer/mailer.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def check_and_send_signal(self):
        """신호 확인 및 메일 발송 (스케줄러 콜백)"""
        self.logger.info("="*60)
        self.logger.info("신호 확인 시작")
        self.logger.info("="*60)
        
        try:
            # 신호 감지
            signal_info = self.detector.detect()
            
            # 이전 상태 조회
            previous_status = self.mailer.get_previous_status()
            
            # 신호 리포트 생성
            report = SignalDetector.format_signal_report(signal_info, previous_status)
            
            self.logger.info(f"신호 상태: {report['status']}")
            self.logger.info(f"이전 상태: {previous_status or 'None'}")
            self.logger.info(f"상태 변화: {report['status_changed']}")
            
            # 신호 이력 저장
            self.mailer.save_history(report['status'], signal_info)
            
            # 상태 변화가 있을 때만 메일 발송
            if report['status_changed']:
                self.logger.info("상태 변화 감지! 메일 발송 중...")
                
                email_config = self.config.get('email', {})
                subject_template = email_config.get('subject_template', "[신호] {status}")
                subject = subject_template.format(status=report['title'])
                
                result = self.mailer.send_email(subject, report['body'])
                
                if result['success']:
                    self.logger.info(f"✓ {result['message']}")
                else:
                    self.logger.error(f"✗ {result['message']}")
            else:
                self.logger.info("상태 변화 없음. 메일 발송 스킵.")
            
        except Exception as e:
            self.logger.error(f"신호 확인 중 오류: {e}")
        
        self.logger.info("신호 확인 완료\n")
    
    def start(self):
        """스케줄러 시작"""
        self.scheduler = BackgroundScheduler()
        
        # 스케줄 설정
        scheduler_config = self.config.get('scheduler', {})
        run_time = scheduler_config.get('run_time', '09:00')  # HH:MM
        timezone = scheduler_config.get('timezone', 'Asia/Seoul')
        
        hour, minute = run_time.split(':')
        
        self.logger.info(f"스케줄러 시작")
        self.logger.info(f"  실행 시간: 매일 {run_time} (타임존: {timezone})")
        self.logger.info(f"  다음 실행: 확인 중...")
        
        # Cron 트리거로 매일 지정된 시간에 실행
        trigger = CronTrigger(
            hour=int(hour),
            minute=int(minute),
            timezone=timezone
        )
        
        self.scheduler.add_job(
            self.check_and_send_signal,
            trigger=trigger,
            id='signal_check_job',
            name='신호 확인 및 메일 발송'
        )
        
        self.scheduler.start()
        
        self.logger.info("스케줄러 활성화됨")
        self.logger.info("Ctrl+C를 눌러 종료하세요")
        
        # 첫 실행 (즉시 테스트)
        self.logger.info("\n[초기화] 첫 신호 확인 실행...")
        self.check_and_send_signal()
        
        # 스케줄러 유지
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("\n스케줄러 종료 중...")
            self.scheduler.shutdown()
            self.logger.info("종료되었습니다")
    
    def test_signal(self):
        """신호 감지 테스트 (실시간 실행)"""
        print("\n신호 감지 테스트 실행 중...\n")
        self.check_and_send_signal()
    
    def test_email(self, recipient_email=None):
        """이메일 발송 테스트"""
        print("\n테스트 이메일 발송 중...\n")
        
        test_subject = "[포트폴리오] 테스트 메일"
        test_body = f"""
테스트 메일입니다.

발송 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

이메일 설정이 올바르게 완료되었습니다.

포트폴리오 신호 메일러가 정상 작동 중입니다.
"""
        
        result = self.mailer.send_email(test_subject, test_body, recipient_email)
        print(f"결과: {result['message']}")


def print_usage():
    """사용법 출력"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║          포트폴리오 신호 자동 메일 발송 시스템                      ║
║                                                                    ║
║  QQQ->XLP 전환 신호를 매일 아침 자동으로 감지하고 메일 발송        ║
╚════════════════════════════════════════════════════════════════════╝

[ 1단계 ] config.yaml 설정
  - email.sender_email: Gmail 주소 입력
  - email.sender_password: 앱 비밀번호 입력 (Gmail의 경우)
  - email.recipient_email: 수신할 이메일 주소 입력
  - scheduler.run_time: 실행 시간 설정 (기본값: 09:00)

[ 2단계 ] 테스트 실행
  python main.py --test-signal      # 신호 감지 테스트
  python main.py --test-email       # 이메일 발송 테스트

[ 3단계 ] 시작
  python main.py                    # 스케줄러 시작 (Ctrl+C로 종료)

[ 참고사항 ]
- Gmail 앱 비밀번호: https://myaccount.google.com/apppasswords
- 신호 이력: signal_history.json에 기록됨
- 로그: mailer.log에 저장됨
- debug_mode: true 설정 시 실제 메일 발송 안함

    """)


if __name__ == '__main__':
    import sys
    
    print_usage()
    
    system = SignalMailerSystem()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test-signal':
            system.test_signal()
        elif sys.argv[1] == '--test-email':
            system.test_email()
        else:
            print(f"알 수 없는 옵션: {sys.argv[1]}")
    else:
        system.start()
