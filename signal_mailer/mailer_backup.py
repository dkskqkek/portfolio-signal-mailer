# -*- coding: utf-8 -*-
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
import os
import logging

class MailerService:
    """신호를 이메일로 발송하는 서비스"""
    
    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger('MailerService')
        logger.setLevel(logging.INFO)
        
        log_file = self.config.get('log_file', 'd:/gg/signal_mailer/mailer.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def send_email(self, subject, body, recipient_email=None):
        """이메일 발송
        
        Args:
            subject: 제목
            body: 본문
            recipient_email: 수신자 (기본값: config의 수신자)
        
        Returns:
            dict: {
                'success': bool,
                'message': str
            }
        """
        email_config = self.config.get('email', {})
        debug_mode = self.config.get('debug_mode', False)
        
        sender_email = email_config.get('sender_email')
        sender_password = email_config.get('sender_password')
        recipient = recipient_email or email_config.get('recipient_email')
        
        # 설정 검증
        if not all([sender_email, sender_password, recipient]):
            error_msg = "이메일 설정이 불완전합니다. config.yaml을 확인하세요."
            self.logger.warning(error_msg)
            return {
                'success': False,
                'message': error_msg
            }
        
        # 기본값 확인
        if 'your_email' in sender_email or 'your_app_password' in sender_password:
            error_msg = "config.yaml의 이메일 설정을 완료하세요."
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg
            }
        
        # 디버그 모드
        if debug_mode:
            self.logger.info(f"[DEBUG 모드] 이메일 발송 시뮬레이션")
            self.logger.info(f"  수신자: {recipient}")
            self.logger.info(f"  제목: {subject}")
            self.logger.info(f"  본문:\n{body}")
            return {
                'success': True,
                'message': '[DEBUG] 이메일 발송 시뮬레이션 완료'
            }
        
        try:
            # 이메일 구성
            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = recipient
            message['Subject'] = subject
            
            message.attach(MIMEText(body, 'html', 'utf-8'))
            
            # SMTP 발송
            smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = email_config.get('smtp_port', 587)
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)
            
            self.logger.info(f"✓ 이메일 발송 성공: {recipient}")
            return {
                'success': True,
                'message': f'이메일 발송 완료: {recipient}'
            }
            
        except Exception as e:
            error_msg = f"이메일 발송 실패: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg
            }
    
    def load_history(self):
        """신호 이력 로드"""
        history_file = self.config.get('history_file', 'd:/gg/signal_mailer/signal_history.json')
        
        if not os.path.exists(history_file):
            return {}
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def save_history(self, signal_status, signal_info):
        """신호 이력 저장"""
        history_file = self.config.get('history_file', 'd:/gg/signal_mailer/signal_history.json')
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        history = self.load_history()
        
        timestamp = datetime.now().isoformat()
        history[timestamp] = {
            'status': signal_status,
            'is_danger': bool(signal_info.get('is_danger', False)),
            'reason': str(signal_info.get('reason', '')),
            'mf_score': float(signal_info.get('mf_score', 0))
        }
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            self.logger.info(f"신호 이력 저장: {signal_status}")
        except Exception as e:
            self.logger.error(f"신호 이력 저장 실패: {e}")
    
    def get_previous_status(self):
        """이전 신호 상태 조회"""
        history = self.load_history()
        
        if not history:
            return None
        
        # 최신 항목 조회
        latest_timestamp = sorted(history.keys())[-1]
        latest_entry = history[latest_timestamp]
        
        if latest_entry.get('is_danger'):
            return 'DANGER'
        else:
            return 'NORMAL'
