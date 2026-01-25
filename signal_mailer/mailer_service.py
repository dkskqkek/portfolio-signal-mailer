# -*- coding: utf-8 -*-
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
import os
import logging
import numpy as np


class MailerService:
    """신호를 기록하고 이메일/텔레그램으로 전파하는 서비스 (v3.1 Final)"""

    @staticmethod
    def _to_py_type(val):
        """JSON 저장을 위한 Numpy 타입 변환 (Serialization Error 방지)"""
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        if isinstance(val, (np.floating, float)):
            return float(val) if not np.isnan(val) else 0.0
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return val

    def __init__(self, config):
        self.config = config
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """로거 설정"""
        log_file = self.config.get("log_file", "d:/gg/logs/system.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logger = logging.getLogger("MailerService")
        logger.setLevel(logging.INFO)

        # 중복 핸들러 방지
        if not logger.handlers:
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def send_email(self, subject, body_text, recipient_email=None, html_body=None):
        """이메일 발송 (Rich HTML 지원)"""
        email_config = self.config.get("email", {})
        debug_mode = self.config.get("debug_mode", False)

        sender_email = email_config.get("sender_email")
        sender_password = email_config.get("sender_password")
        recipient = recipient_email or email_config.get("recipient_email")

        if not all([sender_email, sender_password, recipient]):
            self.logger.warning("이메일 설정 불충분 (ID/PW/Recipient 확인 필요)")
            return {"success": False, "message": "Email configuration incomplete"}

        if debug_mode:
            self.logger.info(f"[DEBUG] 이메일 발송 시뮬레이션: {recipient}")
            return {"success": True, "message": "[DEBUG] Simulated"}

        try:
            message = MIMEMultipart()
            message["From"] = f"Antigravity System <{sender_email}>"
            message["To"] = recipient
            message["Subject"] = subject

            # HTML Body 결정 Logic:
            # 1. html_body(프리미엄 템플릿)가 있으면 그걸 그대로 사용
            # 2. 없으면 body_text(일반 텍스트)를 <pre> 태그로 감싸서 터미널 스타일로 변환
            if html_body:
                final_html = html_body
            else:
                # 기존 Fallback 로직 (Terminal Style)
                final_html = f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: 'Consolas', 'Courier New', monospace; background-color: #121212; padding: 20px; color: #e0e0e0; }}
                        .container {{ background-color: #1e1e1e; padding: 25px; border-radius: 12px; border: 1px solid #333; }}
                        pre {{ white-space: pre-wrap; font-size: 14px; color: #00ff41; background: #000; padding: 15px; border-radius: 5px; }}
                        .footer {{ margin-top: 25px; font-size: 11px; color: #666; text-align: center; }}
                    </style>
                </head>
                <body>
                    <div class="container"><pre>{body_text}</pre></div>
                    <div class="footer">Antigravity System (Text fallback)</div>
                </body>
                </html>
                """

            message.attach(MIMEText(final_html, "html", "utf-8"))

            smtp_server = email_config.get("smtp_server", "smtp.gmail.com")
            smtp_port = email_config.get("smtp_port", 587)

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(message)

            self.logger.info(f"✓ 이메일 발송 성공: {recipient}")
            return {"success": True, "message": "Sent"}
        except Exception as e:
            self.logger.error(f"이메일 발송 실패: {e}")
            return {"success": False, "message": str(e)}

    def load_history(self):
        """이전 신호 상태 로딩"""
        history_file = self.config.get("history_file", "d:/gg/data/signal_history.json")
        if not os.path.exists(history_file):
            return {}
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"히스토리 로드 에러: {e}")
            return {}

    def save_history(self, signal_info):
        """오늘의 판단 결과 저장 (Numpy 호환성 필터 적용)"""
        history_file = self.config.get("history_file", "d:/gg/data/signal_history.json")
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        history = self.load_history()

        # 신규 항목 데이터 정제 (Numpy 타입을 Python 표준 타입으로 변환)
        new_entry = {k: self._to_py_type(v) for k, v in signal_info.items()}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history[timestamp] = new_entry

        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            self.logger.info(f"✓ 히스토리 저장 완료: {signal_info['status_label']}")
        except Exception as e:
            self.logger.error(f"히스토리 저장 실패: {e}")

    def get_previous_status(self):
        """Hysteresis 로직을 위한 전일 상태 조회"""
        history = self.load_history()
        if not history:
            return None

        # 마지막 타임스탬프 기준으로 상태 추출
        try:
            sorted_keys = sorted(history.keys())
            latest_key = sorted_keys[-1]
            return history[latest_key].get("status_label")
        except:
            return None
