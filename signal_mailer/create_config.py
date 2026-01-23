# -*- coding: utf-8 -*-
"""
GitHub Secrets에서 환경 변수를 읽어 config.yaml 생성
"""
import os
import yaml

config = {
    'scheduler': {
        'run_time': '09:00',
        'timezone': 'Asia/Seoul'
    },
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': os.getenv('SENDER_EMAIL', 'your_email@gmail.com'),
        'sender_password': os.getenv('SENDER_PASSWORD', 'your_password'),
        'recipient_email': os.getenv('RECIPIENT_EMAIL', 'your_email@gmail.com'),
        'subject_template': '[포트폴리오 신호] {status}'
    },
    'strategy_info': {
        'description': 'QLD (2x Nasdaq) + Top-3 Defensive Ensemble',
        'signal_logic': 'Dual SMA (110/250) with Hysteresis',
        'weights': {
            'tactical': 0.45,
            'kospi': 0.20,
            'spy': 0.20,
            'gold': 0.15
        }
    },
    'gemini': {
        'api_key': os.getenv('GEMINI_API_KEY', ''),
        'enabled': bool(os.getenv('GEMINI_API_KEY'))
    },
    'history_file': 'signal_mailer/signal_history.json',
    'log_file': 'signal_mailer/mailer.log',
    'debug_mode': False
}

output_path = 'signal_mailer/config.yaml'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

print(f"✓ Config created: {output_path}")
print(f"  Sender: {config['email']['sender_email']}")
print(f"  Recipient: {config['email']['recipient_email']}")
print(f"  Gemini enabled: {config['gemini']['enabled']}")
