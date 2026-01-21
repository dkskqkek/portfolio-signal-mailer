# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SignalDetector:
    """QQQ->XLP 전환 신호를 감지하는 클래스"""
    
    def __init__(self):
        self.spy = yf.Ticker("SPY")
        self.schd = yf.Ticker("SCHD")
        self.qqq_ticker = yf.Ticker("QQQ")
        self.xlp_ticker = yf.Ticker("XLP")
        self.kospi200 = yf.Ticker("^KS200")
        self.gld_ticker = yf.Ticker("GLD")
        
    def fetch_data(self, days_back=300):
        """최근 데이터 수집"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            spy_data = self.spy.history(start=start_date, end=end_date)['Close']
            kospi_data = self.kospi200.history(start=start_date, end=end_date)['Close']
            
            # 타임존 제거
            if spy_data.index.tz is not None:
                spy_data.index = spy_data.index.tz_localize(None)
            if kospi_data.index.tz is not None:
                kospi_data.index = kospi_data.index.tz_localize(None)
                
            return spy_data, kospi_data
            
        except Exception as e:
            print(f"데이터 수집 오류: {e}")
            return None, None
    
    def calculate_danger_signal(self, spy_data):
        """위험신호 계산
        
        위험신호 발생 조건:
        1. 20일 이동평균(로그수익률) < 25 percentile
        2. 20일 변동성(로그수익률) > 75 percentile
        
        Returns:
            dict: {
                'is_danger': bool,
                'reason': str,
                'date': datetime,
                'ma20': float,
                'volatility_20': float,
                'ma_threshold': float,
                'vol_threshold': float
            }
        """
        if spy_data is None or len(spy_data) < 20:
            return {
                'is_danger': False,
                'reason': '데이터 부족',
                'date': datetime.now(),
                'error': True
            }
        
        # 로그 수익률 계산
        log_returns = np.log(spy_data.values[1:] / spy_data.values[:-1])
        
        # 20일 이동평균과 변동성
        ma20_returns = pd.Series(log_returns).rolling(20).mean().values
        std20_returns = pd.Series(log_returns).rolling(20).std().values
        
        # 임계값 (전체 데이터 기준 percentile)
        ma_threshold = np.nanpercentile(ma20_returns, 25)
        vol_threshold = np.nanpercentile(std20_returns, 75)
        
        # 최신 값
        latest_ma = ma20_returns[-1]
        latest_vol = std20_returns[-1]
        
        # 위험신호 판정
        is_danger = False
        reason = ""
        
        if latest_ma < ma_threshold:
            is_danger = True
            reason = f"20일 이동평균({latest_ma:.4f}) < 임계값({ma_threshold:.4f})"
        
        if latest_vol > vol_threshold:
            is_danger = True
            if reason:
                reason += " AND "
            reason += f"20일 변동성({latest_vol:.4f}) > 임계값({vol_threshold:.4f})"
        
        if not is_danger:
            reason = "정상 상태: 신호 없음"
        
        return {
            'is_danger': is_danger,
            'reason': reason,
            'date': datetime.now(),
            'ma20': latest_ma,
            'volatility_20': latest_vol,
            'ma_threshold': ma_threshold,
            'vol_threshold': vol_threshold,
            'error': False
        }
    
    def detect(self):
        """신호 감지 실행"""
        spy_data, kospi_data = self.fetch_data()
        signal_info = self.calculate_danger_signal(spy_data)
        
        return signal_info
    
    @staticmethod
    def format_signal_report(signal_info, previous_status=None):
        """신호 리포트 포맷팅
        
        Returns:
            dict: {
                'title': str,
                'body': str,
                'status_changed': bool
            }
        """
        is_danger = signal_info.get('is_danger', False)
        error = signal_info.get('error', False)
        
        if error:
            return {
                'title': '신호 감지 오류',
                'body': f"신호 감지 중 오류 발생:\n{signal_info.get('reason', 'Unknown error')}",
                'status_changed': False,
                'status': 'ERROR'
            }
        
        # 신호 상태 판정
        if is_danger:
            current_status = 'DANGER (QQQ->XLP 전환)'
        else:
            current_status = 'NORMAL (QQQ 유지)'
        
        # 상태 변화 확인
        status_changed = (previous_status != current_status) if previous_status else False
        
        # 리포트 본문
        timestamp = signal_info['date'].strftime("%Y-%m-%d %H:%M:%S")
        
        body = f"""
포트폴리오 신호 리포트
{'='*60}

[감지 시간]
{timestamp}

[신호 상태]
{current_status}

[상태 변화]
{'✓ 변화 있음 (메일 발송 필요)' if status_changed else '상태 유지'}

[상세 정보]
- 20일 이동평균: {signal_info['ma20']:.6f}
- 20일 변동성: {signal_info['volatility_20']:.6f}
- MA 임계값: {signal_info['ma_threshold']:.6f}
- 변동성 임계값: {signal_info['vol_threshold']:.6f}

[판정 사유]
{signal_info['reason']}

[포지션 조정 가이드]
"""
        
        if is_danger:
            body += """
위험신호 발생! 다음과 같이 포지션 조정을 권장합니다:

현재 포지션 (정상 상태):
  - SCHD:   34%
  - QQQ:    34%  <- QQQ 축소
  - XLP:     0%  <- XLP 확대
  - KOSPI:  17%
  - GOLD:   15%

조정 후 포지션 (위험 회피):
  - SCHD:   34%
  - QQQ:     0%  (축소 완료)
  - XLP:    34%  (확대 완료)
  - KOSPI:  17%
  - GOLD:   15%

조정 방법: QQQ 포지션을 매각하여 XLP로 전환하세요.
"""
        else:
            body += """
신호 없음: 현재 포지션 유지

포지션 (정상 상태):
  - SCHD:   34%
  - QQQ:    34%
  - XLP:     0%
  - KOSPI:  17%
  - GOLD:   15%

추가 조정 불필요합니다.
"""
        
        return {
            'title': current_status,
            'body': body,
            'status_changed': status_changed,
            'status': 'DANGER' if is_danger else 'NORMAL'
        }
