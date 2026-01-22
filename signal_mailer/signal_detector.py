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
        # yfinance 최신 버전은 자체적으로 curl_cffi 세션을 관리합니다
        # 커스텀 세션을 전달하면 오류가 발생하므로 제거
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
            
            # 데이터가 비어있는지 확인
            if spy_data.empty or kospi_data.empty:
                print("⚠️ 데이터가 비어있습니다. (YFinance 응답 오류 의심)")
                return None, None

            # 타임존 제거 (안전한 접근)
            if hasattr(spy_data.index, 'tz') and spy_data.index.tz is not None:
                spy_data.index = spy_data.index.tz_localize(None)
            if hasattr(kospi_data.index, 'tz') and kospi_data.index.tz is not None:
                kospi_data.index = kospi_data.index.tz_localize(None)
                
            return spy_data, kospi_data
            
        except Exception as e:
            print(f"데이터 수집 오류: {e}")
            return None, None
    
    def calculate_danger_signal(self, spy_data):
        """위험신호 계산"""
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
        """신호 리포트 포맷팅"""
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
        current_status = 'DANGER' if is_danger else 'NORMAL'
        status_changed = (previous_status != current_status) if previous_status else False
        timestamp = signal_info['date'].strftime("%Y-%m-%d %H:%M:%S")
        
        body = f"Portfolio Signal Report\nTime: {timestamp}\nStatus: {current_status}\nReason: {signal_info['reason']}"
        
        return {
            'title': current_status,
            'body': body,
            'status_changed': status_changed,
            'status': current_status
        }
