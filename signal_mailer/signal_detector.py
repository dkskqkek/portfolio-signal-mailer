# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
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
        self.vix_ticker = yf.Ticker("^VIX")
        
    def fetch_data(self, days_back=450):
        """최근 데이터 및 지표용 선행 데이터 수집"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            # yfinance history 사용 시 auto_adjust=False 설정 권장 (배당 미포함 종가 산출 위해)
            spy_data = self.spy.history(start=start_date, end=end_date, auto_adjust=True)['Close']
            qqq_data = self.qqq_ticker.history(start=start_date, end=end_date, auto_adjust=True)['Close']
            kospi_data = self.kospi200.history(start=start_date, end=end_date, auto_adjust=False)['Close']
            vix_data = self.vix_ticker.history(start=start_date, end=end_date, auto_adjust=False)['Close']
            
            # 데이터가 비어있는지 확인
            if spy_data.empty or qqq_data.empty or kospi_data.empty or vix_data.empty:
                print("⚠️ 데이터가 비어있습니다. (YFinance 응답 오류 의심)")
                return None, None, None, None

            # 타임존 제거
            for df in [spy_data, qqq_data, kospi_data, vix_data]:
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
            return spy_data, qqq_data, kospi_data, vix_data
            
        except Exception as e:
            print(f"데이터 수집 오류: {e}")
            return None, None, None
    
    def calculate_multifactor_score(self, spy_data, vix_data, lookback=126):
        """사용자 제공 멀티팩터 CDF 스코어링 (0~100)"""
        if spy_data is None or vix_data is None or len(spy_data) < lookback:
            return 50.0 # 기본값
            
        # 1. EMA 200 이격도
        ema200 = spy_data.ewm(span=200, adjust=False).mean()
        ema_dist = (spy_data - ema200) / ema200
        
        # 2. RSI 14
        delta = spy_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain/loss.replace(0, np.nan))).fillna(100)
        
        # 정규화 함수 (CDF)
        def get_score(series, inv=False):
            m = series.rolling(lookback).mean()
            s = series.rolling(lookback).std()
            z = (series - m) / (s + 1e-6)
            score = norm.cdf(z.iloc[-1]) * 100
            return 100 - score if inv else score
            
        s_trend = get_score(ema_dist, inv=True)
        s_mom = get_score(rsi, inv=True)
        s_vol = get_score(vix_data, inv=False)
        
        # 최종 점수 (3일 평균 가중치)
        # 실제 운영에서는 최신 데이터의 CDF 점수를 가중 평균
        score = (s_trend * 0.2 + s_mom * 0.4 + s_vol * 0.4)
        return score

    def calculate_danger_signal(self, spy_data, qqq_data, vix_data):
        """
        [최적화 융합 모델] 위함신호 계산
        - Sentinel (M1): 15d MA / 30d Vol (SPY 기반)
        - Validator (M2): Multifactor CDF Score <= 40 (SPY/VIX 기반)
        - Hard Floor: QQQ SMA 150 (설정값 기반)
        """
        if spy_data is None or qqq_data is None or len(spy_data) < 126:
            return {'is_danger': False, 'reason': '데이터 부족', 'date': datetime.now(), 'error': True}
        
        # 1. 기존 Sentinel 시그널 계산 (SPY 기반)
        log_returns = np.log(spy_data.values[1:] / spy_data.values[:-1])
        ma15_returns = pd.Series(log_returns).rolling(15).mean().values
        std30_returns = pd.Series(log_returns).rolling(30).std().values
        
        ma_threshold = np.nanpercentile(ma15_returns, 25)
        vol_threshold = np.nanpercentile(std30_returns, 65)
        
        latest_ma = ma15_returns[-1]
        latest_vol = std30_returns[-1]
        
        m1_danger = (latest_ma < ma_threshold) or (latest_vol > vol_threshold)
        
        # 2. Multifactor Validator 점수 계산
        mf_score = self.calculate_multifactor_score(spy_data, vix_data)
        
        # 3. 전략 모드 및 설정 로드
        import yaml
        try:
            with open('d:/gg/signal_mailer/config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                strategy_mode = config.get('strategy', {}).get('mode', 'hybrid')
                ma_long = config.get('strategy', {}).get('ma_period_long', 150)
                ma_short = config.get('strategy', {}).get('ma_period_short', 50)
        except:
            strategy_mode = 'hybrid'
            ma_long = 150
            ma_short = 50

        is_danger = False
        reason = ""
        
        # --- 전략별 로직 ---
        
        # A. Hybrid (Fusion + QQQ SMA Floor) - 추천/기본
        if strategy_mode == 'hybrid':
            # 1. Fusion 로직 (Sentinel & Validator 동시 만족)
            fusion_danger = m1_danger and (mf_score <= 40)
            
            # 2. Hard Floor 체크 (QQQ 기준 SMA 하한선)
            ma_floor = qqq_data.rolling(window=ma_long).mean().iloc[-1]
            current_price = qqq_data.iloc[-1]
            floor_breached = current_price < ma_floor
            
            if floor_breached:
                is_danger = True
                reason = f"DANGER (Hard Floor): QQQ 가격이 {ma_long}일 단순이평선 하회 ({current_price:.2f} < {ma_floor:.2f})"
            elif fusion_danger:
                is_danger = True
                reason = f"DANGER (Fusion): 이중 확정 기술/심리 위기 (Score: {mf_score:.1f})"
            else:
                is_danger = False
                reason = f"NORMAL: QQQ 이평선 지지중 & Fusion 안정 (Score: {mf_score:.1f})"

        # B. Fusion (순수 기존 로직)
        elif strategy_mode == 'fusion':
            if m1_danger and mf_score <= 40:
                is_danger = True
                reason = f"이중 확정 위험: 기술지표 위기(Sentinel) & 심리지수 과열({mf_score:.1f}점)"
            else:
                is_danger = False
                reason = f"정상 상태: Fusion 데이터 안정 (Score: {mf_score:.1f}점)"

        # C. Faber (Meb Faber Style - Monthly)
        elif strategy_mode == 'faber':
            monthly_data = qqq_data.resample('M').last() # QQQ 기준으로 변경 가능
            if len(monthly_data) < 12:
                is_danger = False
                reason = "데이터 부족 (Faber)"
            else:
                ma10mo = monthly_data.rolling(10).mean().iloc[-1]
                current_month_close = monthly_data.iloc[-1]
                if current_month_close < ma10mo:
                    is_danger = True
                    reason = f"DANGER (Faber): 월봉({current_month_close:.2f}) < 10개월 이평({ma10mo:.2f})"
                else:
                    is_danger = False
                    reason = f"NORMAL (Faber): 월봉({current_month_close:.2f}) > 10개월 이평({ma10mo:.2f})"

        # D. Golden Cross
        elif strategy_mode == 'golden_cross':
            ma_s = qqq_data.rolling(ma_short).mean().iloc[-1]
            ma_l = qqq_data.rolling(ma_long).mean().iloc[-1]
            if ma_s < ma_l:
                is_danger = True
                reason = f"DANGER (Death Cross): {ma_short}일({ma_s:.2f}) < {ma_long}일({ma_l:.2f})"
            else:
                is_danger = False
                reason = f"NORMAL (Golden Cross): {ma_short}일({ma_s:.2f}) > {ma_long}일({ma_l:.2f})"
        
        # E. MA Only (Pure SMA 150)
        elif strategy_mode == 'ma_only':
            ma_val = qqq_data.rolling(window=ma_long).mean().iloc[-1]
            current_price = qqq_data.iloc[-1]
            if current_price < ma_val:
                is_danger = True
                reason = f"DANGER (SMA {ma_long}): QQQ가 {ma_long}일 이평선을 하회했습니다. ({current_price:.2f} < {ma_val:.2f})"
            else:
                is_danger = False
                reason = f"NORMAL (SMA {ma_long}): QQQ가 {ma_long}일 이평선 위에 있습니다. ({current_price:.2f} > {ma_val:.2f})"
        
        else:
            is_danger = False
            reason = f"NORMAL (Undefined Mode: {strategy_mode})"

        return {
            'is_danger': is_danger,
            'reason': reason,
            'date': datetime.now(),
            'mf_score': mf_score,
            'm1_danger': m1_danger,
            'current_price': qqq_data.iloc[-1],
            'ma_value': qqq_data.rolling(window=ma_long).mean().iloc[-1] if 'ma_long' in locals() else None,
            'error': False,
            'strategy': strategy_mode
        }
    
    def detect(self):
        """신호 감지 실행"""
        spy_data, qqq_data, kospi_data, vix_data = self.fetch_data()
        return self.calculate_danger_signal(spy_data, qqq_data, vix_data)
    
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
        
        # SMA 150 정보 추가
        ma_info = ""
        if 'ma_value' in signal_info and signal_info['ma_value']:
            ma_info = f"\nQQQ Current: {signal_info['current_price']:.2f}\nQQQ SMA 150: {signal_info['ma_value']:.2f}"
            
        body = f"Portfolio Signal Report\nTime: {timestamp}\nStatus: {current_status}\nReason: {signal_info['reason']}{ma_info}"
        
        return {
            'title': current_status,
            'body': body,
            'status_changed': status_changed,
            'status': current_status
        }
