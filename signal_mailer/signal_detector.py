# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")


class SignalDetector:
    """QLD + Top-3 Defensive Ensemble 전환 신호를 감지하는 클래스"""

    def __init__(self):
        self.spy = yf.Ticker("SPY")
        self.qqq_ticker = yf.Ticker("QQQ")
        self.kospi200 = yf.Ticker("^KS200")
        self.vix_ticker = yf.Ticker("^VIX")
        self.gld_ticker = yf.Ticker("GLD")
        self.def_pool = [
            "BTAL",
            "XLP",
            "XLU",
            "GLD",
            "FXY",
            "UUP",
            "MNA",
            "QAI",
            "DBC",
            "USFR",
            "GSY",
            "PFIX",
            "DBMF",
            "TAIL",
            "IVOL",
            "KMLM",
            "CTA",
            "PDBC",
            "SCHP",
            "TLT",
            "IEF",
            "BIL",
            "VXV",
        ]

    def fetch_data(self, days_back=500):
        """최근 데이터 및 지표용 선행 데이터 수집"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        try:
            core_tickers = ["SPY", "QQQ", "^KS200", "^VIX", "GLD", "BIL"]
            all_tickers = list(set(core_tickers + self.def_pool))

            # Group by ticker for more reliable extraction
            raw_data = yf.download(
                all_tickers,
                start=start_date,
                end=end_date,
                progress=False,
                group_by="ticker",
            )

            data_dict = {}
            for ticker in all_tickers:
                try:
                    if ticker in raw_data.columns.get_level_values(0):
                        t_data = raw_data[ticker]
                        col = "Adj Close" if "Adj Close" in t_data.columns else "Close"
                        data_dict[ticker] = t_data[col]
                except Exception:
                    pass

            data = pd.DataFrame(data_dict)

            # QQQ가 필수인데 누락된 경우 개별 재시도
            if "QQQ" not in data.columns or data["QQQ"].dropna().empty:
                qqq_fix = yf.download(
                    "QQQ", start=start_date, end=end_date, progress=False
                )
                data["QQQ"] = (
                    qqq_fix["Adj Close"]
                    if "Adj Close" in qqq_fix.columns
                    else qqq_fix["Close"]
                )

            if data.empty:
                print("⚠️ 데이터가 비어있습니다.")
                return None

            data = data.ffill()
            return data

        except Exception as e:
            print(f"데이터 수집 오류: {e}")
            return None

    def calculate_multifactor_score(self, data, lookback=126):
        """사용자 제공 멀티팩터 CDF 스코어링 (0~100)"""
        spy_data = data["SPY"]
        vix_data = data["^VIX"]

        # 1. EMA 200 이격도
        ema200 = spy_data.ewm(span=200, adjust=False).mean()
        ema_dist = (spy_data - ema200) / ema200

        # 2. RSI 14
        delta = spy_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan))).fillna(100)

        def get_score(series, inv=False):
            m = series.rolling(lookback).mean()
            s = series.rolling(lookback).std()
            z = (series - m) / (s + 1e-6)
            score = norm.cdf(z.iloc[-1]) * 100
            return 100 - score if inv else score

        s_trend = get_score(ema_dist, inv=True)
        s_mom = get_score(rsi, inv=True)
        s_vol = get_score(vix_data, inv=False)

        return s_trend * 0.2 + s_mom * 0.4 + s_vol * 0.4

    def calculate_danger_signal(self, data, previous_status=None):
        """
        [최적화 황금 조합] Dual SMA (110, 250) + Defensive Ensemble
        """
        if data is None or len(data) < 250:
            return {"is_danger": False, "reason": "데이터 부족", "error": True}

        # 1. Dual SMA 110/250 Hysteresis Logic
        curr_price = data["QQQ"].iloc[-1]
        ma110 = data["QQQ"].rolling(110).mean().iloc[-1]
        ma250 = data["QQQ"].rolling(250).mean().iloc[-1]

        if curr_price > ma110 and curr_price > ma250:
            status = "NORMAL"
        elif curr_price < ma110 and curr_price < ma250:
            status = "DANGER"
        else:
            status = previous_status if previous_status else "NORMAL"

        is_danger = status == "DANGER"

        # 2. Defensive Asset Selection (Top-3 Momentum Ensemble)
        # 8개월(168일) 수익률 기준 상위 3종 균등 배분 전략
        mom_returns = (
            data[self.def_pool]
            .pct_change(168)
            .iloc[-1]
            .dropna()
            .sort_values(ascending=False)
        )

        # Absolute Momentum Filter 적용 (모멘텀 > 0 인 것만)
        valid_assets = mom_returns[mom_returns > 0].head(3)

        if valid_assets.empty:
            defensive_assets = ["BIL"]
        else:
            defensive_assets = valid_assets.index.tolist()

        # 3. 추가 지표 (리포트용)
        mf_score = self.calculate_multifactor_score(data)
        rsi = (
            100
            - (
                100
                / (
                    1
                    + (
                        data["SPY"]
                        .diff()
                        .where(data["SPY"].diff() > 0, 0)
                        .rolling(14)
                        .mean()
                        / data["SPY"]
                        .diff()
                        .where(data["SPY"].diff() < 0, 0)
                        .abs()
                        .rolling(14)
                        .mean()
                    ).replace(0, np.nan)
                )
            ).fillna(100)
        ).iloc[-1]

        return {
            "is_danger": is_danger,
            "status_label": status,
            "defensive_assets": defensive_assets,
            "current_price": curr_price,
            "ma110": ma110,
            "ma250": ma250,
            "mf_score": mf_score,
            "rsi": rsi,
            "vix": data["^VIX"].iloc[-1],
            "date": datetime.now(),
            "error": False,
        }

    def detect(self, previous_status=None):
        """신호 감지 실행"""
        data = self.fetch_data()
        return self.calculate_danger_signal(data, previous_status)

    @staticmethod
    def format_signal_report(signal_info, previous_status=None):
        """최적화 황금 조합 리포트 포맷팅"""
        if signal_info.get("error"):
            return {
                "title": "ERROR",
                "body": f"오류: {signal_info.get('reason')}",
                "status": "ERROR",
            }

        is_danger = signal_info["is_danger"]
        current_status = signal_info["status_label"]
        # [Korean Defense Proxy Mapping]
        def_assets = signal_info["defensive_assets"]
        def_asset_str = ", ".join(def_assets)
        emoji = "🔴" if is_danger else "🟢"
        timestamp = signal_info["date"].strftime("%Y-%m-%d")

        # Action Label
        action = (
            f"DEFENSIVE SWITCH (to {def_asset_str})"
            if is_danger
            else "CORE HOLDING (QLD/KOSPI)"
        )

        korea_proxy_map = {
            "GLD": "ACE KRX금현물",
            "BIL": "TIGER/KODEX CD금리액티브",
            "IEF": "TIGER 미국채10년선물",
            "TLT": "ACE 미국30년국채액티브(H)",
            "UUP": "KOSEF 미국달러선물",
            "DBC": "ACE KRX금현물(대체)",  # Commodities fallback
        }

        def_korea = []
        for asset in def_assets:
            proxy = korea_proxy_map.get(asset, "TIGER CD금리액티브(기본)")
            def_korea.append(f"{asset}→{proxy}")
        def_korea_str = " / ".join(def_korea)

        body = f"""
============================================================
📅 [{timestamp}] PORTFOLIO STRATEGY BRIEFING
============================================================

[1] MARKET STATUS: {emoji} {current_status} (Optimized Dual SMA)
------------------------------------------------------------
현재 전략     : {action}
판단 근거     : QQQ 가격 vs Dual SMA (110, 250) 확정 신호
QQQ 현재가    : ${signal_info["current_price"]:.2f}
SMA 110 (중기): ${signal_info["ma110"]:.2f}
SMA 250 (장기): ${signal_info["ma250"]:.2f}

[2] TOP-3 DEFENSIVE ENSEMBLE (미국/국내 대응)
------------------------------------------------------------
미국 계좌 방어: {def_asset_str} (각 15% 배분)
국내 대안(Proxy): {def_korea_str}

※ 국내 계좌(ISA/연금) 간편 대응 가이드:
   👉 DANGER 시 [금현물 50% + CD금리 50%] 반반 전략 권장

[3] ACTIONABLE ALLOCATION GUIDE
------------------------------------------------------------
| 전략자산 |    45.0%  | {"상기 방어 자산 매수" if is_danger else "QLD 유지"} |
| KOSPI   |    20.0%  | 코어 분산 유지 |
| SPY     |    20.0%  | 코어 포지션 유지 |
| GOLD    |    15.0%  | 안전 자산 유지 |

[4] TECHNICAL SNAPSHOT
------------------------------------------------------------
- Quant Score  : {signal_info["mf_score"]:.1f} / 100
- RSI(14)      : {signal_info["rsi"]:.1f}
- VIX(공포지수): {signal_info["vix"]:.1f}

------------------------------------------------------------
Automated Daily Report | Golden Combo (110/250)
============================================================
"""
        return {
            "title": f"{emoji} {current_status}",
            "body": body,
            "status": current_status,
            "status_changed": (previous_status != current_status)
            if previous_status
            else False,
        }
