# -*- coding: utf-8 -*-
"""
Ticker Mapper (The Reality Interface)
-------------------------------------
전략 신호와 실제 계좌(특히 연금저축) 간의 연결 고리.
[Strategy Ticker] -> [Real Ticker (Pension Compatible)]
레버리지/인버스 불가 -> 1배수/현금성 자산으로 강제 변환

Author: Antigravity AI Partner
Date: 2026-01-25
"""


class TickerMapper:
    """
    [Strategy Ticker] -> [Real Ticker (Pension Compatible)]
    레버리지/인버스 불가 -> 1배수/현금성 자산으로 강제 변환
    """

    # 한국 ETF 매핑 테이블 (2026년 기준)
    MAPPING = {
        # [공격 자산]
        "QQQ": ("379800", "TIGER 미국테크TOP10"),  # or 133690 (나스닥100)
        "QLD": ("379800", "Lev 제한->1배수 대체"),  # ⚠️ 핵심: 2배 -> 1배 강제 다운
        "TQQQ": ("379800", "Lev 제한->1배수 대체"),
        "SPY": ("360750", "TIGER 미국S&P500"),
        # [방어 자산]
        "BIL": ("459580", "KODEX 미국달러SOFR"),  # 달러 현금 -> SOFR ETF
        "SHV": ("459580", "KODEX 미국달러SOFR"),
        "GLD": ("411060", "ACE KRX금현물"),
        "IEF": ("305080", "TIGER 미국채10년선물"),
        "TLT": ("305080", "TIGER 미국채10년선물"),  # 30년물이 없으면 10년물로 대체
        # [인버스/헤지] - 연금계좌는 인버스 불가
        "PSQ": ("459580", "Inv 제한->현금 대체"),  # 인버스 -> 현금(SOFR)
        "SH": ("459580", "Inv 제한->현금 대체"),
    }

    def map_ticker(self, ticker):
        if ticker in self.MAPPING:
            real_ticker, note = self.MAPPING[ticker]
            return real_ticker, note
        return ticker, ""  # 매핑 없으면 그대로 (일반 계좌용)
