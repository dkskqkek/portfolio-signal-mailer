# -*- coding: utf-8 -*-
"""
Virtual Broker (Paper Trading Engine)
-------------------------------------
Antigravity 시스템을 위한 가상 증권사 모듈
- 포트폴리오 관리: 잔고, 보유 종목, 평가금액 계산 (State Machine)
- 주문 집행: 매수/매도 시뮬레이션 (슬리피지 반영)
- 계좌 유형별 제약: 연금저축(Pension) vs 일반(Direct)

Author: Antigravity AI Partner
Date: 2026-01-25
"""

import os
import json
import logging
from datetime import datetime


class VirtualPortfolio:
    """계좌 상태 관리 (잔고, 보유종목) - State Machine"""

    def __init__(self, account_id, initial_balance=50000000, currency="KRW"):
        self.account_id = account_id
        self.file_path = f"./data/portfolios/{account_id}_state.json"

        # 기본 상태 초기화
        self.state = {
            "balance": initial_balance,  # 예수금
            "holdings": {},  # 보유 종목 {ticker: qty}
            "avg_price": {},  # 평단가 {ticker: price}
            "currency": currency,
            "total_equity": initial_balance,
            "last_update": None,
        }
        self.load_state()

    def load_state(self):
        # 상대 경로 문제 해결을 위해 절대 경로 변환
        abs_path = os.path.abspath(self.file_path)
        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                self.state = json.load(f)

    def save_state(self):
        abs_path = os.path.abspath(self.file_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        self.state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=4, ensure_ascii=False)

    def log_trade(self, trade_record):
        """거래 로그 별도 저장 (Append Mode)"""
        abs_log_path = os.path.abspath(
            f"./data/portfolios/{self.account_id}_trades.json"
        )
        trades = []
        if os.path.exists(abs_log_path):
            with open(abs_log_path, "r", encoding="utf-8") as f:
                try:
                    trades = json.load(f)
                except:
                    pass

        trades.append(trade_record)
        with open(abs_log_path, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=4, ensure_ascii=False)


class VirtualBroker:
    """주문 집행 및 제약 조건 처리 (Execution Engine)"""

    def __init__(self, mapper=None, commission=0.0015):  # 수수료+슬리피지 0.15% 가정
        self.mapper = mapper
        self.commission = commission
        self.logger = logging.getLogger("VirtualBroker")

    def sync_portfolio(self, portfolio, current_prices, usd_krw=1400.0):
        """현재가 반영하여 총 자산(Equity) 업데이트"""
        holdings_value = 0
        for ticker, qty in portfolio.state["holdings"].items():
            price = current_prices.get(ticker, 0)
            if price > 0:
                # 미국 티커인 경우 환율 적용 (간단한 구분: 숫자 없으면 미국)
                is_us = not any(
                    char.isdigit() for char in ticker
                ) and not ticker.startswith("^")
                if is_us and portfolio.state["currency"] == "KRW":
                    # KRW 계좌인데 미국 주식을 보유하고 있다는 건, 매핑 전 시뮬레이션 상황
                    # 혹은 매핑된 티커의 가격을 미국 주식 가격 * 환율로 추정하는 상황
                    holdings_value += price * qty * usd_krw
                else:
                    holdings_value += price * qty

        # 현금 합산
        # USD 계좌의 경우 balance는 달러이므로 환산 필요 (리포팅용 Total Equity는 KRW 기준 통일 권장되나
        # 여기서는 계좌 통화 기준으로 저장하고, 리포팅 시 변환)
        portfolio.state["total_equity"] = portfolio.state["balance"] + holdings_value
        portfolio.save_state()
        return portfolio.state["total_equity"]

    def execute_order(self, portfolio, ticker, target_amount, price, date):
        """
        주문 집행 (매수/매도)
        - ticker: 전략상 티커 (예: QLD)
        - target_amount: 목표 매수 금액 (계좌 통화 기준)
        """
        # 0. 가격 유효성 체크
        if price <= 0:
            return

        # 1. Ticker Mapping (Pension 제약 적용)
        real_ticker = ticker
        note = ""
        if "Pension" in portfolio.account_id and self.mapper:
            real_ticker, note = self.mapper.map_ticker(ticker)
            if ticker != real_ticker:
                self.logger.warning(
                    f"⚠️ [Constraint] {ticker} -> {real_ticker} ({note})"
                )

        # 2. 수량 계산
        # 현재 보유량 확인
        current_qty = portfolio.state["holdings"].get(real_ticker, 0)
        current_val = current_qty * price

        diff_amount = target_amount - current_val

        # 3. 매매 결정 (최소 주문 금액 필터링: 주가보다 작으면 거래 불가)
        if abs(diff_amount) < price:
            return

        qty_to_trade = int(diff_amount / price)
        if qty_to_trade == 0:
            return

        trade_amount = qty_to_trade * price
        fee = abs(trade_amount) * self.commission

        # 4. 자금 확인 (매수 시)
        if qty_to_trade > 0:
            cost = trade_amount + fee
            if portfolio.state["balance"] < cost:
                self.logger.warning(
                    f"❌ 잔고 부족: 필요 {cost} > 보유 {portfolio.state['balance']}"
                )
                # 가능한 만큼만 매수 (Optional)
                qty_to_trade = int((portfolio.state["balance"] - fee) / price)
                if qty_to_trade <= 0:
                    return
                trade_amount = qty_to_trade * price
                fee = abs(trade_amount) * self.commission
                cost = trade_amount + fee

        # 5. 장부 반영
        if qty_to_trade > 0:  # Buy
            portfolio.state["balance"] -= trade_amount + fee
            # 평단가 갱신 (가중평균)
            old_qty = current_qty
            old_avg = portfolio.state["avg_price"].get(real_ticker, 0)
            new_avg = ((old_qty * old_avg) + (qty_to_trade * price)) / (
                old_qty + qty_to_trade
            )
            portfolio.state["avg_price"][real_ticker] = new_avg
        else:  # Sell
            portfolio.state["balance"] += abs(trade_amount) - fee
            # 평단가 유지 (FIFO/이동평균 등에서 이동평균은 매도 시 불변)

        portfolio.state["holdings"][real_ticker] = current_qty + qty_to_trade

        # 보유량 0이면 삭제
        if portfolio.state["holdings"][real_ticker] <= 0:
            if real_ticker in portfolio.state["holdings"]:
                del portfolio.state["holdings"][real_ticker]
            if real_ticker in portfolio.state["avg_price"]:
                del portfolio.state["avg_price"][real_ticker]

        # 6. 로그 기록 및 저장
        record = {
            "date": date.strftime("%Y-%m-%d"),
            "ticker_strategy": ticker,
            "ticker_real": real_ticker,
            "action": "BUY" if qty_to_trade > 0 else "SELL",
            "qty": qty_to_trade,
            "price": price,
            "amount": trade_amount,
            "fee": fee,
            "note": note,
        }
        portfolio.log_trade(record)
        portfolio.save_state()

        self.logger.info(
            f"✅ Order Executed: {record['action']} {real_ticker} {qty_to_trade}sh @ {price}"
        )
