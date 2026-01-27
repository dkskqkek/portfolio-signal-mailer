# -*- coding: utf-8 -*-
"""
Antigravity v3.2 Hybrid (Track B: JARVIS Enhanced)
------------------------------------------
통합 퀀트 포트폴리오 관리 시스템 (Hybrid Track)
- 신호 탐지: JARVIS ML 제안 + Guardrail 적용
- 가상 매매: 연금저축(5천) / 외화직투(5천) 시뮬레이션
- 역할: ML 기반 알파 탐색 및 최적화 전략 실행

Author: Antigravity AI Partner
Date: 2026-01-25
"""

import sys
import os
import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings

# [Phase 14 Integration] Virtual Broker
from virtual_broker import VirtualBroker, VirtualPortfolio
from ticker_mapper import TickerMapper

warnings.filterwarnings("ignore")

CONFIG = {
    "email": {
        "sender_email": os.getenv("ANTIGRAVITY_EMAIL", "YOUR_GMAIL@gmail.com"),
        "sender_password": os.getenv("ANTIGRAVITY_EMAIL_PW", "YOUR_APP_PASSWORD"),
        "recipient_email": os.getenv("ANTIGRAVITY_RECIPIENT", "YOUR_GMAIL@gmail.com"),
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
    },
    "telegram": {
        "use": os.getenv("ANTIGRAVITY_TG_USE", "False").lower() == "true",
        "bot_token": os.getenv("ANTIGRAVITY_TG_TOKEN", "YOUR_BOT_TOKEN"),
        "chat_id": os.getenv("ANTIGRAVITY_TG_CHAT_ID", "YOUR_CHAT_ID"),
    },
    "base_dir": os.path.dirname(os.path.abspath(__file__)),
    "debug_mode": os.getenv("ANTIGRAVITY_DEBUG", "False").lower() == "true",
    "mdd_window": 1200,
    "use_ml_guide": True,
    "ml_guardrail": 0.20,
}


class SignalDetector:
    def __init__(self):
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
            "VIXM",
        ]
        self.jarvis_config = self._load_jarvis_config()

    def _load_jarvis_config(self):
        path = os.path.join(CONFIG["base_dir"], "data", "jarvis_config.json")
        if not os.path.exists(path) or not CONFIG["use_ml_guide"]:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if (datetime.now() - datetime.strptime(data["date"], "%Y-%m-%d")).days > 3:
                return None
            logging.info(f"🧠 JARVIS 제안 로드됨: {data['suggested_params']}")
            return data
        except:
            return None

    def _apply_guardrails(self, base_s1, base_s2, proposed_s1, proposed_s2):
        limit = CONFIG["ml_guardrail"]
        min_s1, max_s1 = base_s1 * (1 - limit), base_s1 * (1 + limit)
        min_s2, max_s2 = base_s2 * (1 - limit), base_s2 * (1 + limit)
        return int(max(min_s1, min(max_s1, proposed_s1))), int(
            max(min_s2, min(max_s2, proposed_s2))
        )

    def fetch_data(self, days_back=None):
        if days_back is None:
            days_back = CONFIG.get("mdd_window", 1200)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back + 60)
        try:
            core_tickers = [
                "SPY",
                "QQQ",
                "QLD",
                "^KS11",
                "^VIX",
                "GLD",
                "BIL",
                "KRW=X",
                "DX-Y.NYB",
                "^TNX",
            ]
            all_tickers = list(set(core_tickers + self.def_pool))
            logging.info(f"데이터 다운로드 중 ({len(all_tickers)}개 종목)...")
            raw_data = yf.download(
                all_tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                repair=True,
                group_by="ticker",
            )

            data_dict = {}
            for ticker in all_tickers:
                try:
                    if ticker in raw_data.columns.get_level_values(0):
                        t_data = raw_data[ticker]
                        col = "Close"
                        if col in t_data.columns:
                            data_dict[ticker] = t_data[col]
                        elif not t_data.empty:
                            data_dict[ticker] = t_data.iloc[:, 0]
                except:
                    pass
            data = pd.DataFrame(data_dict)
            if "^KS11" in data.columns:
                data["^KS200"] = data["^KS11"]
            data = data.ffill().bfill()

            last_prices = data.iloc[-1].to_dict()
            return data.iloc[-days_back:], last_prices
        except Exception as e:
            logging.error(f"데이터 수집 오류: {e}")
            return None, None

    def calculate_current_mdd(self, data, window=252 * 5):
        if data is None or "QQQ" not in data.columns:
            return 0.0
        effective_window = min(len(data), window)
        recent_data = data["QQQ"].iloc[-effective_window:]
        peak = recent_data.cummax()
        drawdown = (recent_data - peak) / peak
        return float(drawdown.iloc[-1])

    def detect(self):
        data, last_prices = self.fetch_data()
        if data is None or len(data) < 300:
            return {"error": True, "reason": "데이터 부족"}

        current_mdd = self.calculate_current_mdd(data)
        is_emergency = current_mdd < -0.40

        vix = data["^VIX"].iloc[-1]
        base_s1, base_s2 = (110, 250) if vix > 30 else (130, 260)
        regime = "고변동성 (Fast)" if vix > 30 else "일반 (Robust)"

        if self.jarvis_config and not is_emergency:
            ml_s1 = self.jarvis_config["suggested_params"]["s1"]
            ml_s2 = self.jarvis_config["suggested_params"]["s2"]
            final_s1, final_s2 = self._apply_guardrails(base_s1, base_s2, ml_s1, ml_s2)
            if (final_s1 != base_s1) or (final_s2 != base_s2):
                regime = f"JARVIS Hybrid (S1:{final_s1}/S2:{final_s2})"
                s1, s2 = final_s1, final_s2
            else:
                s1, s2 = base_s1, base_s2
        else:
            s1, s2 = base_s1, base_s2

        dxy_90d = data["DX-Y.NYB"].pct_change(90).iloc[-1]
        kospi_126d = data["^KS11"].pct_change(126).iloc[-1]
        base_krw = 0.40 if dxy_90d < -0.05 else 0.20
        if kospi_126d > 0.10:
            krw_ratio = min(base_krw + 0.20, 0.60)
        elif kospi_126d < 0:
            krw_ratio = max(base_krw - 0.20, 0.10)
        else:
            krw_ratio = base_krw

        if self.jarvis_config and self.jarvis_config.get("crash_probability", 0) > 0.70:
            krw_ratio = max(krw_ratio, 0.50)
            regime += " + ⚠️ CRASH WARNING"

        curr_price = data["QQQ"].iloc[-1]
        ma_fast = data["QQQ"].rolling(s1).mean().iloc[-1]
        ma_slow = data["QQQ"].rolling(s2).mean().iloc[-1]
        if curr_price > ma_fast and curr_price > ma_slow:
            status = "NORMAL"
        elif curr_price < ma_fast and curr_price < ma_slow:
            status = "DANGER"
        else:
            status = "NORMAL" if curr_price > ma_slow else "DANGER"
        if is_emergency:
            status = "EMERGENCY (STOP)"

        mom_returns = (
            data[self.def_pool]
            .pct_change(168)
            .iloc[-1]
            .dropna()
            .sort_values(ascending=False)
        )
        defensive_assets = mom_returns[mom_returns > 0].head(3).index.tolist() or [
            "BIL"
        ]
        tnx = data["^TNX"].iloc[-1] if "^TNX" in data.columns else 0.0

        info = {
            "status_label": status,
            "is_emergency": is_emergency,
            "calculated_mdd": current_mdd,
            "qqq_price": float(curr_price),
            "ma_fast": float(ma_fast),
            "ma_slow": float(ma_slow),
            "s_params": (s1, s2),
            "regime": regime,
            "krw_ratio": float(krw_ratio),
            "dxy_90d": float(dxy_90d),
            "kospi_126d": float(kospi_126d),
            "defensive_assets": defensive_assets,
            "date": datetime.now(),
            "vix": float(vix),
            "tnx": float(tnx),
            "error": False,
        }
        return info, last_prices

    def format_report(self, info, paper_text):
        status = info["status_label"]
        emoji = "🟢" if status == "NORMAL" else "🔴" if status == "DANGER" else "🛑"
        krw_pct, usd_pct = (
            f"{info['krw_ratio'] * 100:.0f}%",
            f"{(1 - info['krw_ratio']) * 100:.0f}%",
        )
        tactical = (
            f"미국 자산 ({usd_pct}): QLD/QQQ\n  한국 자산 ({krw_pct}): KOSPI/금"
            if status == "NORMAL"
            else f"방어 자산: {', '.join(info['defensive_assets'])}"
            if status == "DANGER"
            else "전량 매도 (CASH)"
        )

        return f"""
============================================================
📅 [{info["date"].strftime("%Y-%m-%d %H:%M")}] Antigravity Hybrid (Track B)
============================================================
[1] JARVIS 판단: {emoji} {status}
------------------------------------------------------------
시장 국면   : {info["regime"]} (이평선 {info["s_params"][0]}/{info["s_params"][1]})
현재 MDD    : {info["calculated_mdd"] * 100:.1f}%

[2] 동적 자산 배분
------------------------------------------------------------
목표 비중   : [원화 {krw_pct}] vs [달러 {usd_pct}]
팩터 현황   : 달러 {info["dxy_90d"] * 100:+.1f}%, 코스피 {info["kospi_126d"] * 100:+.1f}%

[3] 행동 지침
------------------------------------------------------------
{tactical}

[4] 가상 계좌 (Paper Trading)
============================================================
{paper_text}============================================================
"""


class MailerService:
    def __init__(self, config):
        self.config = config
        self.log_dir = os.path.join(config["base_dir"], "logs")
        self.data_dir = os.path.join(config["base_dir"], "data")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        log_file = os.path.join(self.log_dir, "system_hybrid.log")
        logger = logging.getLogger()
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        logger.setLevel(logging.INFO)
        h = logging.FileHandler(log_file, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s [Hybrid] %(message)s"))
        logger.addHandler(h)
        logger.addHandler(logging.StreamHandler(sys.stdout))

    def _to_py(self, val):
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        if isinstance(val, (np.floating, float)):
            return float(val) if not np.isnan(val) else 0.0
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return val

    def send_email(self, subject, body_text):
        e_cfg = self.config["email"]
        if "YOUR_" in e_cfg["sender_email"]:
            return False
        try:
            msg = MIMEMultipart()
            msg["From"] = f"Antigravity Hybrid <{e_cfg['sender_email']}>"
            msg["To"] = e_cfg["recipient_email"]
            msg["Subject"] = subject
            msg.attach(
                MIMEText(
                    f"<html><body><pre>{body_text}</pre></body></html>", "html", "utf-8"
                )
            )
            with smtplib.SMTP(e_cfg["smtp_server"], e_cfg["smtp_port"]) as s:
                s.starttls()
                s.login(e_cfg["sender_email"], e_cfg["sender_password"])
                s.send_message(msg)
            logging.info("이메일 발송 성공")
            return True
        except:
            return False

    def save_history(self, info):
        path = os.path.join(self.data_dir, "history_hybrid.json")
        history = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
        history[datetime.now().strftime("%Y-%m-%d %H:%M:%S")] = {
            k: self._to_py(v) for k, v in info.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


def send_tg(msg):
    t_cfg = CONFIG["telegram"]
    if not t_cfg["use"]:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{t_cfg['bot_token']}/sendMessage",
            json={"chat_id": t_cfg["chat_id"], "text": msg, "parse_mode": "Markdown"},
            timeout=5,
        )
    except:
        pass


def main():
    print(f"--- 🧠 Antigravity Hybrid Start [{datetime.now().strftime('%H:%M')}] ---")
    service = MailerService(CONFIG)
    detector = SignalDetector()
    broker = VirtualBroker(
        CONFIG["base_dir"]
    )  # VirtualBroker init does not take path arg in this version?
    # Checking virtual_broker.py init used in Classic: broker = VirtualBroker(mapper=mapper, ...), wait, standard VirtualBroker init is __init__(mapper, comm)
    # Correct usage:
    mapper = TickerMapper()
    broker = VirtualBroker(mapper=mapper, commission=0.0015)

    try:
        info, prices = detector.detect()
        if info.get("error"):
            raise Exception(info["reason"])

        service.save_history(info)

        # [Paper Trading]
        portfolios = [
            VirtualPortfolio("Hybrid_Pension", 50000000, "KRW"),
            VirtualPortfolio("Hybrid_USD", 35700, "USD"),
        ]

        usd_krw = prices.get("KRW=X", 1400.0)

        for pf in portfolios:
            target_krw_ratio = info["krw_ratio"]
            total_eq = broker.sync_portfolio(pf, prices, usd_krw)

            target_attack_amt = total_eq * (1 - target_krw_ratio)

            if info["status_label"] == "NORMAL":
                if "Hybrid_USD" in pf.account_id:
                    attack_ticker = "QLD"
                else:
                    # 연금 제약 -> 자동으로 TickerMapper가 QQQ(379800)로 변환
                    attack_ticker = "QLD"
            elif info["status_label"] == "DANGER":
                attack_ticker = "BIL"
            else:  # STOP
                attack_ticker = "BIL"

            price_attack = prices.get(attack_ticker, 0)
            if price_attack == 0 and attack_ticker == "QLD":
                price_attack = prices.get("QQQ", 0)

            broker.execute_order(
                pf, attack_ticker, target_attack_amt, price_attack, info["date"]
            )
            broker.sync_portfolio(pf, prices, usd_krw)

        paper_text = "ACCOUNT          | EQUITY        | YIELD   | NOTE\n"
        paper_text += "-----------------|---------------|---------|-----\n"
        for pf in portfolios:
            eq = pf.state["total_equity"]
            init = 35700 * usd_krw if pf.state["currency"] == "USD" else 50000000
            yield_pct = ((eq / init) - 1) * 100
            unit = "$" if pf.state["currency"] == "USD" else "₩"
            disp_eq = f"₩{eq:,.0f}"  # KRW 환산 가치 통일

            paper_text += (
                f"{pf.account_id:<16} | {disp_eq:<13} | {yield_pct:+.2f}%  |\n"
            )

        report = detector.format_report(info, paper_text)
        service.send_email(f"[AG-Hybrid] {info['status_label']}", report)

        print(f"✅ Hybrid 완료: {info['status_label']}")

    except Exception as e:
        import traceback

        logging.critical(f"오류: {e}\n{traceback.format_exc()}")
        send_tg(f"🔥 [Hybrid] 오류: {e}")


if __name__ == "__main__":
    main()
