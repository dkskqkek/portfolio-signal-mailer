# -*- coding: utf-8 -*-
"""
Antigravity v3.2 Pro (JARVIS Connected)
------------------------------------------
통합 퀀트 포트폴리오 관리 시스템 (ML Hybrid)
- 신호 탐지 (전략 v3.0 PLUS): 듀얼 모멘텀 + VIX 동적 대응
- 지능형 제어 (JARVIS Connected): ML 기반 파라미터 제안 수용 (Guardrail 적용)
- 데이터 수집: ML 학습용 일일 상태 기록
- 인프라: 로깅 + 타임가드 + 환경변수 보안
- 알림: 프리미엄 HTML 이메일 + 텔레그램 (전면 한글화)
- 안전장치: 1200일 MDD 윈도우 + 자동 데이터 보정

Author: Antigravity AI Partner
Date: 2026-01-25
"""

import sys
import os
import json
import logging
import traceback
import smtplib
from datetime import datetime, timedelta, time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# [1] 사용자 설정 (환경 변수 권장)
# =========================================================
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
    # [JARVIS 설정]
    "use_ml_guide": True,  # ML 제안 사용 여부
    "ml_guardrail": 0.20,  # 파라미터 변동 허용폭 (20%)
}


# =========================================================
# [2] 핵심 엔진: SignalDetector (+ Data Collector + ML Reader)
# =========================================================
class SignalDetector:
    """전략 v3.0 PLUS - 무결성 강화 두뇌 (with JARVIS Link)"""

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
        """[JARVIS] ML 제안 설정 로드 (Sidecar 읽기)"""
        path = os.path.join(CONFIG["base_dir"], "data", "jarvis_config.json")
        if not os.path.exists(path) or not CONFIG["use_ml_guide"]:
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 날짜 확인 (오늘/어제 생성된 제안만 유효)
            cfg_date = datetime.strptime(data["date"], "%Y-%m-%d")
            if (datetime.now() - cfg_date).days > 3:  # 3일 지난 제안은 무시
                logging.warning("⚠️ JARVIS 제안이 너무 오래되어 무시합니다.")
                return None

            logging.info(f"🧠 JARVIS 제안 로드됨: {data['suggested_params']}")
            return data
        except Exception as e:
            logging.error(f"JARVIS 설정 로드 실패: {e}")
            return None

    def _apply_guardrails(self, base_s1, base_s2, proposed_s1, proposed_s2):
        """[Safety] Guardrail: 급격한 파라미터 변경 방지"""
        limit = CONFIG["ml_guardrail"]

        # S1 (Fast MA) 제한
        min_s1, max_s1 = base_s1 * (1 - limit), base_s1 * (1 + limit)
        final_s1 = max(min_s1, min(max_s1, proposed_s1))

        # S2 (Slow MA) 제한
        min_s2, max_s2 = base_s2 * (1 - limit), base_s2 * (1 + limit)
        final_s2 = max(min_s2, min(max_s2, proposed_s2))

        return int(final_s1), int(final_s2)

    def fetch_data(self, days_back=None):
        if days_back is None:
            days_back = CONFIG.get("mdd_window", 1200)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back + 60)

        try:
            core_tickers = [
                "SPY",
                "QQQ",
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

            if raw_data.empty:
                raise ValueError("다운로드된 데이터가 없습니다.")

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
                except Exception as e:
                    logging.debug(f"{ticker} 처리 중 건너뜀: {e}")

            data = pd.DataFrame(data_dict)
            if "^KS11" in data.columns:
                data["^KS200"] = data["^KS11"]
            data = data.ffill().bfill()

            return data.iloc[-days_back:]
        except Exception as e:
            logging.error(f"데이터 수집 치명적 오류: {e}")
            return None

    def calculate_current_mdd(self, data, window=252 * 5):
        if data is None or "QQQ" not in data.columns:
            return 0.0
        effective_window = min(len(data), window)
        recent_data = data["QQQ"].iloc[-effective_window:]
        peak = recent_data.cummax()
        drawdown = (recent_data - peak) / peak
        return float(drawdown.iloc[-1])

    def detect(self, previous_status=None, current_mdd=None):
        data = self.fetch_data()
        if data is None or len(data) < 300:
            return {"error": True, "reason": "데이터 부족 (yfinance 연결 확인 필요)"}

        if current_mdd is None:
            current_mdd = self.calculate_current_mdd(data)

        # 1. 비상 정지 (-40% MDD) - ML도 해제 불가 (Constitution)
        is_emergency = current_mdd < -0.40

        # 2. 시장 국면 판단 및 파라미터 결정 (JARVIS Hybrid)
        vix = data["^VIX"].iloc[-1]

        # 기본 파라미터 (Rule-Based)
        base_s1, base_s2 = (110, 250) if vix > 30 else (130, 260)
        regime = "고변동성 (Fast)" if vix > 30 else "일반 (Robust)"

        # [NEW] JARVIS 제안 적용 (Guardrail)
        if self.jarvis_config and not is_emergency:
            ml_s1 = self.jarvis_config["suggested_params"]["s1"]
            ml_s2 = self.jarvis_config["suggested_params"]["s2"]

            # Guardrail: Limit changes to +/- 20%
            final_s1, final_s2 = self._apply_guardrails(base_s1, base_s2, ml_s1, ml_s2)

            if (final_s1 != base_s1) or (final_s2 != base_s2):
                regime = f"JARVIS Hybrid (S1:{final_s1}/S2:{final_s2})"
                s1, s2 = final_s1, final_s2
            else:
                s1, s2 = base_s1, base_s2  # ML 제안이 Guardrail 밖이거나 유사함
        else:
            s1, s2 = base_s1, base_s2

        # 3. 자산 비중 로직
        dxy_90d = data["DX-Y.NYB"].pct_change(90).iloc[-1]
        kospi_126d = data["^KS11"].pct_change(126).iloc[-1]

        base_krw = 0.40 if dxy_90d < -0.05 else 0.20
        if kospi_126d > 0.10:
            krw_ratio = min(base_krw + 0.20, 0.60)
        elif kospi_126d < 0:
            krw_ratio = max(base_krw - 0.20, 0.10)
        else:
            krw_ratio = base_krw

        # [NEW] JARVIS Crash Warning (Crash Prob > 70% -> Increase KRW cash)
        if self.jarvis_config and self.jarvis_config.get("crash_probability", 0) > 0.70:
            krw_ratio = max(krw_ratio, 0.50)  # 강제로 안전자산 50% 이상 확보
            regime += " + ⚠️ CRASH WARNING"

        # 4. SMA 모멘텀 진단
        curr_price = data["QQQ"].iloc[-1]
        ma_fast = data["QQQ"].rolling(s1).mean().iloc[-1]
        ma_slow = data["QQQ"].rolling(s2).mean().iloc[-1]

        if curr_price > ma_fast and curr_price > ma_slow:
            status = "NORMAL"
        elif curr_price < ma_fast and curr_price < ma_slow:
            status = "DANGER"
        else:
            if previous_status is None:
                status = "NORMAL" if curr_price > ma_slow else "DANGER"
            else:
                status = (
                    previous_status
                    if previous_status in ["NORMAL", "DANGER"]
                    else "NORMAL"
                )

        if is_emergency:
            status = "EMERGENCY (STOP)"

        # 5. 방어 자산 선정
        mom_returns = (
            data[self.def_pool]
            .pct_change(168)
            .iloc[-1]
            .dropna()
            .sort_values(ascending=False)
        )
        valid_assets = mom_returns[mom_returns > 0].head(3)
        defensive_assets = (
            valid_assets.index.tolist() if not valid_assets.empty else ["BIL"]
        )

        tnx = data["^TNX"].iloc[-1] if "^TNX" in data.columns else 0.0

        return {
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

    def format_report(self, info, previous=None):
        status = info["status_label"]
        emoji = "🟢" if status == "NORMAL" else "🔴" if status == "DANGER" else "🛑"
        krw_pct, usd_pct = (
            f"{info['krw_ratio'] * 100:.0f}%",
            f"{(1 - info['krw_ratio']) * 100:.0f}%",
        )

        tactical = (
            f"미국 자산 ({usd_pct}): QLD/QQQ 분할\n  한국 자산 ({krw_pct}): KOSPI/골드현물"
            if status == "NORMAL"
            else f"방어 자산: {', '.join(info['defensive_assets'])}"
            if status == "DANGER"
            else "전량 매도 -> 100% 현금 보유"
        )

        return f"""
============================================================
📅 [{info["date"].strftime("%Y-%m-%d %H:%M")}] Antigravity Pro v3.2
============================================================
[1] 시스템 판단: {emoji} {status}
------------------------------------------------------------
시장 국면   : {info["regime"]} (이평선 {info["s_params"][0]}/{info["s_params"][1]})
비상 모드   : {"🚨 작동 중" if info["is_emergency"] else "🟢 대기 중"}
현재 MDD    : {info["calculated_mdd"] * 100:.1f}% (1200일 기준)

[2] 동적 자산 배분 (Adaptive Balance)
------------------------------------------------------------
목표 비중   : [원화 {krw_pct}] vs [달러 {usd_pct}]
알파 팩터   : 달러 추세 ({info["dxy_90d"] * 100:+.1f}%), 코스피 모멘텀 ({info["kospi_126d"] * 100:+.1f}%)

[3] 행동 지침 (Action)
------------------------------------------------------------
{tactical}

[4] 기술적 지표 (Snapshot)
------------------------------------------------------------
QQQ 가격 : ${info["qqq_price"]:.2f} (MA: {info["ma_fast"]:.0f}/{info["ma_slow"]:.0f})
VIX 지수 : {info["vix"]:.1f}
국채 금리: {info["tnx"]:.2f}% (10년물)
============================================================
"""


# =========================================================
# [3] 서비스 모듈: MailerService & DataCollector
# =========================================================
class MailerService:
    def __init__(self, config):
        self.config = config
        self.log_dir = os.path.join(config["base_dir"], "logs")
        self.data_dir = os.path.join(config["base_dir"], "data")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        log_file = os.path.join(self.log_dir, "system.log")
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            h = logging.FileHandler(log_file, encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
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
        if "YOUR_" in e_cfg["sender_email"] or "YOUR_" in e_cfg["sender_password"]:
            logging.error("보안 확인: 환경 변수가 설정되지 않았습니다.")
            return False

        if self.config["debug_mode"]:
            logging.info("[DEBUG] 메일 발송 시뮬레이션 완료.")
            return True

        try:
            msg = MIMEMultipart()
            msg["From"] = f"Antigravity Pro <{e_cfg['sender_email']}>"
            msg["To"] = e_cfg["recipient_email"]
            msg["Subject"] = subject
            html = f"<html><body><div style='font-family:monospace; background:#121212; color:#00ff41; padding:20px; border-radius:10px;'><pre>{body_text}</pre></div></body></html>"
            msg.attach(MIMEText(html, "html", "utf-8"))

            with smtplib.SMTP(e_cfg["smtp_server"], e_cfg["smtp_port"]) as s:
                s.starttls()
                s.login(e_cfg["sender_email"], e_cfg["sender_password"])
                s.send_message(msg)
            logging.info(f"✓ 이메일 발송 성공: {e_cfg['recipient_email']}")
            return True
        except Exception as e:
            logging.error(f"이메일 발송 실패: {e}")
            return False

    def save_history(self, info):
        path = os.path.join(self.data_dir, "history.json")
        history = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
        clean_info = {k: self._to_py(v) for k, v in info.items()}
        history[datetime.now().strftime("%Y-%m-%d %H:%M:%S")] = clean_info
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def save_features(self, info):
        path = os.path.join(self.data_dir, "features.csv")
        ma_fast_dist = (info["qqq_price"] / info["ma_fast"]) - 1
        ma_slow_dist = (info["qqq_price"] / info["ma_slow"]) - 1
        record = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "status": info["status_label"],
            "qqq_price": info["qqq_price"],
            "vix": info["vix"],
            "tnx": info["tnx"],
            "ma_fast_dist": ma_fast_dist,
            "ma_slow_dist": ma_slow_dist,
            "current_mdd": info["calculated_mdd"],
            "krw_ratio": info["krw_ratio"],
            "dxy_90d": info["dxy_90d"],
            "s1": info["s_params"][0],
            "s2": info["s_params"][1],
        }
        df = pd.DataFrame([record])
        if not os.path.exists(path):
            df.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
        logging.info("✓ JARVIS 데이터 수집 완료")

    def get_last_status(self):
        path = os.path.join(self.data_dir, "history.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if not history:
                return None
            return history[sorted(history.keys())[-1]].get("status_label")
        except:
            return None


def send_tg(msg):
    t_cfg = CONFIG["telegram"]
    if not t_cfg["use"]:
        return
    try:
        url = f"https://api.telegram.org/bot{t_cfg['bot_token']}/sendMessage"
        requests.post(
            url,
            json={"chat_id": t_cfg["chat_id"], "text": msg, "parse_mode": "Markdown"},
            timeout=5,
        )
    except:
        pass


# =========================================================
# [4] 메인 오케스트레이션
# =========================================================
def main():
    print(
        f"--- 🚀 Antigravity Pro v3.2 시작 [{datetime.now().strftime('%Y-%m-%d %H:%M')}] ---"
    )
    service = MailerService(CONFIG)
    detector = SignalDetector()

    if "YOUR_" in CONFIG["email"]["sender_email"]:
        logging.warning(
            "⚠️ 보안 경고: SMTP 계정 정보가 환경 변수로 설정되지 않았습니다."
        )

    now = datetime.now().time()
    if time(9, 0) <= now <= time(15, 30):
        logging.warning("⚠️ 시장 개장 중 (데이터 노이즈 주의)")

    try:
        prev = service.get_last_status()
        info = detector.detect(previous_status=prev)
        if info.get("error"):
            raise Exception(info["reason"])

        service.save_history(info)
        service.save_features(info)

        report_text = detector.format_report(info, previous=prev)
        subject = f"[Antigravity] {info['status_label']}: 원화 {info['krw_ratio'] * 100:.0f}% / 달러 {(1 - info['krw_ratio']) * 100:.0f}%"
        service.send_email(subject, report_text)

        tg_emoji = (
            "🛑"
            if info["is_emergency"]
            else "🟢"
            if info["status_label"] == "NORMAL"
            else "🔴"
        )
        send_tg(
            f"{tg_emoji} *오늘의 신호: {info['status_label']}*\n비중: 원화 {info['krw_ratio'] * 100:.0f}% / 달러 {(1 - info['krw_ratio']) * 100:.0f}%\nMDD: {info['calculated_mdd'] * 100:.2f}%"
        )

        print(f"✅ 작업 완료. 상태: {info['status_label']}")

    except Exception as e:
        err = traceback.format_exc()
        logging.critical(f"치명적 오류: {e}\n{err}")
        send_tg(f"🔥 *시스템 크래시 발생*\n{str(e)}")


if __name__ == "__main__":
    main()
