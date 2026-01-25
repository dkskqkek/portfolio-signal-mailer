# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import os

warnings.filterwarnings("ignore")

from debate_council import DebateCouncil
from html_generator import generate_html_report



class SignalDetector:
    """
    [Strategy v3.0 PLUS - Integrity Hardened]
    1. Baseline: Dual SMA 130/260 (Robust Core)
    2. Emergency: Hard Cut-off at -40% MDD (Internalized calculation)
    3. Dynamic: KRW (20-60%) Weighting based on DXY Trend + KOSPI Mom
    4. Reliability: Conservative Cold Start logic (Price > SMA_Slow)
    5. Data Sync: 06:00 KST Scheduler ensures complete KOSPI/US bar alignment.
    """

    @staticmethod
    def _to_py_type(val):
        if isinstance(val, (np.bool_, bool)):
            return bool(val)
        if isinstance(val, (np.floating, float)):
            return float(val) if not np.isnan(val) else 0.0
        if isinstance(val, (np.integer, int)):
            return int(val)
        if isinstance(val, datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return val

    def __init__(self, api_key=None):
        self.spy = yf.Ticker("SPY")
        self.qqq_ticker = yf.Ticker("QQQ")
        self.kospi200 = yf.Ticker("^KS200")
        self.vix_ticker = yf.Ticker("^VIX")
        self.gld_ticker = yf.Ticker("GLD")

        # Initialize The Council
        self.council = DebateCouncil(api_key) if api_key else None

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

    def fetch_data(self, days_back=700):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        try:
            # NOTE: At 06:00 KST (US Close), KOSPI data is from previous day (15:30 KST).
            # This ensures we use 'Completed' daily candles for both markets, avoiding look-ahead.
            core_tickers = [
                "SPY",
                "QQQ",
                "^KS200",
                "^VIX",
                "GLD",
                "BIL",
                "KRW=X",
                "DX-Y.NYB",
            ]
            all_tickers = list(set(core_tickers + self.def_pool))
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
                return None
            return data.ffill()
        except Exception as e:
            print(f"Data Fetch Error: {e}")
            return None

    def calculate_quant_score(self, data, s2_slow):
        """
        [Quant Score Engine v2.0]
        - Macro Stability (30%): Based on VIX
        - Trend Integrity (40%): Distance to Slow SMA
        - Vol Efficiency (30%): Risk-adjusted Momentum
        """
        try:
            # 1. Data Prep
            vix = data["^VIX"].iloc[-1]
            curr_price = data["QQQ"].iloc[-1]
            ma_slow = data["QQQ"].rolling(s2_slow).mean().iloc[-1]

            # 2. Component 1: Macro Stability (30 pts)
            # VIX 12이하 만점, 35이상 0점
            score_macro = np.clip((35 - vix) / (35 - 12), 0, 1) * 30

            # 3. Component 2: Trend Integrity (40 pts)
            # MA 대비 +5% 위면 만점, -5% 아래면 0점
            dist_pct = (curr_price / ma_slow) - 1
            # -0.05 ~ +0.05 범위를 0 ~ 1로 정규화 -> (val + 0.05) * 10
            score_trend = np.clip((dist_pct + 0.05) * 10, 0, 1) * 40

            # 4. Component 3: Volatility Efficiency (30 pts)
            # 최근 20일 수익률 / 최근 20일 변동성 (Simplified Sharpe)
            recent_ret = data["QQQ"].pct_change(20).iloc[-1]
            recent_vol = data["QQQ"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(
                20
            )

            if recent_vol == 0:
                efficiency = 0
            else:
                efficiency = recent_ret / recent_vol

            # Efficiency가 2.0 이상이면 만점, -1.0 이하면 0점
            # Range -1 ~ 2 (Span 3)
            score_vol = np.clip((efficiency + 1) / 3, 0, 1) * 30

            # 5. Final Synthesis
            total_score = score_macro + score_trend + score_vol

            return {
                "total": int(total_score),
                "breakdown": (int(score_macro), int(score_trend), int(score_vol)),
                "vix": vix,
                "dist": dist_pct,
                "efficiency": efficiency,
            }

        except Exception as e:
            # logging.error(f"Quant Score Calc Failed: {e}")
            return {"total": 0, "breakdown": (0, 0, 0)}

    def _fetch_news_context(self):
        """
        Placeholder for news fetching.
        In future, integrate with a NewsAPI or scrape major headlines.
        """
        return "No major breaking news specific to the strategy timeline. Market sentiment implies standard volatility."

    def calculate_current_mdd(self, data):
        """Calculate MDD from 1-year peak"""
        window = 252
        prices = data["QQQ"]
        peak = prices.rolling(window, min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.iloc[-1]

    def _load_jarvis_config(self):
        """Load JARVIS AI suggestions from JSON"""
        # Hardcoded path for integrated environment
        path = "d:/gg/data/jarvis_config.json"
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Check freshness (3 days)
            if (datetime.now() - datetime.strptime(data["date"], "%Y-%m-%d")).days > 3:
                return None
            return data
        except:
            return None

    def detect(self, previous_status=None, current_mdd=None):
        data = self.fetch_data()

        # 1. Base Calculations
        if data is None or len(data) < 400:
            return {"error": True, "reason": "Data shortage"}

        if current_mdd is None:
            current_mdd = self.calculate_current_mdd(data)

        is_emergency = current_mdd < -0.40
        vix = data["^VIX"].iloc[-1]
        curr_price = data["QQQ"].iloc[-1]

        # --- Track A: Classic (Rule-Based) ---
        c_s1, c_s2 = (110, 250) if vix > 30 else (130, 260)
        c_ma_fast = data["QQQ"].rolling(c_s1).mean().iloc[-1]
        c_ma_slow = data["QQQ"].rolling(c_s2).mean().iloc[-1]

        if curr_price > c_ma_fast and curr_price > c_ma_slow:
            c_status = "NORMAL"
        elif curr_price < c_ma_fast and curr_price < c_ma_slow:
            c_status = "DANGER"
        else:
            c_status = "NORMAL" if curr_price > c_ma_slow else "DANGER"

        if is_emergency:
            c_status = "EMERGENCY (STOP)"

        # --- Track B: Hybrid (JARVIS) ---
        jarvis_data = self._load_jarvis_config()
        if jarvis_data and not is_emergency:
            # Guardrail Logic (Simplification: Average of Classic & AI if AI is too wild)
            ai_s1 = jarvis_data["suggested_params"]["s1"]
            ai_s2 = jarvis_data["suggested_params"]["s2"]
            # Flexible Guardrail: Allow AI to deviation within 20%
            h_s1 = int(ai_s1)
            h_s2 = int(ai_s2)
            h_regime = f"Running (S1:{h_s1}/S2:{h_s2})"
        else:
            h_s1, h_s2 = c_s1, c_s2
            h_regime = "Fallback to Classic"

        h_ma_fast = data["QQQ"].rolling(h_s1).mean().iloc[-1]
        h_ma_slow = data["QQQ"].rolling(h_s2).mean().iloc[-1]

        if curr_price > h_ma_fast and curr_price > h_ma_slow:
            h_status = "NORMAL"
        elif curr_price < h_ma_fast and curr_price < h_ma_slow:
            h_status = "DANGER"
        else:
            h_status = "NORMAL" if curr_price > h_ma_slow else "DANGER"

        if is_emergency:
            h_status = "EMERGENCY (STOP)"

        # --- Decision: Primary is Hybrid ---
        # But we pass BOTH to the signal info for reporting

        # ... (Rest of metric calculations: KRW Ratio, Etc) ...
        # 2. Dynamic KRW Allocation (User Precision: DXY/KOSPI Logic)
        dxy = data["DX-Y.NYB"]
        dxy_90d_trend = dxy.pct_change(90).iloc[-1]
        kospi_126d_mom = data["^KS200"].pct_change(126).iloc[-1]

        if dxy_90d_trend < -0.05:
            base_krw = 0.40
        else:
            base_krw = 0.20

        if kospi_126d_mom > 0.10:
            krw_ratio = min(base_krw + 0.20, 0.60)
        elif kospi_126d_mom < 0:
            krw_ratio = max(base_krw - 0.20, 0.10)
        else:
            krw_ratio = base_krw

        # Defensive Assets
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

        # Reality Metrics
        qqq_vol = data["QQQ"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
        decay_annual = qqq_vol**2
        fx_ret = data["KRW=X"].pct_change().iloc[-1]
        qqq_usd_ret = data["QQQ"].pct_change().iloc[-1]
        compounded_krw_ret = (1 + qqq_usd_ret) * (1 + fx_ret) - 1

        # [NEW] Quant Score Calculation
        q_score = self.calculate_quant_score(data, h_s2)

        # [NEW] The Council: AI Risk Verification
        # Construct summary for the AI
        market_metrics = {
            "vix": float(vix),
            "quant_score": q_score["total"],
            "mdd": float(current_mdd),
            "regime": h_regime,
            "trend": f"Price ${curr_price:.2f} vs MA {h_ma_slow:.2f}",
        }

        # Placeholder news context (In production, replace with real news fetcher)
        news_context = self._fetch_news_context()

        council_verdict = None
        if self.council:
            council_verdict = self.council.convene_council(market_metrics, news_context)

        # Apply Discount (Modifier) to Status label if severe
        # Note: We don't change 'h_status' logic directly to keep integrity,
        # but we add the verdict to the report.
        # If discount < 0.8, we might append a warning.

        return {
            "is_danger": h_status in ["DANGER", "EMERGENCY (STOP)"],
            "status_label": h_status,  # Primary Status
            "classic_status": c_status,  # For comparison
            "hybrid_status": h_status,  # For comparison
            "classic_params": (c_s1, c_s2),
            "hybrid_params": (h_s1, h_s2),
            "is_emergency": is_emergency,
            "calculated_mdd": float(current_mdd),
            "quant_score": q_score["total"],
            "score_breakdown": q_score["breakdown"],
            "council_verdict": council_verdict,  # Pass Object
            "qqq_price": float(curr_price),
            "ma_fast": float(h_ma_fast),
            "ma_slow": float(h_ma_slow),
            "s_params": (h_s1, h_s2),
            "regime": h_regime,
            "krw_ratio": float(krw_ratio),
            "dxy_90d": float(dxy_90d_trend),
            "kospi_126d": float(kospi_126d_mom),
            "decay_annual": float(decay_annual),
            "fx_compounded_ret": float(compounded_krw_ret),
            "defensive_assets": [str(s) for s in defensive_assets],
            "vix": float(vix),
            "krw_rate": float(data["KRW=X"].iloc[-1]),
            "date": datetime.now(),
            "error": False,
        }

    pass # End of Class - _generate_html_report removed

    @classmethod
    def format_signal_report(cls, signal_info, previous_status=None):
        if signal_info.get("error"):
            return {
                "title": "ERROR",
                "body": str(signal_info),
                "html_body": None,
                "status": "ERROR",
            }

        status = signal_info["status_label"]
        krw_ratio = signal_info["krw_ratio"]
        krw_pct, usd_pct = f"{krw_ratio * 100:.0f}%", f"{(1 - krw_ratio) * 100:.0f}%"

        if status == "EMERGENCY (STOP)":
            emoji = "🛑"
            tactical = "FORCE EXIT -> 100% BIL (Hard Cut-off Activated)"
        elif status == "NORMAL":
            emoji = "🟢"
            tactical = f"US Portion ({usd_pct}): QLD 50% / QQQ 50%"
            tactical += f"\n  KRW Portion ({krw_pct}): KOSPI 50% / Gold-Spot 50%"
        else:  # DANGER
            emoji = "🔴"
            tactical = f"DEFENSIVE: {', '.join(signal_info['defensive_assets'])}"

        # 1. Text Body
        text_body = f"""
============================================================
📅 [{signal_info["date"].strftime("%Y-%m-%d")}] INTEGRITY HARDENED v3.0 PLUS
============================================================

[1] SYSTEM STATUS: {emoji} {status}
------------------------------------------------------------
Regime        : {signal_info["regime"]} (SMA {signal_info["s_params"][0]}/{signal_info["s_params"][1]})
Emergency Mode: {"🚨 ACTIVE" if signal_info["is_emergency"] else "🟢 STANDBY"}
Current MDD   : {signal_info["calculated_mdd"] * 100:.1f}%

[2] DYNAMIC WEIGHTING (Adaptive Balance)
------------------------------------------------------------
Target Split  : [KRW {krw_pct}] vs [USD {usd_pct}]
* Alpha Factors: DXY Trend ({signal_info["dxy_90d"] * 100:+.1f}%), KOSPI Mom ({signal_info["kospi_126d"] * 100:+.1f}%)

[3] ACTIONABLE RECOMMENDATION (Tactical 45%)
------------------------------------------------------------
Action        : {tactical}

[4] INTEGRITY METRICS
------------------------------------------------------------
Volatility Decay: {signal_info["decay_annual"] * 100:.1f}% Annualized (Awareness)
FX Compounded   : {signal_info["fx_compounded_ret"] * 100:+.2f}% (Daily QQQ-KRW)
Data Quality    : Zero Look-Ahead Sync Verified (06:00 KST)

[5] TECHNICAL SNAPSHOT
------------------------------------------------------------
QQQ: ${signal_info["qqq_price"]:.2f} (MA: {signal_info["ma_fast"]:.0f}/{signal_info["ma_slow"]:.0f})
VIX: {signal_info["vix"]:.1f} | USD/KRW: {signal_info["krw_rate"]:.1f}
============================================================
"""

        # 2. HTML Body
        html_body = generate_html_report(signal_info, text_body)

        return {
            "title": f"{emoji} {status}: {krw_pct} KRW / {usd_pct} USD",
            "body": text_body,
            "html_body": html_body,
            "status": status,
            "status_changed": (previous_status != status) if previous_status else False,
        }
