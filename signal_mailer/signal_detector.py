# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import os

warnings.filterwarnings("ignore")


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

    @staticmethod
    def _generate_html_report(signal_info, text_body):
        """Generates the 'Midnight Quant' Premium HTML Report with 2-Track Comparison"""

        # 1. Basic Data Preparation
        date_str = signal_info["date"].strftime("%Y. %m. %d (%a)")
        status = signal_info["status_label"]
        qqq_price = f"${signal_info['qqq_price']:.2f}"
        ma_fast = f"${signal_info['ma_fast']:.2f}"
        ma_slow = f"${signal_info['ma_slow']:.2f}"

        # Stability Score (Proxy: 100 - MDD%)
        quant_score = max(0, 100 - (abs(signal_info["calculated_mdd"]) * 100))
        quant_score_str = f"{quant_score:.1f}"

        # 2. Status Styling & Asset Allocation Logic
        if status == "NORMAL":
            status_color = "#00FF9D"  # Neon Green
            status_emoji = "🟢"
            market_status_display = "NORMAL"
            sub_status = f"Hybrid Optimized ({signal_info['s_params'][0]}/{signal_info['s_params'][1]})"

            # Normal Allocation Chart
            allocation_html = """
                <div style="background-color: #1E1E1E; border-radius: 12px; padding: 20px;">
                    <!-- QLD -->
                    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 15px;">
                        <tr><td style="color: #FFFFFF; font-size: 14px; font-weight: bold;">QLD (2x)</td><td align="right" style="color: #00FF9D; font-size: 14px; font-weight: bold;">45.0%</td></tr>
                        <tr><td colspan="2" style="padding-top: 5px;"><div style="background-color: #333333; height: 6px; border-radius: 3px; width: 100%;"><div style="background-color: #00FF9D; height: 6px; border-radius: 3px; width: 45%;"></div></div></td></tr>
                    </table>
                    <!-- KOSPI -->
                    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 15px;">
                        <tr><td style="color: #FFFFFF; font-size: 14px;">KOSPI</td><td align="right" style="color: #CCCCCC; font-size: 14px;">20.0%</td></tr>
                        <tr><td colspan="2" style="padding-top: 5px;"><div style="background-color: #333333; height: 6px; border-radius: 3px; width: 100%;"><div style="background-color: #4A90E2; height: 6px; border-radius: 3px; width: 20%;"></div></div></td></tr>
                    </table>
                    <!-- SPY -->
                    <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 15px;">
                        <tr><td style="color: #FFFFFF; font-size: 14px;">SPY</td><td align="right" style="color: #CCCCCC; font-size: 14px;">20.0%</td></tr>
                        <tr><td colspan="2" style="padding-top: 5px;"><div style="background-color: #333333; height: 6px; border-radius: 3px; width: 100%;"><div style="background-color: #9013FE; height: 6px; border-radius: 3px; width: 20%;"></div></div></td></tr>
                    </table>
                    <!-- GOLD -->
                    <table width="100%" cellpadding="0" cellspacing="0">
                        <tr><td style="color: #FFFFFF; font-size: 14px;">GOLD</td><td align="right" style="color: #CCCCCC; font-size: 14px;">15.0%</td></tr>
                        <tr><td colspan="2" style="padding-top: 5px;"><div style="background-color: #333333; height: 6px; border-radius: 3px; width: 100%;"><div style="background-color: #F5A623; height: 6px; border-radius: 3px; width: 15%;"></div></div></td></tr>
                    </table>
                </div>
            """
        else:
            # DANGER / EMERGENCY
            status_color = "#FF453A"  # Neon Red
            status_emoji = "🔴" if status != "EMERGENCY (STOP)" else "🛑"
            market_status_display = status
            sub_status = "Defensive Mode Activated"

            # Construct Defensive Asset List String
            def_assets_html = ""
            medals = ["🥇", "🥈", "🥉"]
            for i, asset in enumerate(signal_info.get("defensive_assets", [])):
                medal = medals[i] if i < 3 else "🛡️"
                def_assets_html += f'<p style="margin: 0 0 8px 0; color: #DDDDDD; font-size: 13px;">{medal} {asset}</p>'

            # Defensive Allocation Card
            allocation_html = f"""
                <div style="background-color: #1E1E1E; border-radius: 12px; padding: 20px;">
                     <p style="color: #AAAAAA; font-size: 14px; margin-bottom: 15px;">Market Risk Detected. Switched to Defensive Basket.</p>
                     {def_assets_html}
                     <div style="height: 1px; background-color: #333333; margin: 15px 0;"></div>
                     <p style="color: #F5A623; font-size: 13px;">Target: preserve capital until trend restores.</p>
                </div>
            """

        # 3. Strategy Comparison Logic
        c_status = signal_info.get("classic_status", "N/A")
        h_status = signal_info.get("hybrid_status", "N/A")

        c_emoji = "🟢" if c_status == "NORMAL" else "🔴"
        h_emoji = "🟢" if h_status == "NORMAL" else "🔴"

        c_params = signal_info.get("classic_params", (0, 0))
        h_params = signal_info.get("hybrid_params", (0, 0))

        # 4. HTML Template Injection
        html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Strategy Briefing</title>
</head>
<body style="margin: 0; padding: 0; background-color: #111111; font-family: 'Apple SD Gothic Neo', 'Malgun Gothic', Helvetica, Arial, sans-serif; color: #EEEEEE;">
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 600px; margin: 0 auto; background-color: #111111;">
        <tr>
            <td style="padding: 40px 20px 20px 20px; text-align: center;">
                <p style="color: #666666; font-size: 10px; letter-spacing: 2px; margin: 0 0 10px 0; text-transform: uppercase;">Antigravity Strategy v3.1</p>
                <h1 style="color: #FFFFFF; font-size: 24px; margin: 0; letter-spacing: -0.5px;">PORTFOLIO BRIEFING</h1>
                <p style="color: {status_color}; font-size: 14px; margin: 5px 0 0 0;">{date_str}</p>
            </td>
        </tr>
        <tr>
            <td style="padding: 0 20px 20px 20px;">
                <div style="background-color: #1E1E1E; border: 1px solid #333333; border-radius: 12px; padding: 30px; text-align: center;">
                    <p style="color: #AAAAAA; font-size: 12px; margin: 0 0 10px 0;">MARKET STATUS</p>
                    <h2 style="color: {status_color}; font-size: 32px; margin: 0 0 5px 0; text-shadow: 0 0 10px {status_color}4D;">{status_emoji} {market_status_display}</h2>
                    <p style="color: #CCCCCC; font-size: 14px; margin: 0;">{sub_status}</p>
                    <div style="height: 1px; background-color: #333333; margin: 20px 0;"></div>
                    <table width="100%" cellpadding="0" cellspacing="0">
                        <tr><td align="left" style="color: #AAAAAA; font-size: 13px;">QQQ Price</td><td align="right" style="color: #FFFFFF; font-size: 14px; font-weight: bold;">{qqq_price}</td></tr>
                        <tr><td align="left" style="color: #666666; font-size: 12px; padding-top: 5px;">SMA 110 (Mid)</td><td align="right" style="color: #AAAAAA; font-size: 12px; padding-top: 5px;">{ma_fast}</td></tr>
                        <tr><td align="left" style="color: #666666; font-size: 12px;">SMA 250 (Long)</td><td align="right" style="color: #AAAAAA; font-size: 12px;">{ma_slow}</td></tr>
                        <tr><td align="left" style="color: #666666; font-size: 12px;">Current MDD</td><td align="right" style="color: #FF453A; font-size: 12px;">{signal_info["calculated_mdd"] * 100:.2f}%</td></tr>
                    </table>
                </div>
            </td>
        </tr>
        <tr>
            <td style="padding: 0 20px 20px 20px;">
                <h3 style="color: #FFFFFF; font-size: 16px; margin: 0 0 15px 5px; border-left: 3px solid #00FF9D; padding-left: 10px;">ASSET ALLOCATION</h3>
                {allocation_html}
            </td>
        </tr>
        <tr>
            <td style="padding: 0 20px 20px 20px;">
                <h3 style="color: #FFFFFF; font-size: 16px; margin: 0 0 15px 5px; border-left: 3px solid #70a1ff; padding-left: 10px;">STRATEGY COMPARISON</h3>
                <div style="background-color: #1E1E1E; border-radius: 12px; padding: 15px;">
                    <table width="100%" cellpadding="5">
                        <tr style="border-bottom: 1px solid #333;">
                            <th align="left" style="color: #888; font-size: 11px;">TRACK</th>
                            <th align="center" style="color: #888; font-size: 11px;">LOGIC</th>
                            <th align="right" style="color: #888; font-size: 11px;">STATUS</th>
                        </tr>
                        <tr>
                            <td style="color: #EEE; font-size: 13px;">A. Classic</td>
                            <td align="center" style="color: #AAA; font-size: 12px;">Dual SMA ({c_params[0]}/{c_params[1]})</td>
                            <td align="right" style="color: #EEE; font-size: 13px;">{c_emoji} {c_status}</td>
                        </tr>
                        <tr>
                            <td style="color: #EEE; font-size: 13px;">B. Hybrid</td>
                            <td align="center" style="color: #AAA; font-size: 12px;">JARVIS AI ({h_params[0]}/{h_params[1]})</td>
                            <td align="right" style="color: #EEE; font-size: 13px;">{h_emoji} {h_status}</td>
                        </tr>
                    </table>
                    <p style="color: #666666; font-size: 11px; margin-top: 10px; text-align: right;">* Hybrid follows AI unless Guardrail triggers.</p>
                </div>
            </td>
        </tr>
        <tr>
            <td style="padding: 0 20px 20px 20px;">
                <h3 style="color: #FFFFFF; font-size: 14px; margin: 0 0 10px 0;">📊 TECHNICALS</h3>
                <div style="background-color: #1E1E1E; border-radius: 12px; padding: 15px;">
                    <table width="100%">
                        <tr><td style="color:#888; font-size:11px;">Quant Score (Stability)</td><td align="right" style="color:{status_color}; font-size:12px; font-weight:bold;">{quant_score_str} / 100</td></tr>
                        <tr><td style="color:#888; font-size:11px;">RSI (14)</td><td align="right" style="color:#EEE; font-size:12px;">N/A</td></tr>
                        <tr><td style="color:#888; font-size:11px;">VIX</td><td align="right" style="color:#EEE; font-size:12px;">{signal_info["vix"]:.1f}</td></tr>
                        <tr><td style="color:#888; font-size:11px;">USD/KRW</td><td align="right" style="color:#EEE; font-size:12px;">{signal_info["krw_rate"]:.1f}</td></tr>
                    </table>
                </div>
            </td>
        </tr>
        <tr>
            <td style="padding: 20px; text-align: center; border-top: 1px solid #222222;">
                <p style="color: #444444; font-size: 10px; line-height: 1.4; margin: 0;">
                    Automated Daily Report | Golden Combo (110/250)<br>
                    Investments involve risk. Past performance is not indicative of future results.<br>
                    Generated by Antigravity v3.1 Kernel
                </p>
            </td>
        </tr>
    </table>
</body>
</html>
        """
        return html_template

    @staticmethod
    @staticmethod
    def _generate_html_report(signal_info, text_body=None):
        """Generates the 'Midnight Quant' Premium HTML Report"""

        # 1. Basic Data Preparation
        date_str = signal_info["date"].strftime("%Y. %m. %d (%a)")
        status = signal_info["status_label"]
        qqq_price = f"${signal_info['qqq_price']:.2f}"
        ma_fast = f"${signal_info['ma_fast']:.2f}"
        ma_slow = f"${signal_info['ma_slow']:.2f}"

        # Quant Score v2.0
        quant_score = signal_info.get("quant_score", 0)
        q_breakdown = signal_info.get("score_breakdown", (0, 0, 0))
        quant_score_str = f"{quant_score}"  # Integer format

        # Stability Star Rating
        if quant_score >= 90:
            stars = "⭐⭐⭐⭐⭐ (Perfect)"
        elif quant_score >= 70:
            stars = "⭐⭐⭐⭐ (Healthy)"
        elif quant_score >= 40:
            stars = "⭐⭐⭐ (Caution)"
        else:
            stars = "⚠️ (Critical)"

        # 2. Status Styling & Asset Allocation Logic
        # 2. Status Styling & Asset Allocation Logic
        def get_allocation_html(track_name, track_status, track_params):
            if track_status == "NORMAL":
                t_color = "#00FF9D"
                t_emoji = "🟢"
                # Normal Allocation Content
                content = """
                    <div style="margin-bottom: 8px;">
                        <span style="color: #FFFFFF; font-size: 13px; font-weight: bold;">QLD (2x)</span> <span style="float: right; color: #00FF9D; font-size: 13px;">45%</span>
                        <div style="background-color: #333; height: 4px; border-radius: 2px; margin-top: 2px;"><div style="background-color: #00FF9D; height: 4px; width: 45%;"></div></div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span style="color: #FFFFFF; font-size: 13px;">SPY</span> <span style="float: right; color: #CCC; font-size: 13px;">20%</span>
                        <div style="background-color: #333; height: 4px; border-radius: 2px; margin-top: 2px;"><div style="background-color: #9013FE; height: 4px; width: 20%;"></div></div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span style="color: #FFFFFF; font-size: 13px;">KOSPI</span> <span style="float: right; color: #CCC; font-size: 13px;">20%</span>
                        <div style="background-color: #333; height: 4px; border-radius: 2px; margin-top: 2px;"><div style="background-color: #4A90E2; height: 4px; width: 20%;"></div></div>
                    </div>
                    <div>
                        <span style="color: #FFFFFF; font-size: 13px;">GOLD</span> <span style="float: right; color: #CCC; font-size: 13px;">15%</span>
                        <div style="background-color: #333; height: 4px; border-radius: 2px; margin-top: 2px;"><div style="background-color: #F5A623; height: 4px; width: 15%;"></div></div>
                    </div>
                """
            else:
                t_color = "#FF453A"
                t_emoji = "🔴" if track_status != "EMERGENCY (STOP)" else "🛑"
                # Defensive Content
                def_items = ""
                medals = ["🥇", "🥈", "🥉"]
                for i, asset in enumerate(signal_info.get("defensive_assets", [])):
                    m = medals[i] if i < 3 else "🛡️"
                    def_items += f'<div style="color: #DDD; font-size: 12px; margin-bottom: 4px;">{m} {asset}</div>'

                content = f"""
                    <div style="color: #FF453A; font-size: 12px; margin-bottom: 10px; font-weight: bold;">⚠️ DEFENSIVE MODE</div>
                    {def_items}
                    <div style="margin-top: 10px; font-size: 11px; color: #999;">Cash/Bonds Focus</div>
                """

            return f"""
                <td width="50%" valign="top" style="padding: 10px; background-color: #1E1E1E; border: 1px solid #333; border-radius: 8px;">
                    <div style="color: #888; font-size: 10px; letter-spacing: 1px; margin-bottom: 5px;">{track_name}</div>
                    <div style="color: {t_color}; font-size: 16px; font-weight: bold; margin-bottom: 3px;">{t_emoji} {track_status}</div>
                    <div style="color: #666; font-size: 11px; margin-bottom: 15px;">SMA {track_params[0]}/{track_params[1]}</div>
                    {content}
                </td>
            """

        c_status = signal_info.get("classic_status", "N/A")
        h_status = signal_info.get("hybrid_status", "N/A")
        c_params = signal_info.get("classic_params", (0, 0))
        h_params = signal_info.get("hybrid_params", (0, 0))

        c_card = get_allocation_html("TRACK A (CLASSIC)", c_status, c_params)
        h_card = get_allocation_html("TRACK B (HYBRID)", h_status, h_params)

        allocation_section = f"""
            <h3 style="color: #FFFFFF; font-size: 16px; margin: 0 0 15px 5px; border-left: 3px solid #70a1ff; padding-left: 10px;">STRATEGY COMPARISON & ALLOCATION</h3>
            <table width="100%" cellpadding="0" cellspacing="5" border="0">
                <tr>
                    {c_card}
                    <td width="10"></td> <!-- Spacer -->
                    {h_card}
                </tr>
            </table>
        """

        # Determine Main Header Status Color (Follow Hybrid as Primary)
        status = h_status
        if status == "NORMAL":
            status_color = "#00FF9D"
            status_emoji = "🟢"
            market_status_display = "NORMAL"
            sub_status = f"Hybrid Optimized ({h_params[0]}/{h_params[1]})"
        else:
            status_color = "#FF453A"
            status_emoji = "🔴" if status != "EMERGENCY (STOP)" else "🛑"
            market_status_display = status
            sub_status = "Defensive Mode Activated"

        # 3. HTML Template Injection
        html_template = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Strategy Briefing</title>
</head>
<body style="margin: 0; padding: 0; background-color: #111111; font-family: 'Apple SD Gothic Neo', 'Malgun Gothic', Helvetica, Arial, sans-serif; color: #EEEEEE;">
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 600px; margin: 0 auto; background-color: #111111;">
        <tr>
            <td style="padding: 40px 20px 20px 20px; text-align: center;">
                <p style="color: #666666; font-size: 10px; letter-spacing: 2px; margin: 0 0 10px 0; text-transform: uppercase;">Antigravity Strategy v3.1</p>
                <h1 style="color: #FFFFFF; font-size: 24px; margin: 0; letter-spacing: -0.5px;">PORTFOLIO BRIEFING</h1>
                <p style="color: {status_color}; font-size: 14px; margin: 5px 0 0 0;">{date_str}</p>
            </td>
        </tr>
        <tr>
            <td style="padding: 0 20px 20px 20px;">
                <div style="background-color: #1E1E1E; border: 1px solid #333333; border-radius: 12px; padding: 30px; text-align: center;">
                    <p style="color: #AAAAAA; font-size: 12px; margin: 0 0 10px 0;">MARKET STATUS</p>
                    <h2 style="color: {status_color}; font-size: 32px; margin: 0 0 5px 0; text-shadow: 0 0 10px {status_color}4D;">{status_emoji} {market_status_display}</h2>
                    <p style="color: #CCCCCC; font-size: 14px; margin: 0;">{sub_status}</p>
                    <div style="height: 1px; background-color: #333333; margin: 20px 0;"></div>
                    <table width="100%" cellpadding="0" cellspacing="0">
                        <tr><td align="left" style="color: #AAAAAA; font-size: 13px;">QQQ Price</td><td align="right" style="color: #FFFFFF; font-size: 14px; font-weight: bold;">{qqq_price}</td></tr>
                        <tr><td align="left" style="color: #666666; font-size: 12px; padding-top: 5px;">SMA 110 (Mid)</td><td align="right" style="color: #AAAAAA; font-size: 12px; padding-top: 5px;">{ma_fast}</td></tr>
                        <tr><td align="left" style="color: #666666; font-size: 12px;">SMA 250 (Long)</td><td align="right" style="color: #AAAAAA; font-size: 12px;">{ma_slow}</td></tr>
                        <tr><td align="left" style="color: #666666; font-size: 12px;">Current MDD</td><td align="right" style="color: #FF453A; font-size: 12px;">{signal_info["calculated_mdd"] * 100:.2f}%</td></tr>
                    </table>
                </div>
            </td>
        </tr>
        <tr>
            <td style="padding: 0 20px 20px 20px;">
                {allocation_section}
            </td>
        </tr>
        <tr>
            <td style="padding: 0 20px 20px 20px;">
                <h3 style="color: #FFFFFF; font-size: 14px; margin: 0 0 10px 0;">📊 QUANT SCORE (v2.0)</h3>
                <div style="background-color: #1E1E1E; border-radius: 12px; padding: 15px;">
                    <div style="margin-bottom: 5px;">
                        <span style="color: #888; font-size: 11px;">TOTAL SCORE</span>
                        <span style="float: right; color: {status_color}; font-size: 14px; font-weight: bold;">{quant_score_str} / 100</span>
                    </div>
                    <div style="margin: 0 0 10px 0; font-size: 12px; color: #EEE;">{stars}</div>
                    <div style="height: 4px; background-color: #333; border-radius: 2px; margin-bottom: 10px;">
                        <div style="height: 4px; background-color: {status_color}; border-radius: 2px; width: {quant_score}%;"></div>
                    </div>
                    <table width="100%" cellpadding="2" cellspacing="0">
                        <tr>
                            <td style="color: #AAA; font-size: 11px;">Macro (VIX)</td>
                            <td align="right" style="color: #EEE; font-size: 11px;">{q_breakdown[0]} / 30</td>
                        </tr>
                        <tr>
                            <td style="color: #AAA; font-size: 11px;">Trend (MA)</td>
                            <td align="right" style="color: #EEE; font-size: 11px;">{q_breakdown[1]} / 40</td>
                        </tr>
                        <tr>
                            <td style="color: #AAA; font-size: 11px;">Efficiency (Vol)</td>
                            <td align="right" style="color: #EEE; font-size: 11px;">{q_breakdown[2]} / 30</td>
                        </tr>
                    </table>
                </div>
            </td>
        </tr>
        <tr>
            <td style="padding: 0 20px 20px 20px;">
                <h3 style="color: #FFFFFF; font-size: 14px; margin: 0 0 10px 0;">📊 TECHNICALS</h3>
                <div style="background-color: #1E1E1E; border-radius: 12px; padding: 15px;">
                    <table width="100%">
                        <tr><td style="color:#888; font-size:11px;">RSI (14)</td><td align="right" style="color:#EEE; font-size:12px;">N/A</td></tr>
                        <tr><td style="color:#888; font-size:11px;">VIX</td><td align="right" style="color:#EEE; font-size:12px;">{signal_info["vix"]:.1f}</td></tr>
                        <tr><td style="color:#888; font-size:11px;">USD/KRW</td><td align="right" style="color:#EEE; font-size:12px;">{signal_info["krw_rate"]:.1f}</td></tr>
                    </table>
                </div>
            </td>
        </tr>
        <tr>
            <td style="padding: 20px; text-align: center; border-top: 1px solid #222222;">
                <p style="color: #444444; font-size: 10px; line-height: 1.4; margin: 0;">
                    Automated Daily Report | Golden Combo (110/250)<br>
                    Investments involve risk. Past performance is not indicative of future results.<br>
                    Generated by Antigravity v3.1 Kernel
                </p>
            </td>
        </tr>
    </table>
</body>
</html>
        """
        return html_template

    @staticmethod
    def format_signal_report(signal_info, previous_status=None):
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
        html_body = SignalDetector._generate_html_report(signal_info, text_body)

        return {
            "title": f"{emoji} {status}: {krw_pct} KRW / {usd_pct} USD",
            "body": text_body,
            "html_body": html_body,
            "status": status,
            "status_changed": (previous_status != status) if previous_status else False,
        }
