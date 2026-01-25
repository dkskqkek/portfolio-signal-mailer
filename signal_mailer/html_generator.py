# -*- coding: utf-8 -*-
"""
HTML Report Generator Module
Separated from SignalDetector to resolve staticmethod callability issues and improve modularity.
"""


def generate_html_report(signal_info, text_body=None):
    """Generates the 'Midnight Quant' Premium HTML Report with 2-Track Comparison"""

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
    # Check if 'status' is already defined above? Yes, line 12.
    # But logic below re-defines status_color.

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

    # The Council HTML Generation
    council_html = ""
    council_verdict = signal_info.get("council_verdict")

    if council_verdict:
        c_discount = council_verdict.discount_factor
        c_reason = council_verdict.reason

        # Simple color coding for the verdict
        if c_discount >= 0.9:
            c_color = "#00FF9D"  # Green
            c_title = "APPROVED"
            c_icon = "⚖️"
        elif c_discount >= 0.7:
            c_color = "#F5A623"  # Orange
            c_title = "CAUTION"
            c_icon = "⚠️"
        else:
            c_color = "#FF453A"  # Red
            c_title = "RESTRICT"
            c_icon = "🚫"

        percent_view = int(c_discount * 100)

        council_html = f"""
        <h3 style="color: #FFFFFF; font-size: 14px; margin: 0 0 10px 0;">⚖️ THE COUNCIL (AI RISK COMMITTEE)</h3>
        <div style="background-color: #1E1E1E; border-radius: 12px; padding: 15px; border-left: 4px solid {c_color};">
            <div style="margin-bottom: 8px;">
                <span style="color: {c_color}; font-size: 14px; font-weight: bold;">{c_icon} {c_title}</span>
                <span style="float: right; color: #888; font-size: 12px;">Exposure Cap: {percent_view}%</span>
            </div>
            <p style="color: #DDD; font-size: 12px; line-height: 1.5; margin: 0;">
                "{c_reason}"
            </p>
        </div>
        """
    else:
        # Fallback if no API key or error
        council_html = """
        <h3 style="color: #FFFFFF; font-size: 14px; margin: 0 0 10px 0;">⚖️ THE COUNCIL</h3>
        <div style="background-color: #1E1E1E; border-radius: 12px; padding: 15px;">
             <p style="color: #666; font-size: 11px; margin: 0;">AI Risk Module Offline (No API Key or Recess)</p>
        </div>
        """

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
                {council_html}
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
