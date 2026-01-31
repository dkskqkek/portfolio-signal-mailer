def _build_header(date_str, color_sig, version="v4.1"):
    return f"""
            <div style="background-color: #1a1a20; padding: 20px; border-bottom: 2px solid {color_sig};">
                <h2 style="margin: 0; color: #fff;">🌌 Antigravity <span style="font-weight:lighter; color:#888;">{version}</span></h2>
                <div style="margin-top: 5px; font-size: 12px; color: #666;">{date_str} | The Reality Engine</div>
            </div>
    """


def _build_signal_banner(signal, action_plan, color_sig):
    return f"""
            <div style="padding: 30px; text-align: center; background: linear-gradient(180deg, #1a1a20 0%, #0f0f12 100%);">
                <div style="font-size: 14px; color: #888; margin-bottom: 10px;">MARKET REGIME</div>
                <div style="font-size: 36px; font-weight: bold; color: {color_sig}; letter-spacing: 2px;">{signal}</div>
                <div style="margin-top: 20px; font-size: 16px; color: #fff; border: 1px solid #444; display: inline-block; padding: 10px 20px; border-radius: 5px;">
                    {action_plan}
                </div>
            </div>
    """


def _build_rebalancing_section(orders_text):
    if not orders_text:
        return ""
    return f"""
            <div style="padding: 20px; border-top: 1px solid #333; background: #1a1a1e;">
                <h3 style="margin: 0 0 10px 0; font-size: 14px; color: #aaa;">📋 Acitve Orders</h3>
                <div style="background: #222; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 13px; color: #ccc; line-height: 1.6; white-space: pre-wrap;">{orders_text}</div>
            </div>
        """


def _build_sniper_tags(sniper_signal):
    tags = []
    if sniper_signal.is_buy:
        tags.append(
            '<span style="background: #004d00; color: #00ff41; padding: 4px 8px; border-radius: 3px; font-size: 11px;">🟢 BUY SIGNAL</span>'
        )
    if sniper_signal.is_sell:
        tags.append(
            '<span style="background: #4d0000; color: #ff3333; padding: 4px 8px; border-radius: 3px; font-size: 11px;">🔴 SELL SIGNAL</span>'
        )
    if sniper_signal.fear_zone:
        tags.append(
            '<span style="background: #4d0000; color: #ff6666; padding: 4px 8px; border-radius: 3px; font-size: 11px;">😱 FEAR ZONE</span>'
        )
    if sniper_signal.buy_window:
        tags.append(
            '<span style="background: #4d2600; color: #ffaa00; padding: 4px 8px; border-radius: 3px; font-size: 11px;">🟠 BUY WINDOW</span>'
        )
    return "\n".join(tags)


def _build_sniper_section(sniper_signal, sniper_color):
    tags_html = _build_sniper_tags(sniper_signal)
    return f"""
            <div style="padding: 20px; border-top: 1px solid #333; background: #16161a;">
                <h3 style="margin: 0 0 10px 0; font-size: 14px; color: #aaa;">🎯 Index Sniper (QQQ Weekly)</h3>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div style="background: #222; padding: 10px; border-radius: 4px;">
                        <span style="font-size: 11px; color: #888;">State</span><br>
                        <span style="font-size: 14px; font-weight: bold; color: {sniper_color};">{sniper_signal.current_state}</span>
                    </div>
                    <div style="background: #222; padding: 10px; border-radius: 4px;">
                        <span style="font-size: 11px; color: #888;">Momentum</span><br>
                        <span style="font-size: 13px; color: #fff;">{sniper_signal.momentum_status}</span>
                    </div>
                </div>

                <div style="margin-top: 10px; display: flex; gap: 10px;">
                    {tags_html}
                </div>
            </div>
    """


def _build_metrics_grid(dist_sma250, vix, sma_color):
    return f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 20px;">
                <div style="background: #1a1a20; padding: 15px; border-radius: 5px;">
                    <div style="font-size: 11px; color: #666;">QQQ vs SMA250</div>
                    <div style="font-size: 18px; font-weight: bold; color: {sma_color};">
                        {dist_sma250 * 100:+.2f}%
                    </div>
                </div>
                <div style="background: #1a1a20; padding: 15px; border-radius: 5px;">
                    <div style="font-size: 11px; color: #666;">VIX Index</div>
                    <div style="font-size: 18px; font-weight: bold; color: #fff;">{vix:.2f}</div>
                </div>
            </div>
    """


def _build_council_section(verdict, discount):
    return f"""
            <div style="padding: 20px; border-top: 1px solid #333;">
                <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #aaa;">⚖️ The Council of Risk</h3>
                <div style="background: #222; padding: 15px; border-left: 3px solid #888; font-style: italic; color: #ccc; font-size: 14px;">
                    "{verdict}"
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #666; text-align: right;">
                    Discount Factor: <strong>{discount}x</strong>
                </div>
            </div>
    """


def _build_defensive_rows(data_list):
    rows = ""
    for asset in data_list:
        role = "Cash Proxy" if asset in ["BIL", "SHY"] else "Defensive/Macro"
        rows += f"""
        <tr style="border-top: 1px solid #333;">
            <td style="padding: 10px 0; color: #fff;"><b>{asset}</b></td>
            <td style="padding: 10px 0; color: #888;">{role}</td>
        </tr>
        """
    return rows


def _build_defensive_section(assets):
    rows_html = _build_defensive_rows(assets)
    return f"""
            <div style="padding: 20px; border-top: 1px solid #333;">
                <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #aaa;">🛡️ This Month's Shield</h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <tr style="color: #666; text-align: left;">
                        <th style="padding-bottom: 10px;">Asset</th>
                        <th style="padding-bottom: 10px;">Role</th>
                    </tr>
                    {rows_html}
                </table>
            </div>
    """


def generate_html_report(report_data):
    """
    Generates a premium 'Midnight Quant' HTML email report.
    Refactored to reduce f-string nesting depth for LSP stability.
    """
    r = report_data

    # Color Logic
    color_sig = "#00ff41" if r["signal"] == "NORMAL" else "#ff3333"

    # Helper for Sniper Status color
    sniper_state = r["sniper_signal"].current_state
    sniper_color = "#00ff41" if sniper_state == "HOLD" else "#ff3333"

    # Helper for SMA color
    sma_color = "#00ff41" if r["dist_sma250"] > 0 else "#ff3333"

    # --- Build HTML Parts ---
    parts = []

    # 1. Structure Start
    parts.append("""
    <html>
    <body style="font-family: 'Roboto', sans-serif; background-color: #0f0f12; color: #e0e0e0; padding: 20px;">
        <div style="max_width: 600px; margin: 0 auto; border: 1px solid #333; border-radius: 10px; overflow: hidden;">
    """)

    # 2. Components
    parts.append(_build_header(r["date"], color_sig))
    parts.append(_build_signal_banner(r["signal"], r["action_plan"], color_sig))

    if "rebalancing_orders" in r:
        parts.append(_build_rebalancing_section(r.get("rebalancing_orders")))

    parts.append(_build_sniper_section(r["sniper_signal"], sniper_color))
    parts.append(_build_metrics_grid(r["dist_sma250"], r["vix"], sma_color))
    parts.append(_build_council_section(r["council_verdict"], r["council_discount"]))
    parts.append(_build_defensive_section(r["defensive_selection"]))

    # Footer
    parts.append("""
            <div style="padding: 20px; text-align: center; font-size: 10px; color: #444; border-top: 1px solid #333;">
                Generated by Antigravity Engine v4.1 | <span style="color:#666;">Tax Deferral Logic Active</span>
            </div>
        </div>
    </body>
    </html>
    """)

    return "".join(parts)
