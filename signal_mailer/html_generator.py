def generate_html_report(report_data):
    """
    Generates a premium 'Midnight Quant' HTML email report.
    """
    r = report_data

    # Color Logic
    color_sig = "#00ff41" if r["signal"] == "NORMAL" else "#ff3333"

    html = f"""
    <html>
    <body style="font-family: 'Roboto', sans-serif; background-color: #0f0f12; color: #e0e0e0; padding: 20px;">
        <div style="max_width: 600px; margin: 0 auto; border: 1px solid #333; border-radius: 10px; overflow: hidden;">
            
            <!-- Header -->
            <div style="background-color: #1a1a20; padding: 20px; border-bottom: 2px solid {color_sig};">
                <h2 style="margin: 0; color: #fff;">🌌 Antigravity <span style="font-weight:lighter; color:#888;">v4.1</span></h2>
                <div style="margin-top: 5px; font-size: 12px; color: #666;">{r["date"]} | The Reality Engine</div>
            </div>
            
            <!-- Signal Banner -->
            <div style="padding: 30px; text-align: center; background: linear-gradient(180deg, #1a1a20 0%, #0f0f12 100%);">
                <div style="font-size: 14px; color: #888; margin-bottom: 10px;">MARKET REGIME</div>
                <div style="font-size: 36px; font-weight: bold; color: {color_sig}; letter-spacing: 2px;">{r["signal"]}</div>
                <div style="margin-top: 20px; font-size: 16px; color: #fff; border: 1px solid #444; display: inline-block; padding: 10px 20px; border-radius: 5px;">
                    {r["action_plan"]}
                </div>
            </div>
            
            <!-- Index Sniper Monitor (QQQ) -->
            <div style="padding: 20px; border-top: 1px solid #333; background: #16161a;">
                <h3 style="margin: 0 0 10px 0; font-size: 14px; color: #aaa;">🎯 Index Sniper (QQQ Weekly)</h3>
                
                <!-- Sniper Status Grid -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <!-- State -->
                    <div style="background: #222; padding: 10px; border-radius: 4px;">
                        <span style="font-size: 11px; color: #888;">State</span><br>
                        <span style="font-size: 14px; font-weight: bold; color: {"#00ff41" if r["sniper_signal"].current_state == "HOLD" else "#ff3333"};">{r["sniper_signal"].current_state}</span>
                    </div>
                    <!-- Momentum -->
                    <div style="background: #222; padding: 10px; border-radius: 4px;">
                        <span style="font-size: 11px; color: #888;">Momentum</span><br>
                        <span style="font-size: 13px; color: #fff;">{r["sniper_signal"].momentum_status}</span>
                    </div>
                </div>

                <!-- Signal Indicators -->
                <div style="margin-top: 10px; display: flex; gap: 10px;">
                    {f'<span style="background: #004d00; color: #00ff41; padding: 4px 8px; border-radius: 3px; font-size: 11px;">🟢 BUY SIGNAL</span>' if r["sniper_signal"].is_buy else ""}
                    {f'<span style="background: #4d0000; color: #ff3333; padding: 4px 8px; border-radius: 3px; font-size: 11px;">🔴 SELL SIGNAL</span>' if r["sniper_signal"].is_sell else ""}
                    {f'<span style="background: #4d0000; color: #ff6666; padding: 4px 8px; border-radius: 3px; font-size: 11px;">😱 FEAR ZONE</span>' if r["sniper_signal"].fear_zone else ""}
                    {f'<span style="background: #4d2600; color: #ffaa00; padding: 4px 8px; border-radius: 3px; font-size: 11px;">🟠 BUY WINDOW</span>' if r["sniper_signal"].buy_window else ""}
                </div>
            </div>

            <!-- Metrics Grid -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 20px;">
                <div style="background: #1a1a20; padding: 15px; border-radius: 5px;">
                    <div style="font-size: 11px; color: #666;">QQQ vs SMA250</div>
                    <div style="font-size: 18px; font-weight: bold; color: {"#00ff41" if r["dist_sma250"] > 0 else "#ff3333"};">
                        {r["dist_sma250"] * 100:+.2f}%
                    </div>
                </div>
                <div style="background: #1a1a20; padding: 15px; border-radius: 5px;">
                    <div style="font-size: 11px; color: #666;">VIX Index</div>
                    <div style="font-size: 18px; font-weight: bold; color: #fff;">{r["vix"]:.2f}</div>
                </div>
            </div>
            
            <!-- The Council -->
            <div style="padding: 20px; border-top: 1px solid #333;">
                <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #aaa;">⚖️ The Council of Risk</h3>
                <div style="background: #222; padding: 15px; border-left: 3px solid #888; font-style: italic; color: #ccc; font-size: 14px;">
                    "{r["council_verdict"]}"
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #666; text-align: right;">
                    Discount Factor: <strong>{r["council_discount"]}x</strong>
                </div>
            </div>
            
            <!-- Defensive Selection -->
            <div style="padding: 20px; border-top: 1px solid #333;">
                <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #aaa;">🛡️ This Month's Shield</h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                    <tr style="color: #666; text-align: left;">
                        <th style="padding-bottom: 10px;">Asset</th>
                        <th style="padding-bottom: 10px;">Role</th>
                    </tr>
    """

    for asset in r["defensive_selection"]:
        role = "Cash Proxy" if asset in ["BIL", "SHY"] else "Defensive/Macro"
        html += f"""
        <tr style="border-top: 1px solid #333;">
            <td style="padding: 10px 0; color: #fff;"><b>{asset}</b></td>
            <td style="padding: 10px 0; color: #888;">{role}</td>
        </tr>
        """

    html += """
                </table>
            </div>
            
            <!-- Footer -->
            <div style="padding: 20px; text-align: center; font-size: 10px; color: #444; border-top: 1px solid #333;">
                Generated by Antigravity Engine v4.1 | <span style="color:#666;">Tax Deferral Logic Active</span>
            </div>
        </div>
    </body>
    </html>
    """
    return html
