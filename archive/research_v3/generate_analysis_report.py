# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('d:/gg')
from signal_mailer.signal_detector import SignalDetector
import yfinance as yf
from datetime import datetime

template = """# Portfolio Signal Analysis Report (Current)

## 1. Market Overview
- **Date**: {date}
- **SPY Price**: ${spy_price:.2f}
- **SPY 200-day EMA**: ${ma200:.2f}
- **Market Regime**: **{regime}** (Price is {distance:.2f}% {distance_dir} MA200)

## 2. Signal Logic ({strategy})
- **Hard Floor Status**: {floor_status}
- **Fusion Score (Psychology)**: {mf_score:.1f} / 100
- **Sentinel Status (Technical)**: {m1_status}

## 3. Final Verdict
> [!IMPORTANT]
> **Status**: **{status}**
> **Reason**: {reason}

---

## 4. Action Guide
{action_guide}
"""

def generate_report():
    detector = SignalDetector()
    results = detector.detect()
    
    spy = yf.Ticker("SPY")
    hist = spy.history(period="250d")
    current_price = hist['Close'].iloc[-1]
    ma200 = hist['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
    
    dist = (current_price - ma200) / ma200 * 100
    regime = "BULLISH" if current_price > ma200 else "BEARISH"
    dist_dir = "above" if current_price > ma200 else "below"
    
    status = "DANGER" if results['is_danger'] else "NORMAL"
    floor_status = "BREACHED (Danger)" if current_price < ma200 else "SAFE (Above Floor)"
    m1_status = "STRESS DETECTED" if results['m1_danger'] else "STABLE"
    
    if status == "DANGER":
        action_guide = "- **SELL QQQ / BUY XLP (or JEPI)** as per configured weights.\n- Reduce risky asset exposure.\n- Monitor for recovery above 200 SMA."
    else:
        action_guide = "- **Maintain target allocation** (QQQ / Core ETFs).\n- Market is in a supported regime.\n- No immediate defensive action required."

    report = template.format(
        date=results['date'].strftime('%Y-%m-%d %H:%M:%S'),
        spy_price=current_price,
        ma200=ma200,
        regime=regime,
        distance=abs(dist),
        distance_dir=dist_dir,
        strategy=results.get('strategy', 'hybrid'),
        floor_status=floor_status,
        mf_score=results['mf_score'],
        m1_status=m1_status,
        status=status,
        reason=results['reason'],
        action_guide=action_guide
    )
    
    with open(r'C:\Users\gamja\.gemini\antigravity\brain\1c82236c-ee57-4550-a5b0-e45fc300958e\market_analysis.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print("Report generated successfully.")

if __name__ == "__main__":
    generate_report()
