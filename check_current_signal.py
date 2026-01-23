# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('d:/gg')
from signal_mailer.signal_detector import SignalDetector
import yfinance as yf
from datetime import datetime

def run_analysis():
    print("--- Portfolio Signal Analysis ---")
    detector = SignalDetector()
    
    print("Fetching data and calculating signal...")
    signal_info = detector.detect()
    
    spy = yf.Ticker("SPY")
    hist = spy.history(period="250d")
    current_price = hist['Close'].iloc[-1]
    ma200 = hist['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
    
    print(f"\n[Market Status]")
    print(f"Date: {signal_info['date'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"SPY Current: ${current_price:.2f}")
    print(f"SPY MA200 (EMA): ${ma200:.2f}")
    print(f"Status: {'ABOVE' if current_price > ma200 else 'BELOW'} MA200")
    
    print(f"\n[Signal Logic: {signal_info.get('strategy', 'unknown')}]")
    print(f"MF Score (Psychology): {signal_info['mf_score']:.1f} (Threshold: 40)")
    print(f"M1 Danger (Technical): {signal_info['m1_danger']}")
    print(f"Overall Result: {'DANGER' if signal_info['is_danger'] else 'NORMAL'}")
    print(f"Reason: {signal_info['reason']}")
    
    if signal_info['is_danger']:
        print("\n[Action Plan]")
        print("- SELL QQQ / BUY XLP (or JEPI based on weight config)")
        print("- Reduce overall equity exposure")
    else:
        print("\n[Action Plan]")
        print("- HOLD QQQ / Core ETFs")
        print("- Maintain target asset allocation")

if __name__ == "__main__":
    run_analysis()
