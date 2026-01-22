#!/usr/bin/env python
"""
Quick Start Script - Early Crash Detection System
최소 설정으로 전체 파이프라인을 빠르게 실행하는 스크립트
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Ensure logs directory exists
log_dir = Path(__file__).parent / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

from main import CrashDetectionPipeline


def main():
    """
    메인 실행 함수
    
    Process:
    1. Pre-Flight Check (검증)
    2. 전체 파이프라인 실행
    3. 결과 리포트 생성
    """
    
    print("\n" + "="*70)
    print("EARLY CRASH DETECTION SYSTEM - QUICK START")
    print("="*70)
    
    # 기본 설정
    config = {
        'ticker': 'SPY',
        'start_date': '2015-01-01',
        'end_date': None,  # Current date
        'cache_dir': './data'
    }
    
    print(f"\nConfiguration:")
    print(f"  Ticker: {config['ticker']}")
    print(f"  Start Date: {config['start_date']}")
    print(f"  Cache Directory: {config['cache_dir']}")
    
    # 결과 디렉토리 생성
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        # 파이프라인 실행
        pipeline = CrashDetectionPipeline(**config)
        results = pipeline.run_full_pipeline()
        
        print("\n" + "="*70)
        print("[SUCCESS] PIPELINE EXECUTION SUCCESSFUL")
        print("="*70)
        print("\nOutputs:")
        print("  - logs/crash_detection.log (Detailed logs)")
        print("  - results/01_cumulative_returns.png")
        print("  - results/02_signals_and_indicators.png")
        print("  - results/03_hmm_regimes.png")
        print("  - results/backtest_report.json (Performance metrics)")
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection (for data download)")
        print("  2. Verify all packages installed: pip install -r requirements.txt")
        print("  3. Check logs/crash_detection.log for detailed errors")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
