# -*- coding: utf-8 -*-
"""
JARVIS Time Machine (Backfill Trainer)
--------------------------------------
자비스의 "경험"을 과거 15년치 시뮬레이션으로 생성하는 스크립트.
- 2010년부터 현재까지 Walk-Forward 방식으로 시간을 돌리며 자비스를 가동합니다.
- 각 시점에서의 시장 국면(Regime)과 최적 파라미터(Optuna)를 기록합니다.
- 결과물: data/jarvis_memory.csv (자비스의 과거 기억 복원)

Author: Antigravity AI Partner
Date: 2026-01-25
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime, timedelta

# 기존 JARVIS 엔진 가져오기
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from jarvis_engine import DataFetcher, RegimeClassifier, ParamTuner

# 설정
START_YEAR = 2010
WINDOW_SIZE = 365 * 5  # 5년치 데이터로 학습
STEP_SIZE_DAYS = 30  # 1달 단위로 제안 갱신 (월간 리밸런싱 가정)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 로깅
logging.basicConfig(
    filename=os.path.join(DATA_DIR, "../logs/time_machine.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] TIME_MACHINE: %(message)s",
)


def run_time_machine():
    print(f"--- ⏳ JARVIS Time Machine 가동 (Target: {START_YEAR}~Present) ---")

    # 1. 전체 역사 데이터 확보
    print("1. 전체 역사 데이터 다운로드 중...")
    fetcher = DataFetcher()
    # 넉넉하게 2005년부터 가져옴 (2010년 시점에 5년치 윈도우 필요)
    full_df = fetcher.get_market_data(days=365 * 20)
    full_df.index = pd.to_datetime(full_df.index).tz_localize(None)  # TZ 제거

    # 시작 시점 찾기
    start_date = datetime(START_YEAR, 1, 1)
    if full_df.index[0] > start_date:
        print(f"⚠️ 데이터 부족: 데이터가 {full_df.index[0].date()}부터 시작합니다.")
        start_date = full_df.index[0] + timedelta(days=WINDOW_SIZE)

    print(f"2. 시뮬레이션 구간: {start_date.date()} ~ {datetime.now().date()}")

    # 시간 여행 루프
    current_date = start_date
    end_date = datetime.now()
    memory_book = []

    pbar = tqdm(total=(end_date - start_date).days // STEP_SIZE_DAYS)

    classifier = RegimeClassifier()
    tuner = ParamTuner()

    while current_date < end_date:
        # 현재 시점의 "과거" Window 데이터 슬라이싱
        window_start = current_date - timedelta(days=WINDOW_SIZE)
        train_df = full_df[
            (full_df.index >= window_start) & (full_df.index < current_date)
        ].copy()

        if len(train_df) < 252 * 2:  # 최소 2년치 데이터 검증
            current_date += timedelta(days=STEP_SIZE_DAYS)
            pbar.update(1)
            continue

        # A. Regime Classification
        feat_df = classifier.prepare_features(train_df)
        labeled_df = classifier.create_labels(feat_df)

        regime = "Unknown"
        crash_prob = 0.0

        if len(labeled_df) > 100 and classifier.train(labeled_df):
            # 현재 시점 직전의 상태로 예측
            r, prob = classifier.predict_current(feat_df)
            regime = r
            crash_prob = prob[2] if len(prob) > 2 else 0.0

        # B. Parameter Tuning (Optuna)
        # 속도를 위해 trial 수를 줄임 (50 -> 20)
        # Time Machine에서는 과거 시점에서의 최적을 찾는 것이므로 미래 데이터(current_date 이후)는 보면 안됨
        best_params = tuner.optimize(train_df)

        if best_params:
            # 기록 (Memory)
            # 당시의 시장 상황 피처들 + 자비스의 제안
            latest_metrics = feat_df.iloc[-1]
            memory = {
                "date": current_date.strftime("%Y-%m-%d"),
                "regime": regime,
                "crash_prob": crash_prob,
                "s1_suggested": best_params["s1"],
                "s2_suggested": best_params["s2"],
                "qqq_price": train_df["QQQ"].iloc[-1],
                "vix": train_df["^VIX"].iloc[-1] if "^VIX" in train_df.columns else 0,
                "vix_ratio": latest_metrics.get("vix_ratio", 0),
                "tnx_chg": latest_metrics.get("tnx_chg", 0),
                "dxy_chg": latest_metrics.get("dxy_chg", 0),
                "qqq_dist": latest_metrics.get("qqq_dist", 0),
            }
            memory_book.append(memory)

            logging.info(
                f"[{current_date.date()}] Regime:{regime} | S1:{best_params['s1']} S2:{best_params['s2']}"
            )

        current_date += timedelta(days=STEP_SIZE_DAYS)
        pbar.update(1)

    pbar.close()

    # 3. 저장
    print("3. 기억 저장 중...")
    result_df = pd.DataFrame(memory_book)
    save_path = os.path.join(DATA_DIR, "jarvis_memory.csv")
    result_df.to_csv(save_path, index=False)

    print(
        f"✅ JARVIS Time Machine 완료! 총 {len(result_df)}달의 기억이 복원되었습니다."
    )
    print(f"📂 저장 경로: {save_path}")


if __name__ == "__main__":
    run_time_machine()
