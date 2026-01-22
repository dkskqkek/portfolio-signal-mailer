import pandas as pd
import numpy as np

def run_simulation_and_compare(df_input):
    """
    [검증 시뮬레이션]
    기존 전략(Original) vs 최적화 전략(Optimized)을 비교하여
    제거된 노이즈 신호와 매매 횟수 감소율을 분석합니다.
    """
    # 원본 데이터 보존을 위해 복사
    df = df_input.copy()
    
    # -----------------------------------------------------------
    # 1. 데이터 전처리 및 지표 계산 (HMM Smoothing)
    # -----------------------------------------------------------
    # HMM 스무딩: 3일 이동 중앙값 (노이즈로 인한 상태 깜빡임 제거)
    if 'hmm_state' in df.columns:
        df['hmm_smooth'] = df['hmm_state'].rolling(window=3).median().fillna(method='bfill')
    else:
        print("Error: 'hmm_state' 컬럼이 데이터프레임에 없습니다.")
        return

    # -----------------------------------------------------------
    # 2. 전략 로직 구현
    # -----------------------------------------------------------
    
    # 결과 저장용 리스트
    trades_orig = [] # (Date, Type)
    trades_opt = []
    
    # 포지션 상태 (1: QQQ, -1: JEPI/Cash)
    pos_orig = 1
    pos_opt = 1
    
    # 루프 실행
    for i in range(1, len(df)):
        curr_date = df.index[i]
        
        # --- [A] 기존 로직 (Original) ---
        # 매도: (위기 & RSI<45) OR (정상 & (모멘텀붕괴 OR 변동성급등))
        is_crisis_orig = (df['hmm_state'].iloc[i] == 2)
        
        # 모멘텀 붕괴: RSI 70 -> 40 급락 (예시 로직)
        mom_crash = (df['rsi'].iloc[i-1] > 70 and df['rsi'].iloc[i] < 40)
        vol_spike = (df['vix'].iloc[i] > 30)
        
        sell_cond_orig = (is_crisis_orig and df['rsi'].iloc[i] < 45) or \
                         (not is_crisis_orig and (mom_crash or vol_spike))
        
        # 필터: ADX < 20이면 매도 무시 (약한 필터)
        if df['adx'].iloc[i] < 20:
            sell_cond_orig = False

        # 포지션 스위칭
        if pos_orig == 1 and sell_cond_orig:
            pos_orig = -1
            trades_orig.append((curr_date, 'SELL'))
        elif pos_orig == -1 and not sell_cond_orig: # 조건 해제 시 즉시 매수
            pos_orig = 1
            trades_orig.append((curr_date, 'BUY'))


        # --- [B] 최적화 로직 (Optimized) ---
        # 변경점 1: HMM 3일 스무딩 값 사용
        is_crisis_opt = (df['hmm_smooth'].iloc[i] == 2)
        
        # 변경점 2: ADX 필터 25로 상향
        adx_filter = (df['adx'].iloc[i] < 25)
        
        # 매도 로직 (조건은 유사하나 입력 데이터가 스무딩됨)
        mom_crash_opt = (df['rsi'].iloc[i-1] > 65 and df['rsi'].iloc[i] < 40) # 조건 미세 조정
        
        sell_cond_opt = (is_crisis_opt and df['rsi'].iloc[i] < 45) or \
                        (not is_crisis_opt and (mom_crash_opt or vol_spike))
        
        if adx_filter:
            sell_cond_opt = False

        # 포지션 스위칭 (변경점 3: Hysteresis 적용)
        if pos_opt == 1 and sell_cond_opt:
            pos_opt = -1
            trades_opt.append((curr_date, 'SELL'))
        elif pos_opt == -1:
            # 매수 복귀 조건 강화: 매도 조건 해제 AND RSI > 50 (데드밴드)
            if not sell_cond_opt and df['rsi'].iloc[i] > 50:
                pos_opt = 1
                trades_opt.append((curr_date, 'BUY'))
    
    # -----------------------------------------------------------
    # 3. 결과 분석 및 출력
    # -----------------------------------------------------------
    print("="*50)
    print(f"📊 [시뮬레이션 결과 분석] 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print("="*50)
    
    count_orig = len(trades_orig)
    count_opt = len(trades_opt)
    reduction = ((count_orig - count_opt) / count_orig * 100) if count_orig > 0 else 0
    
    print(f"1. 신호 발생 횟수 (노이즈 제거율)")
    print(f"   - 기존 로직  : {count_orig} 회")
    print(f"   - 최적화 로직: {count_opt} 회")
    print(f"   - 📉 감소율   : {reduction:.1f}% (불필요한 매매 제거)")
    
    print("\n2. 제거된 신호 (Optimized에서 사라진 매매 날짜)")
    print("-" * 50)
    
    # 날짜 집합 비교
    dates_orig = set([t[0] for t in trades_orig])
    dates_opt = set([t[0] for t in trades_opt])
    removed_dates = sorted(list(dates_orig - dates_opt))
    
    if not removed_dates:
        print("   제거된 신호가 없습니다.")
    else:
        for d in removed_dates:
            # 해당 날짜가 기존에 매수였는지 매도였는지 확인
            type_str = next((t[1] for t in trades_orig if t[0] == d), "UNKNOWN")
            print(f"   ❌ [제거됨] {d.date()} : {type_str}")

    print("="*50)

# 사용 예시:
# run_simulation_and_compare(df) 
# 주의: df에는 'hmm_state', 'rsi', 'vix', 'adx' 컬럼이 있어야 하며, index는 Datetime이어야 함.