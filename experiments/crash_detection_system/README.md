# Early Crash Detection System

**고급 정량 분석 기반 충돌회피 감지 시스템**

금융 시장의 위기 국면을 조기에 감지하기 위해 Kalman Filter, Hidden Markov Model (HMM), 다중 기술 지표를 활용한 하이브리드 신호 생성 시스템입니다.

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  (yfinance, FRED, HYG/IEF Spread, VIX, Market Breadth)    │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│              NOISE FILTERING LAYER                          │
│  (Kalman Filter: Microstructure Noise 제거)               │
│  (ADX Filter: Weak Trend 신호 억제)                        │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│            REGIME DETECTION LAYER                           │
│  (GaussianHMM: 3 States - Bull/Correction/Crisis)          │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│          SIGNAL GENERATION LAYER                            │
│  (Condition A: Momentum Crash)                             │
│  (Condition B: Crisis Regime)                              │
│  (Condition C: Volatility Spike)                           │
│  ► SELL Signal if (A AND C) OR (B)                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│           BACKTESTING & ANALYTICS                           │
│  (SPY 2010-Present, Metrics: Sharpe, Sortino, MDD)        │
└─────────────────────────────────────────────────────────────┘
```

## 핵심 특징

### 1. 정밀도 (Precision) 최우선
- **False Positive 최소화**: Condition A AND C 동시 만족 필요
- **높은 특이성**: 명확한 신호만 트리거
- **리스크 관리**: ADX < 20 노이즈 필터링

### 2. 멀티팩터 분석
- **기술성 (Breadth)**: RSI 모멘텀 역전
- **레짐 감지**: HMM 기반 시장 상태 인식
- **변동성**: VIX 기간구조 및 수준

### 3. 자동화 최대화
- **멱등성 확보**: 캐시 메커니즘으로 중복 실행 안전성
- **Fail-Safe 설계**: 에러 발생 시 우아한 폴백
- **구조화된 로깅**: 모든 단계에서 감사 기록

### 4. 데이터 무결성
- **전체 해석 (Full Context)**: 부분 요약 금지
- **결측치 처리**: Forward-fill & Backward-fill
- **타임스탐프 검증**: 데이터 연속성 보증

## 설치 및 실행

### 1. 환경 설정

```bash
# 디렉토리 이동
cd crash_detection_system

# 가상환경 생성 (선택)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 검증 실행 (Pre-Flight Check)

```bash
# 모든 요구사항 검증 (8개 테스트)
python tests/test_validation.py
```

**검증 항목:**
1. ✓ DataFetcher 기본 기능
2. ✓ 데이터 무결성 (Full Context)
3. ✓ Kalman Filter 수학적 정확성
4. ✓ HMM 레짐 감지
5. ✓ 기술 지표 계산 (RSI, ADX)
6. ✓ 신호 생성 로직
7. ✓ 엣지 케이스 (NaN 처리)
8. ✓ 멱등성 (캐시 메커니즘)

### 3. 전체 파이프라인 실행

```bash
# 메인 파이프라인 실행
python src/main.py
```

**실행 단계:**
1. **Data Fetching & Preparation**: SPY + VIX + 신용스프레드 수집
2. **Signal Processing**: Kalman + HMM + 기술 지표
3. **Backtesting**: 2010년 이후 SPY 데이터로 검증
4. **Visualization**: 3개 차트 생성 (수익률, 신호, 레짐)
5. **Report**: JSON 형식 리포트 생성

### 4. 결과 확인

```
results/
├── 01_cumulative_returns.png      # 전략 vs 벤치마크 수익률
├── 02_signals_and_indicators.png  # 신호 + RSI
├── 03_hmm_regimes.png            # 시장 레짐 시각화
└── backtest_report.json          # 성과 메트릭스
```

## 주요 성과 지표

| 지표 | 목표 | 설명 |
|------|------|------|
| **Sharpe Ratio** | > 1.5 | 리스크 대비 초과 수익 |
| **Sortino Ratio** | > 2.0 | 하방위험 대비 초과 수익 |
| **Max Drawdown** | < 50% | 최대 낙폭 제한 |
| **CAGR** | > 10% | 연평균 성장률 |
| **Win Rate** | > 50% | 양수 수익일 확률 |

## 신호 생성 로직

### Condition A: 기술 모멘텀 급락
```
IF RSI(smoothed) < 40 AND RSI(prev) > 70
→ SELL signal (나스닥/테크 급조정 신호)
```

### Condition B: 위기 레짐
```
IF HMM State == 2 (Crisis)
→ HIGH sensitivity (RSI < 45에서도 SELL)
```

### Condition C: 변동성 스파이크
```
IF VIX > 30 OR VIX Term Structure Inverted
→ Near-term stress 신호
```

### 최종 신호
```
IF State == 2:
    SELL when RSI < 45  (높은 민감도)
ELSE:
    SELL when (A AND C)  (높은 특이성)

IF ADX < 20:
    SUPPRESS all signals (노이즈 필터)
```

## 코드 구조

```
src/
├── __init__.py                 # 패키지 정의
├── data_fetcher.py            # 데이터 수집 및 캐싱
├── signal_processor.py         # Kalman, HMM, 기술 지표
├── strategy.py                 # 신호 생성 로직
└── main.py                    # 통합 파이프라인

tests/
└── test_validation.py         # Pre-Flight Check (8개 테스트)

requirements.txt               # 패키지 의존성
```

## 주요 설계 결정 (Decision Records)

### 1. Kalman Filter 1차 모델
- **이유**: 단순하면서도 마이크로스트럭처 노이즈 효과적 제거
- **State**: `x_t = x_{t-1}` (상수 속도 모델)
- **장점**: 해석 가능하고 계산 효율적

### 2. GaussianHMM 3 States
- **이유**: 시장의 주요 패턴 (Bull/Correction/Crisis) 표현
- **Features**: Returns + Realized Volatility (정보력 높음)
- **학습**: EM algorithm, 1000 iterations

### 3. Precision > Recall
- **설계**: 위양성 최소화를 명시적 목표
- **Specificity**: Condition A AND C 동시 필요
- **결과**: 신뢰성 높은 신호, 거래 횟수 감소

### 4. 멱등성 우선
- **캐시**: 동일 요청에 대해 동일 결과 보장
- **재실행**: 안전하게 중단/재개 가능
- **감사**: 모든 단계의 로그 기록

## 기술 스택

| 영역 | 라이브러리 | 목적 |
|------|-----------|------|
| **데이터** | `yfinance`, `pandas_datareader` | 시세 및 매크로 데이터 |
| **분석** | `pandas`, `numpy`, `pandas_ta` | 데이터 처리 및 기술 지표 |
| **신호처리** | `pykalman`, `hmmlearn` | 노이즈 제거, 레짐 감지 |
| **백테스팅** | `vectorbt` | 벡터화 백테스팅 |
| **시각화** | `matplotlib`, `seaborn` | 결과 시각화 |

## 주의사항

1. **API 의존성**: yfinance는 오픈 API (API 키 불필요)
2. **데이터 지연**: 실시간 데이터 아님 (1~2일 지연)
3. **과최적화 위험**: 백테스트는 과거 기준 (향후 성과 보장 없음)
4. **거래 비용**: 백테스트에 미포함 (실제 운용시 수익률 감소)

## 라이선스

이 프로젝트는 교육 및 연구 목적입니다. 실제 투자 활용은 신중한 검토와 위험 관리가 필수입니다.

## 저자

Senior Quant Developer & AI Signal Processing Expert

## 참고 자료

- [Kalman Filter in Finance](https://arxiv.org/abs/1105.1264)
- [Hidden Markov Models for Market Regime](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1108055)
- [Crash Prediction using Machine Learning](https://www.tandfonline.com/doi/abs/10.1080/13504851.2019.1567174)
