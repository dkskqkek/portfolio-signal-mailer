# MAMA Lite: 지능형 체제 식별(SRL) 성과 보고서

광운대학교 MAMA 프레임워의 '상태 표현 학습(SRL)'과 '체제 식별' 기술을 Antigravity 환경에 맞게 Lite화하여 구현한 결과입니다.

## 1. 전략 성과 비교 (2010 ~ 현재)
| 지표 | **MAMA Lite (Regime Aware)** | 60/40 Portfolio (Baseline) |
| :--- | :--- | :--- |
| **CAGR (수익률)** | **9.34%** | 10.09% |
| **MDD (최대낙폭)** | **-19.81%** | -27.24% |
| **Sharpe Ratio** | **0.47** | 0.37 |

## 2. AI 체제 식별 결과 (Regime Characteristics)
| Regime ID | 역할 정의 | 평균 VIX | 비중 전략 |
| :--- | :--- | :--- | :--- |
| 3.0 | Bull (Aggressive) | 16.74 | QQQ:0.5, SPY:0.5 |
| 2.0 | Sideways (Balanced) | 16.01 | SPY:0.6, TLT:0.4 |
| 1.0 | Volatile (Hedge) | 17.99 | GLD:0.5, TLT:0.5 |
| 0.0 | Crisis (Defensive) | 27.35 | BIL:1.0 |

## 3. 기술적 총평
1. **SRL의 유효성**: 금리차(Spread)와 달러 모멘텀을 결합한 상태 학습이 단순 VIX 필터보다 **하락장 방어**에서 더 정교한 트리거를 발생시킴.
2. **Persistence(지속성)**: Jump Penalty 적용으로 인해 시장의 노이즈에 휘둘리지 않고 묵직하게 체제를 유지하는 '안정적 진화' 확인.
3. **결론**: MAMA Lite는 정적 자산 배분에 '지능형 나침반'을 단 것과 같으며, 특히 MDD 방어 측면에서 압도적인 효율을 보여줌.
