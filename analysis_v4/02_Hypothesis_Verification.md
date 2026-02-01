
# Antigravity V4: 가설 검증 보고서 (Hypothesis Verification)

**작성일:** 2026-02-01 (수정본)
**작성자:** Antigravity AI
**목적:** V4 전략 고도화를 위한 3가지 핵심 가설의 **데이터 기반 팩트 체크**.
**비고:** 초기 보고서의 수치 오류를 수정하고, CSV 원본 데이터에 기반하여 재작성함.

---

## 1. ETF Tournament (VTI 대안 찾기)
> **가설:** "시장 전체(VTI)보다 V4 전략(추세 추종 + 공포 매도)에 더 적합한 고수익/고변동성 자산이 있을 것이다."

### 1.1 테스트 대상 (10종)
*   **Broad:** VTI, SPY, DIA, RSP, IWM
*   **Growth:** SCHG, QQQ, XLK
*   **Leverage:** QLD (2x), SSO (2x)

### 1.2 검증 결과 (Sortino 기준 정렬)
| 순위  | 티커     | 전략 CAGR  | 전략 MDD    | Sortino  | 비고                      |
| :---- | :------- | :--------- | :---------- | :------- | :------------------------ |
| **1** | **SCHG** | **15.88%** | **-22.67%** | **1.35** | **압도적 1위 (Winner)** 👑 |
| 2     | SPY      | 10.78%     | -25.32%     | 1.09     | Baseline                  |
| 3     | VTI      | 10.15%     | -27.70%     | 1.09     | 기존 안                   |
| 4     | QLD      | 22.36%     | -51.04%     | 1.01     | 수익 1위, MDD 위험        |

### 1.3 결론
*   **성장주(Growth)와의 궁합:** V4의 "하락장 회피(Cash 100%)" 로직은 성장주의 최대 약점(깊은 MDD)을 완벽하게 보완함.
*   **Action:** 메인 포트폴리오 자산을 **VTI → SCHG**로 교체. (수익 1.5배 증가, 위험 감소)
*   **Data:** `data/backtest_results/etf_tournament_results.csv`

---

## 2. Hybrid Signal Strategy (Correction)
> **가설:** "변동성이 큰 SCHG의 자체 신호보다, 묵직한 VTI의 시장 신호를 따르는 것이 휩소(Whipsaw)를 줄일 것이다."

### 2.1 검증 결과 (2010~2025)
| 전략                    | CAGR       | MDD         | Sortino   | 비교                            |
| :---------------------- | :--------- | :---------- | :-------- | :------------------------------ |
| **SCHG 독자 (Self)**    | **15.84%** | -22.67%     | 1.348     | **수익 우위 (+0.2%p)**          |
| **Hybrid (VTI Signal)** | 15.64%     | **-21.18%** | **1.365** | **안정성 우위 (MDD 1.5% 개선)** |

### 2.2 재평가 및 결론
*   **Correction:** Hybrid 전략이 수익률(CAGR) 면에서는 SCHG 독자 전략보다 **소폭 낮음**. "압도적 승리"라는 표현은 오류였음.
*   **Trade-off:** 수익률 0.2%를 희생하고, MDD를 1.5% 줄이는 선택. Sortino Ratio(위험 대비 수익)는 Hybrid가 미세하게 높음(1.36 vs 1.35).
*   **Action:** **Hybrid 채택 유지.** (이유: 개별 기술주 발작보다 시장 전체 리스크에 반응하는 것이 장기적으로 심리적 안정감을 주기 때문. 수익률 차이도 미미함.)
*   **Data:** `data/backtest_results/hybrid_signal_test.csv`

---

## 3. Yield Inversion Confirmation Lag (Correction)
> **가설:** "장단기 금리 역전 후 3개월(Stage 2)을 기다렸다가 대응하면 거짓 신호를 거르고 수익을 보존할 것이다." (Dividend.com 아이디어)

### 3.1 검증 결과 (SPY 1993~2025)
| 전략                 | CAGR       | MDD         | Sharpe   | 비교            |
| :------------------- | :--------- | :---------- | :------- | :-------------- |
| **Immediate (즉시)** | **10.79%** | -25.32%     | **0.86** | **미세한 우위** |
| 3-Month Lag          | 10.73%     | **-25.30%** | 0.86     | 차이 거의 없음  |

### 3.2 재평가 및 결론
*   **Correction:** 초기 보고서에서 언급된 "-28.90% (High Risk)"는 잘못된 수치(Hallucination)였음. 실제로는 **두 전략의 성과가 거의 동일함.**
*   **No Edge:** 3개월을 기다리는 복잡한 로직을 추가해도, 수익(-0.06%p)과 MDD(+0.02%p) 모두 개선되지 않음.
*   **Action:** 해당 전략 **기각(Reject)**.
    *   **이유:** "위험해서"가 아니라, **"차이가 없어서(No Benefit)"** 단순한 기존 로직(즉시 대응)을 유지함. 복잡성은 오류의 원인이 됨.
*   **Data:** `data/backtest_results/inversion_lag_test.csv`

---

## 4. 최종 확정 전략 (Antigravity V4 Final)
1.  **Universe:** **SCHG** (Growth ETF)
2.  **Signal:** **VTI** MA185 + 3% Buffer (Hybrid)
    *   이유: 수익률 0.2%를 포기하고 MDD 1.5%를 방어하는 보수적 선택.
3.  **Crisis Logic:** Yield Inversion 발생 **즉시** Cash 100%
    *   이유: 3개월 대기 로직이 실익(Edge)이 없으므로 단순함(Simplicity)을 선택.
