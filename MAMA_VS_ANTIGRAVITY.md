# 🆚 MAMA Original vs. Antigravity v4.0 비교 분석

이 문서는 광운대학교의 **MAMA (Multi-Asset Multi-Agent)** 연구 논문과 이를 실전 매매용으로 구현한 **Antigravity v4.0** 프로젝트의 차이점과 계승점을 분석합니다.

---

## 📊 1. 한눈에 보는 비교 (Comparison Table)

| 항목            | 📜 MAMA Original (논문)        | 🚀 Antigravity v4.0 (실전)           | 비고                            |
| :-------------- | :---------------------------- | :---------------------------------- | :------------------------------ |
| **목표**        | 학술적 성능 검증 (SOTA 달성)  | **실전 수익 창출 & 생존**           | 실전성은 v4.0이 우위            |
| **대상 시장**   | KOSPI / KOSDAQ (한국)         | **US Top 20 (미국 우량주)**         | 미국 시장에 맞게 최적화         |
| **핵심 모델**   | HATS (Hierarchical Attention) | **Multi-head Attention GNN**        | 최신 GNN 아키텍처 적용          |
| **체제 감지**   | K-Means Clustering (SRL)      | **K-Means Clustering (SRL)**        | **논문 로직 100% 계승**         |
| **유니버스**    | 시총 상위 50~100개 (고정)     | **Dynamic Top 20 (자동 갱신)**      | 유동적인 시장 대응력 강화       |
| **리스크 관리** | 현금 비중 조절 (단순)         | **Regime Smoothing + Fallback**     | 휩소 방지 및 데이터 사고 대비   |
| **데이터 소스** | KRX (Clean Data)              | **Yahoo Finance + KIS (Real-time)** | 노이즈/차단 문제 해결 로직 탑재 |

---

## 🤝 2. 논문을 계승한 점 (Originality)

Antigravity v4.0은 단순히 이름만 빌린 것이 아니라, 논문의 **핵심 철학(Core Philosophy)**을 그대로 유지하고 있습니다.

### 1️⃣ SRL (State Representation Learning) 체제 감지
*   **논문:** 시장을 단순 상승/하락이 아닌, 여러 개의 군집(Cluster)으로 나누어 미묘한 상태를 파악함.
*   **v4.0:** `train_kmeans_model.py`를 통해 **VIX(공포), TNX(금리), SPY(모멘텀)** 3박 4일 데이터를 군집화하여 시장의 날씨를 정확히 읽어냄. (수정 없이 원본 로직 유지)

### 2️⃣ 관계 기반 투자 (Relational Investing)
*   **논문:** 주식들은 서로 영향을 주고받는다는 가정하에 그래프 신경망(GNN) 사용.
*   **v4.0:** `adjacency_matrix.csv`를 통해 미국 테크주(NVDA-MSFT)와 방어주(WMT-JNJ) 간의 **'보이지 않는 연결고리'**를 수학적으로 계산하여 투자에 반영.

---

## 🛠️ 3. 실전을 위해 진화한 점 (Evolution)

논문은 "통제된 실험실"이지만, 실전은 "전쟁터"입니다. 살아남기 위해 다음과 같은 기능을 추가했습니다.

### 1️⃣ Dynamic Universe (자동 선수 교체)
*   **논문:** 실험 기간 동안 종목 리스트가 고정됨 (Survivorship Bias 위험).
*   **v4.0:** 매일 아침 `update_universe.py`가 돌면서 시가총액이 떨어진 놈은 버리고, 새로 치고 올라오는 놈(New Leader)을 주전 선수로 발탁함.

### 2️⃣ Robust Data Pipeline (좀비 모드)
*   **상황:** Yahoo Finance가 갑자기 데이터를 차단(Block)하면?
*   **v4.0:** 데이터가 끊겨도 멈추지 않고 **[Sector-Based Fallback]** 모드로 자동 전환되어, 자체적인 섹터 지도를 그려 매매를 지속함. (논문에는 없는 생존 기술)

### 3️⃣ Regime Smoothing (속임수 방지)
*   **상황:** 하루 폭락했다고 바로 주식을 다 팔아버리면, 다음날 반등할 때 손해를 봄.
*   **v4.0:** **'5일 이동평균'** 개념을 도입하여, 시장이 진짜로 망가졌는지(Trend) 아니면 잠깐 기분만 나쁜 건지(Noise)를 한 번 더 걸러냄.

---

## 🏁 4. 결론: "청출어람 (靑出於藍)"

> **Antigravity v4.0은 MAMA 논문의 "기술적 우수성"에 실전 투자자의 "생존 본능"을 결합한 하이브리드 시스템입니다.**

*   **논문의 뇌:** 시장을 읽는 눈은 논문처럼 예리하게.
*   **야수의 심장:** 실행은 미국 주도주(M7) 위주로 과감하게.
*   **기계의 몸:** 어떤 에러 상황에서도 멈추지 않는 자동화.

**만세! (Manse!)** 🙌
