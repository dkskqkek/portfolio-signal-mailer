diff --git a/signal_mailer/README.md b/signal_mailer/README.md
index b891cfc..09da6fe 100644
--- a/signal_mailer/README.md
+++ b/signal_mailer/README.md
@@ -46,28 +46,28 @@ python main.py
 
 (스케줄러 시작됨. Ctrl+C로 종료)
 
-## 동작 방식
+## 동작 방식 (QLD Core + Top-3 Ensemble)
 
 1. 매일 아침 설정된 시간에 자동 실행
-2. SPY의 20일 이동평균과 변동성 분석
-3. 위험신호 판단:
-   - 20일 MA < 25 percentile 또는
-   - 20일 변동성 > 75 percentile
-4. 신호 상태 변화가 있을 때만 메일 발송
-5. 모든 신호 이력은 signal_history.json에 저장
-6. 실행 로그는 mailer.log에 저장
-
-## 신호 해석
-
-[신호] NORMAL (QQQ 유지)
-- 정상 상태
-- 포지션 조정 불필요
-- SCHD 34% + QQQ 34% + XLP 0% + KOSPI 17% + GOLD 15%
-
-[신호] DANGER (QQQ->XLP 전환)
-- 위험신호 발생
-- QQQ를 XLP로 전환 권장
-- SCHD 34% + QQQ 0% + XLP 34% + KOSPI 17% + GOLD 15%
+2. QQQ의 110일 및 250일 단순 이동평균(SMA) 분석
+3. 위험신호 판단 (Hysteresis 적용):
+   - **NORMAL 진입**: QQQ 가격이 110일선과 250일선을 모두 상회할 때
+   - **DANGER 진입**: QQQ 가격이 110일선과 250일선을 모두 하회할 때
+   - **유지 (Stay)**: 그 외 구간(완충 지대)에서는 이전 상태를 유지하여 잦은 매매 방지
+4. 하락장 대응 (Top-3 Defensive Ensemble):
+   - 23종의 순수 1배물 방어 자산 중 8개월(168일) 상대 모멘텀 상위 3종 균등 배분
+   - 개별 자산의 모멘텀이 하락세( < 0)일 경우 해당 비중은 BIL(현금)로 대피
+5. 상태 변화 여부와 무관하게 매일 아침 데일리 리포트 발송
+6. 모든 신호 이력은 `signal_history.json`에 저장
+
+[신호] 🟢 NORMAL (CORE HOLDING)
+- 정상 상태: QQQ의 중장기 추세가 우호적임
+- 포트폴리오 비중: **QLD(45%)** + KOSPI(20%) + SPY(20%) + GOLD(15%)
+
+[신호] 🔴 DANGER (DEFENSIVE SWITCH)
+- 위험 상태: QQQ가 주요 이평선을 모두 하회하여 하락장 확정
+- 포트폴리오 비중: **최적 방어 자산 3종(각 15%)** + KOSPI(20%) + SPY(20%) + GOLD(15%)
+- 방어 자산 후보: 23종 슈퍼 앙상블 (BTAL, PFIX, DBMF, UUP, GLD, BIL 등)
 
 ## 다른 이메일 서비스 설정
 
