# Antigravity Project Structure Guide

이 문서는 Antigravity 프로젝트의 파일 구성을 기능적(Functional) 목적과 연구적(Research) 목적으로 구분하여 설명합니다.

## 📁 1. 기능적 구동 파일 (Functional / Production)
실제 트레이딩 루프, 데이터 수집, 포트폴리오 관리를 담당하는 핵심 파일들입니다.

### 🚀 실행 및 스케줄링 (Root)
- `execute_mama_lite.py`: MAMA Lite 전략 실행 진입점.
- `execute_hybrid_alpha.py`: Hybrid Alpha (한국 주식) 전략 실행 진입점.
- `log_daily_equity.py`: 매일 계좌 잔고 및 수익률 기록.
- `weekly_performance_report.py`: 주간 성과 보고서 생성 및 알림.

### 🧠 핵심 엔진 (`signal_mailer/`)
- `mama_lite_rebalancer.py`: MAMA Lite 리밸런싱 로직 총괄.
- `mama_lite_predictor.py`: GNN/SRL 기반 시장 예측 및 종목 선정.
- `order_executor.py`: KIS API를 통한 실제 주문 집행 및 잔고 조회.
- `kis_api_wrapper.py`: KIS API 통신 및 인증 토큰 관리.
- `portfolio_manager.py`: 현재 보유량 및 대상 비중 관리.
- `signal_detector.py`: 매수/매도 시그널 감지 엔진.

### 📊 데이터 수집 (`scripts/`)
- `collect_intraday_kr.py`: 한국 주식 당일 5분봉 수집.
- `collect_intraday_us.py`: 미국 주식 당일 1분봉/5분봉 수집.

---

## 🧪 2. 연구 및 비구동 파일 (Research / Diagnostics)
전략 개발, 데이터 검증, 일회성 분석을 위한 파일들로, 일상적인 트레이딩 루프에는 포함되지 않습니다.

### 🔬 진단 및 유틸리티 (`research/`)
- `probe_*.py`: KIS API 엔드포인트 기능 확인용 (예: `probe_us_10digit.py`).
- `diagnostic_*.py`: 계좌 번호 형식, 상품 코드 등 동기화 오류 진단.
- `test_*.py`: 개별 모듈 기능 테스트 (예: `test_executor_cash.py`).

### 📈 시뮬레이션 및 백테스트 (`research/simulations/`)
- `*simulator.py`: 몬테카를로, 세금 비용, 리밸런싱 주기 등 시나리오 시뮬레이션.
- `*turbo.py`: 고속 시뮬레이션 및 엔진 최적화 테스트 버전.
- `mama_mc_simulator.py`: MAMA Lite 전략 20년 몬테카를로 시뮬레이션.

### � 전략 분석 및 모델링 (`research/analysis/`)
- `gnn_model_trainer.py`, `momentum_persistence_analysis.py` 등 데이터 분석 스크립트.
- `portfolio_spy_correlation.py`, `portfolio_vt_overlap.py` 등 포트폴리오 최적화 관련.

### 📓 리포트 및 결과물
- `research/reports/`: 전략별 분석 마크다운 보고서 (`vt_similarity_report.md` 등).
- `research/results/`: 시뮬레이션 결과 차트 및 텍스트 로그.
- `docs/reports/strategy/`: MAMA Lite 시스템 분석 및 개선 평가서.

---

## 📅 3. 데이터 및 기록 (`data/`, `logs/`)
- `data/`: 유니버스 정보(`kr_universe.json`), 수집된 파케이 데이터, GNN 학습 데이터 보관.
- `logs/`: 리밸런싱 실행 로그 및 시스템 에러 로그 기록.
- `archive/`: 과거 버전의 코드 및 더 이상 사용하지 않는 스크립트 백업.

## 📝 4. 환경 설정
- `config.yaml`: API 키, 계좌 번호, 전략 파라미터 등 전역 설정 (가장 중요).
- `.env`: 민감 정보 및 경로 설정 (로컬 환경 변수).
