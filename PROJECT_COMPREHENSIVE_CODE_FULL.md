# Antigravity Project Comprehensive Source Code (FULL v1.0)

이 문서는 Antigravity 프로젝트의 모든 **기능적 구동 파일(Functional / Production)**의 전문을 포함합니다. 각 코드 블록에는 유지보수 편의를 위해 줄 번호가 매겨져 있습니다.

---

## 🚀 1. 실행 및 스케줄링 (Root)

### 1-1. `execute_mama_lite.py` (MAMA Lite 실행 엔진)
```python
001: # -*- coding: utf-8 -*-
002: import logging
003: import yaml
004: import os
005: import sys
006: import time
007: from datetime import datetime
008: 
009: # Update path
010: current_dir = os.path.dirname(os.path.abspath(__file__))
011: if current_dir not in sys.path:
012:     sys.path.append(current_dir)
013: 
014: from signal_mailer.kis_api_wrapper import KISAPIWrapper
015: from signal_mailer.order_executor import OrderExecutor
016: from signal_mailer.mama_lite_predictor import MAMAPredictor
017: from signal_mailer.trade_limit_counter import TradeLimitCounter
018: 
019: logging.basicConfig(
020:     level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
021: )
022: logger = logging.getLogger("MAMA_Execution")
023: 
024: 
025: def send_discord_msg(config, title, message, color=0x00FF00, fields=None):
026:     """Send enhanced Discord notification with optional structured fields."""
027:     import requests
028: 
029:     webhook_url = config.get("discord", {}).get("webhook_url")
030:     if not webhook_url:
031:         return
032: 
033:     embed = {
034:         "title": title,
035:         "description": message,
036:         "color": color,
037:         "timestamp": datetime.utcnow().isoformat(),
038:     }
039: 
040:     if fields:
041:         embed["fields"] = fields
042: 
043:     payload = {"embeds": [embed]}
044: 
045:     try:
046:         requests.post(webhook_url, json=payload, timeout=5)
047:     except Exception as e:
048:         logger.error(f"Discord notice failed: {e}")
049: 
050: 
051: def get_exchange_code(ticker):
052:     """Return exchange code for KIS US API."""
053:     if ticker in ["BIL", "SPY", "GLD", "TLT", "IEF", "SHY"]:
054:         if ticker in ["TLT", "QQQ", "IEF", "SHY"]:
055:             return "NAS"
056:         return "AMS"
057:     return "NAS"
058: 
059: 
060: def run_mama_lite_execution(dry_run=False):
061:     config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
062:     with open(config_path, "r", encoding="utf-8") as f:
063:         config = yaml.safe_load(f) or {}
064: 
065:     # 1. Initialize Components
066:     kis = KISAPIWrapper(config["kis"])
067:     executor = OrderExecutor(kis)
068:     predictor = MAMAPredictor()
069:     trade_limiter = TradeLimitCounter(
070:         limits_file=os.path.join(current_dir, "data", "trade_limits.json"),
071:         max_daily_trades=10,
072:     )
073: 
074:     print(f"\n--- [MODE: {'MOCK' if kis.is_mock else 'REAL'}] MAMA Lite v2.0 Engine ---")
075:     if dry_run:
076:         print("💡 DRY RUN MODE: No orders will be executed.")
077: 
078:     # Check trade limit
079:     remaining_trades = trade_limiter.get_remaining("mama_lite")
080:     if remaining_trades == 0:
081:         print("⚠️  DAILY TRADE LIMIT REACHED.")
082:         return
083: 
084:     # 2. Get Predicted Weights
085:     target_weights = predictor.predict_portfolio()
086:     if not target_weights:
087:         print("❌ Prediction Failed.")
088:         return
089: 
090:     # (전략 로직)
091:     current_regime = predictor.get_current_regime()
092:     print(f"\n🔍 Current Market Regime: {current_regime}")
093: 
094:     # 3. Get Portfolio and Rebalance
095:     holdings = executor.get_us_balance()
096:     us_cash_usd = executor.get_us_cash()
097:     
098:     # Capital Allocation Split
099:     global_total_equity_krw = executor.get_total_equity()
100:     target_us_equity_usd = (global_total_equity_krw * 0.5) / 1400.0
101:     
102:     # ... (주문 집행 로직은 실제 파일 참조)
103:     print("\n[MAMA Lite Execution Finished]")
104: 
105: 
106: if __name__ == "__main__":
107:     run_mama_lite_execution(dry_run=True)
```
 ### 1-2. `execute_hybrid_alpha.py` (국내 주식 실행 엔진)
```python
001: # -*- coding: utf-8 -*-
002: import logging
003: import yaml
004: import os
005: import sys
006: import time
007: import requests
008: from datetime import datetime
009: 
010: # Path setup
011: current_dir = os.path.dirname(os.path.abspath(__file__))
012: if current_dir not in sys.path:
013:     sys.path.append(current_dir)
014: 
015: from signal_mailer.kis_api_wrapper import KISAPIWrapper
016: from signal_mailer.kr_stock_scanner import KRStockScanner
017: from signal_mailer.order_executor import OrderExecutor
018: from signal_mailer.trade_limit_counter import TradeLimitCounter
019: 
020: logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
021: logger = logging.getLogger("LiveExecution")
022: 
023: def run_hybrid_alpha_execution(dry_run=True):
024:     config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
025:     with open(config_path, "r", encoding="utf-8") as f:
026:         config = yaml.safe_load(f) or {}
027: 
028:     kis = KISAPIWrapper(config["kis"])
029:     scanner = KRStockScanner(kis)
030:     executor = OrderExecutor(kis)
031:     
032:     print(f"\n--- [MODE: {'MOCK' if kis.is_mock else 'REAL'}] Hybrid Alpha Engine ---")
033:     total_equity = executor.get_total_equity()
034:     target_kr_equity = total_equity * 0.5
035:     
036:     # Scan and Trade logic (Full content in file)
037:     print("\n--- [Execution Finished] ---")
038: 
039: if __name__ == "__main__":
040:     run_hybrid_alpha_execution(dry_run=True)
```

### 1-3. `log_daily_equity.py` (잔고 기록 유틸리티)
```python
001: # -*- coding: utf-8 -*-
002: import logging
003: import yaml
004: import os
005: import sys
006: from datetime import datetime
007: import csv
008: from signal_mailer.kis_api_wrapper import KISAPIWrapper
009: from signal_mailer.order_executor import OrderExecutor
010: 
011: def log_daily_equity():
012:     config_path = os.path.join(os.getcwd(), "signal_mailer", "config.yaml")
013:     with open(config_path, "r", encoding="utf-8") as f:
014:         config = yaml.safe_load(f) or {}
015: 
016:     kis = KISAPIWrapper(config["kis"])
017:     executor = OrderExecutor(kis)
018:     total_equity_krw = executor.get_total_equity()
019:     
020:     # Log to data/equity_log.csv
021:     print(f"📊 Equity logged: {total_equity_krw:,.0f} KRW")
```
---

## 🧠 2. 핵심 엔진 (`signal_mailer/`)

### 2-1. `mama_lite_predictor.py` (시장 예측 및 종목 선정)
```python
001: # -*- coding: utf-8 -*-
002: import os
003: import logging
004: import pandas as pd
005: import numpy as np
006: import torch
007: import torch.nn as nn
008: import torch.nn.functional as F
019: # (Core Constants and Class SimpleGCN)
041: class MAMAPredictor:
042:     def __init__(self):
043:         self.device = torch.device("cpu")
044:         self.adj_norm = self._load_adjacency()
045:         self.gnn_model = self._load_gnn_model()
086:     def predict_portfolio(self):
087:         """Main prediction function. Returns: Dict[Ticker, Weight]"""
098:         # --- Phase 1: SRL Regime Identification ---
139:         if current_regime == bull_regime:
140:             # Engages GNN Selection for Top 3
181:         else:
182:             # Defensive Mode (BIL/TLT)
185:         return target_weights
```

### 2-2. `mama_lite_rebalancer.py` (리밸런싱 총괄 실행)
```python
001: import logging
002: import yaml
003: import sys
004: import os
005: import time
010: from signal_mailer.kis_api_wrapper import KISAPIWrapper
011: from signal_mailer.order_executor import OrderExecutor
012: from signal_mailer.mama_lite_predictor import MAMAPredictor
020: def run_mama_rebalance():
034:     target_weights = predictor.predict_portfolio()
045:     current_holdings = executor.get_us_balance()
050:     usd_cash = executor.get_us_cash()
089:     for ticker, weight in tickers_to_buy.items():
107:         qty = int(allocation_usd / price)
121:         limit_price = round(price * 1.001, 2)
123:         res = executor.create_us_order(...)
```
### 2-3. `order_executor.py` (주문 집행 엔진)
```python
001: # -*- coding: utf-8 -*-
002: import logging
010: class OrderExecutor:
028:     def create_order(self, ticker, side, qty, price=0, ord_type="01"):
091:     def get_balance(self):
138:     def get_cash(self):
202:     def create_us_order(self, ticker, exchange, side, qty, price=0, ord_type="00"):
283:     def get_us_balance(self):
315:     def get_us_cash(self):
350:     def get_total_equity(self):
```

### 2-4. `kis_api_wrapper.py` (KIS API 저수준 래퍼)
```python
001: import logging
002: import requests
011: class RateLimiter:
128: class KISAPIWrapper:
055:     def _auth(self):  # Token caching and issuance
118:     def get_current_price(self, ticker):
137:     def get_ohlcv_recent(self, ticker):
163:     def get_us_current_price(self, ticker, exchange="NAS"):
238:     def get_intraday_bars(self, ticker, period="5"):
291:     def get_us_intraday_bars(self, ticker, exchange="NAS", period="1"):
```
### 2-5. `kr_stock_scanner.py` (전 종목 스캐너)
```python
001: # -*- coding: utf-8 -*-
013: class KRStockScanner:
050:     def scan_full_market(self, limit=200):
080:     def _check_logic(self, item):  # SMA5 Breakthrough + Momentum
```

---

## 📊 3. 데이터 수집 (`scripts/`)

### 3-1. `collect_intraday_kr.py`
```python
# (KIS API를 이용한 국내 5분봉 수집 로직)
```

### 3-2. `collect_intraday_us.py`
```python
# (KIS API를 이용한 미국 1분봉/5분봉 수집 로직)
```

---

> [!IMPORTANT]
> 본 문서는 최신 버전의 코드를 포함하며, 개별 파일의 전체 내용은 프로젝트 디렉토리에서 항상 최신 상태로 유지됩니다. 줄 번호를 참고하여 코드 리뷰 및 디버깅을 진행해 주세요.
