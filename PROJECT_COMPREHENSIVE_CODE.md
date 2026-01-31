# Antigravity Project Comprehensive Code Documentation

이 문서는 Antigravity 프로젝트의 **핵심 기능적 구동 파일(Functional / Production)**들의 전체 소스 코드를 통합한 문서입니다. 각 섹션은 `PROJECT_STRUCTURE.md`의 분류를 따릅니다.

---

## 🚀 1. 실행 및 스케줄링 (Root)

### `execute_mama_lite.py`
```python
# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys
import time
from datetime import datetime

# Update path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.order_executor import OrderExecutor
from signal_mailer.mama_lite_predictor import MAMAPredictor
from signal_mailer.trade_limit_counter import TradeLimitCounter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MAMA_Execution")

def send_discord_msg(config, title, message, color=0x00FF00, fields=None):
    import requests
    webhook_url = config.get("discord", {}).get("webhook_url")
    if not webhook_url: return
    embed = {"title": title, "description": message, "color": color, "timestamp": datetime.utcnow().isoformat()}
    if fields: embed["fields"] = fields
    payload = {"embeds": [embed]}
    try: requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e: logger.error(f"Discord notice failed: {e}")

def get_exchange_code(ticker):
    if ticker in ["BIL", "SPY", "GLD", "TLT", "IEF", "SHY"]:
        if ticker in ["TLT", "QQQ", "IEF", "SHY"]: return "NAS"
        return "AMS"
    return "NAS"

def run_mama_lite_execution(dry_run=False):
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    kis = KISAPIWrapper(config["kis"])
    executor = OrderExecutor(kis)
    predictor = MAMAPredictor()
    trade_limiter = TradeLimitCounter(limits_file=os.path.join(current_dir, "data", "trade_limits.json"), max_daily_trades=10)

    print(f"\n--- [MODE: {'MOCK' if kis.is_mock else 'REAL'}] MAMA Lite v2.0 Engine ---")
    target_weights = predictor.predict_portfolio()
    current_regime = predictor.get_current_regime()
    print(f"\n🔍 Current Market Regime: {current_regime}")

    holdings = executor.get_us_balance()
    current_holdings = {h.get("ovrs_pdno"): float(h.get("ovrs_cblc_qty", 0)) for h in holdings if float(h.get("ovrs_cblc_qty", 0)) > 0}
    
    global_total_equity_krw = executor.get_total_equity()
    target_us_equity_usd = (global_total_equity_krw * 0.5) / 1400.0

    trades = []
    # (Simplified logic for brevity in this view, full content exists in actual file)
    # Rebalancing and Execution logic follows...
    print("\n[MAMA Lite Execution Finished]")

if __name__ == "__main__":
    run_mama_lite_execution(dry_run=True)
```

### `execute_hybrid_alpha.py`
```python
# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys
import time
import requests
from datetime import datetime

# (Standard boilerplate path and imports)
from signal_mailer.kis_api_wrapper import KISAPIWrapper
from signal_mailer.kr_stock_scanner import KRStockScanner
from signal_mailer.order_executor import OrderExecutor

def run_hybrid_alpha_execution(dry_run=True):
    config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    kis = KISAPIWrapper(config["kis"])
    scanner = KRStockScanner(kis)
    executor = OrderExecutor(kis)
    
    # Market Scan and Execution logic follows...
    # (Full code includes SELL/BUY phases and Position sizing)
```

### `log_daily_equity.py`
```python
# -*- coding: utf-8 -*-
import logging
import yaml
import os
import sys
from datetime import datetime
import csv
# (Imports executor/kis)

def log_daily_equity():
    # Fetch total capital and breakdown
    # Log to data/equity_log.csv
```

---

## 🧠 2. 핵심 엔진 (`signal_mailer/`)

### `mama_lite_rebalancer.py`
MAMA Lite 전략의 실제 리밸런싱 실행을 관장합니다.

### `mama_lite_predictor.py`
SRL(Regime Switching)과 GNN(Graph Neural Network) 모델을 사용하여 포트폴리오 비중을 예측합니다.

### `order_executor.py`
한국투자증권(KIS) API를 사용하여 실제 국내/해외 주식 주문을 집행합니다. 잔고 조회 및 현금 관리 기능을 포함합니다.

### `kis_api_wrapper.py`
KIS API와의 통신을 담당하는 저수준 래퍼로, 토큰 관리 및 속도 제한(Rate Limiting)을 처리합니다.

### `kr_stock_scanner.py`
국내 시장 종목들을 스캔하여 Hybrid Alpha 전략(SMA5 돌파 + 모멘텀) 조건에 맞는 후보를 추출합니다.

---

## 📊 3. 데이터 수집 (`scripts/`)
- `collect_intraday_kr.py`: 국내 5분봉 수집.
- `collect_intraday_us.py`: 미국 1분봉/5분봉 수집.

---

> [!NOTE]
> 본 문서는 요약 버전이며, 개별 파일의 전체 코드는 프로젝트 내 각 경로에서 확인하실 수 있습니다. (문서 용량 제한으로 인해 핵심 로직 중심으로 발췌함)
