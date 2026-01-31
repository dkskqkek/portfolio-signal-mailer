# 📄 Antigravity Project Absolute Full Source Code (v3.0)

이 문서는 프로젝트의 모든 **기능적 구동 파일**의 전문을 담고 있습니다. 어떠한 생략이나 요약 없이, 실제 소스 코드의 모든 라인을 파일별로 줄 번호를 매겨 수록합니다.

---

## 🚀 1. `execute_mama_lite.py` (333 lines)
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
026:     """Send enhanced Discord notification with optional structured fields.
027: 
028:     Args:
029:         config: Configuration dictionary with Discord webhook URL
030:         title: Notification title
031:         message: Main message body
032:         color: Embed color (default: green)
033:         fields: Optional list of {"name": str, "value": str, "inline": bool} dicts
034:     """
035:     import requests
036: 
037:     webhook_url = config.get("discord", {}).get("webhook_url")
038:     if not webhook_url:
039:         return
040: 
041:     embed = {
042:         "title": title,
043:         "description": message,
044:         "color": color,
045:         "timestamp": datetime.utcnow().isoformat(),
046:     }
047: 
048:     if fields:
049:         embed["fields"] = fields
050: 
051:     payload = {"embeds": [embed]}
052: 
053:     try:
054:         requests.post(webhook_url, json=payload, timeout=5)
055:     except Exception as e:
056:         logger.error(f"Discord notice failed: {e}")
057: 
058: 
059: def get_exchange_code(ticker):
060:     """
061:     Return exchange code for KIS US API.
062:     Ref: NAS(Nasdaq), NYS(NYSE), AMS(Amec/Arca)
063:     """
064:     # MAMA Lite Universe
065:     if ticker in ["BIL", "SPY", "GLD", "TLT", "IEF", "SHY"]:
066:         # Most major ETFs are on NYSE Arca (AMS) or Nasdaq.
067:         # SPY: Arca (AMS)
068:         # BIL: Arca (AMS)
069:         # GLD: Arca (AMS)
070:         # TLT: Nasdaq (NAS)
071:         # QQQ: Nasdaq (NAS)
072:         if ticker in ["TLT", "QQQ", "IEF", "SHY"]:
073:             return "NAS"
074:         return "AMS"
075: 
076:     # Stocks usually Nasdaq for Tech
077:     if ticker in [
078:         "AAPL",
079:         "MSFT",
080:         "GOOGL",
081:         "AMZN",
082:         "META",
083:         "NVDA",
084:         "TSLA",
085:         "NFLX",
086:         "AVGO",
087:         "AMD",
088:         "INTC",
089:     ]:
090:         return "NAS"
091: 
092:     return "NAS"  # Default fallback
093: 
094: 
095: def run_mama_lite_execution(dry_run=False):
096:     config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
097:     with open(config_path, "r", encoding="utf-8") as f:
098:         config = yaml.safe_load(f) or {}
099: 
100:     # 1. Initialize Components
101:     kis = KISAPIWrapper(config["kis"])
102:     executor = OrderExecutor(kis)
103:     predictor = MAMAPredictor()
104:     trade_limiter = TradeLimitCounter(
105:         limits_file=os.path.join(current_dir, "data", "trade_limits.json"),
106:         max_daily_trades=10,
107:     )
108: 
109:     print(
110:         f"\n--- [MODE: {'MOCK' if kis.is_mock else 'REAL'}] MAMA Lite v2.0 Engine ---"
111:     )
112:     if dry_run:
113:         print("💡 DRY RUN MODE: No orders will be executed.")
114: 
115:     # Check trade limit
116:     remaining_trades = trade_limiter.get_remaining("mama_lite")
117:     print(f"📊 Remaining trades today: {remaining_trades}/10")
118:     if remaining_trades == 0:
119:         print("⚠️  DAILY TRADE LIMIT REACHED (10 trades). No more trades allowed today.")
120:         send_discord_msg(
121:             config,
122:             "⚠️ [MAMA Lite] Trade Limit Reached",
123:             "일일 거래 한도 도달 (10건). 오늘은 더 이상 거래하지 않습니다.",
124:             color=0xFFA500,
125:         )
126:         return
127: 
128:     # 2. Get Predicted Weights
129:     target_weights = predictor.predict_portfolio()
130:     if not target_weights:
131:         print("❌ Prediction Failed. Aborting.")
132:         return
133: 
134:     print("\n🎯 Target Weights:")
135:     for t, w in target_weights.items():
136:         print(f"   - {t}: {w:.1%}")
137: 
138:     # Regime Monitoring Alert
139:     current_regime = predictor.get_current_regime()
140:     print(f"\n🔍 Current Market Regime: {current_regime}")
141: 
142:     if current_regime in ["Bear", "Crisis"]:
143:         regime_msg = f"⚠️ MAMA Lite가 **{current_regime}** 체제를 감지했습니다.\n\n"
144:         if current_regime == "Crisis":
145:             regime_msg += "🚨 **위험**: 극단적 변동성 구간\n"
146:             regime_msg += "📊 방어 자산(BIL/TLT) 비중 증가 중"
147:         else:
148:             regime_msg += "📉 하락 추세 감지\n"
149:             regime_msg += "🛡️ 방어 모드 진입 중"
150: 
151:         send_discord_msg(
152:             config,
153:             f"⚠️ [MAMA Lite] {current_regime} Regime Detected",
154:             regime_msg,
155:             color=0xFF6B6B if current_regime == "Crisis" else 0xFFA500,
156:         )
157: 
158:     # 3. Get Current Portfolio (US)
159:     holdings = executor.get_us_balance()
160:     # Convert holdings to usable format: {Ticker: Qty}
161:     current_holdings = {}
162:     current_total_val_usd = 0.0
163: 
164:     print("\n📊 Current US Holdings:")
165:     for h in holdings:
166:         ticker = h.get("ovrs_pdno", "")
167:         qty = float(h.get("ovrs_cblc_qty", 0))
168:         val_usd = float(h.get("frcr_evlu_amt2", 0))
169: 
170:         if qty > 0:
171:             current_holdings[ticker] = qty
172:             current_total_val_usd += val_usd
173:             print(f"   - {ticker}: {qty} shares (${val_usd:,.2f})")
174: 
175:     # Global Asset Management: 50/50 Capital Split
176:     global_total_equity_krw = executor.get_total_equity()
177:     target_us_equity_krw = global_total_equity_krw * 0.5
178: 
179:     # Fixed exchange rate for display logic (1400)
180:     exch_rate = 1400.0
181:     target_us_equity_usd = target_us_equity_krw / exch_rate
182: 
183:     us_cash_usd = executor.get_us_cash()
184: 
185:     print(f"\n🌍 Global Total Equity: {global_total_equity_krw:,.0f}원")
186:     print(
187:         f"🎯 Target US Allocation (50%): ${target_us_equity_usd:,.2f} ({target_us_equity_krw:,.0f}원)"
188:     )
189:     print(f"💰 Available US Cash: ${us_cash_usd:,.2f}")
190: 
191:     # Use Target US Allocation as the base for rebalancing
192:     total_equity = target_us_equity_usd
193: 
194:     # 4. Rebalancing Logic
195:     trades = []
196: 
197:     # Sell Logic: Tickers in Current but not in Target (or weight reduction)
198:     for t, qty in current_holdings.items():
199:         target_w = target_weights.get(t, 0.0)
200: 
201:         # Get Current Price
202:         xc = get_exchange_code(t)
203:         curr_price = kis.get_us_current_price(t, exchange=xc)
204: 
205:         # Retry with alternate if failed
206:         if not curr_price and xc == "AMS":
207:             curr_price = kis.get_us_current_price(t, exchange="NYS")
208:         elif not curr_price and xc == "NYS":
209:             curr_price = kis.get_us_current_price(t, exchange="AMS")
210: 
211:         if not curr_price:
212:             print(f"⚠️ Could not get price for {t} ({xc}), skipping...")
213:             continue
214: 
215:         current_val = qty * curr_price
216:         target_val = total_equity * target_w
217: 
218:         diff_val = target_val - current_val
219: 
220:         # If diff is negative significantly, SELL
221:         if diff_val < -100:  # Sell threshold $100
222:             sell_amt_usd = abs(diff_val)
223:             sell_qty = int(sell_amt_usd / curr_price)
224:             if sell_qty > 0:
225:                 trades.append(
226:                     {
227:                         "ticker": t,
228:                         "side": "SELL",
229:                         "qty": sell_qty,
230:                         "price": curr_price,
231:                         "reason": "Rebalance",
232:                     }
233:                 )
234: 
235:     # Buy Logic
236:     for t, w in target_weights.items():
237:         # Get Current Price
238:         xc = get_exchange_code(t)
239:         curr_price = kis.get_us_current_price(t, exchange=xc)
240: 
241:         # Retry logic
242:         if not curr_price and xc == "AMS":
243:             curr_price = kis.get_us_current_price(t, exchange="NYS")
244:         elif not curr_price and xc == "NYS":
245:             curr_price = kis.get_us_current_price(t, exchange="AMS")
246: 
247:         if not curr_price:
248:             print(f"⚠️ Could not get price for {t} ({xc}), skipping...")
249:             continue
250: 
251:         current_qty = current_holdings.get(t, 0)
252:         current_val = current_qty * curr_price
253:         target_val = total_equity * w
254: 
255:         diff_val = target_val - current_val
256: 
257:         if diff_val > 100:  # Buy threshold $100
258:             buy_amt_usd = diff_val
259:             buy_qty = int(buy_amt_usd / curr_price)
260: 
261:             # Position size limit check: max 20% of total equity per order
262:             order_value_usd = buy_qty * curr_price
263:             max_position_size_usd = total_equity * 0.20
264: 
265:             if order_value_usd > max_position_size_usd:
266:                 print(
267:                     f"   ⚠️  Order for {t} (${order_value_usd:.2f}) exceeds 20% limit (${max_position_size_usd:.2f})"
268:                 )
269:                 print("   Reducing order size to comply with risk limits...")
270:                 buy_qty = int(max_position_size_usd / curr_price)
271:                 order_value_usd = buy_qty * curr_price
272: 
273:                 if buy_qty <= 0:
274:                     print(f"   ⚠️  Cannot buy {t}: Even minimum order exceeds 20% limit")
275:                     continue
276: 
277:             if buy_qty > 0:
278:                 trades.append(
279:                     {
280:                         "ticker": t,
281:                         "side": "BUY",
282:                         "qty": buy_qty,
283:                         "price": curr_price,
284:                         "reason": "Rebalance",
285:                     }
286:                 )
287: 
288:     # 5. Execution
289:     print(f"\n📝 Generated {len(trades)} Trades:")
290:     for trade in trades:
291:         print(
292:             f"   {trade['side']} {trade['ticker']} {trade['qty']} shares @ ${trade['price']:.2f}"
293:         )
294: 
295:     if not dry_run:
296:         # execute
297:         for trade in trades:
298:             xc = get_exchange_code(trade["ticker"])
299: 
300:             # Check trade limit
301:             if not trade_limiter.check_and_increment("mama_lite"):
302:                 print(
303:                     f"   ⚠️  Cannot execute {trade['side']} for {trade['ticker']}: Daily trade limit reached"
304:                 )
305:                 send_discord_msg(
306:                     config,
307:                     "⚠️ [MAMA Lite] Trade Limit Blocked",
308:                     f"{trade['ticker']} {trade['side']} 차단: 일일 거래 한도 도달",
309:                     color=0xFFA500,
310:                 )
311:                 continue
312: 
313:             res = executor.create_us_order(
314:                 ticker=trade["ticker"],
315:                 side=trade["side"],
316:                 qty=trade["qty"],
317:                 price=trade["price"],
318:                 exchange=xc,
319:                 ord_type="00",
320:             )
321: 
322:             if res:
323:                 msg = f"{trade['side']} {trade['ticker']} ({trade['qty']}sh) Executed"
324:                 send_discord_msg(config, "MAMA Lite Trade", msg)
325:                 time.sleep(0.5)
326: 
327:     print("\n[MAMA Lite Execution Finished]")
328: 
329: 
330: if __name__ == "__main__":
331:     # Default to DRY RUN for safety integration
332:     run_mama_lite_execution(dry_run=True)
333: ```
---

## 🚀 2. `execute_hybrid_alpha.py` (272 lines)
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
010: # Update path
011: current_dir = os.path.dirname(os.path.abspath(__file__))
012: if current_dir not in sys.path:
013:     sys.path.append(current_dir)
014: 
015: from signal_mailer.kis_api_wrapper import KISAPIWrapper
016: from signal_mailer.kr_stock_scanner import KRStockScanner
017: from signal_mailer.order_executor import OrderExecutor
018: from signal_mailer.trade_limit_counter import TradeLimitCounter
019: 
020: logging.basicConfig(
021:     level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
022: )
023: logger = logging.getLogger("LiveExecution")
024: 
025: 
026: def send_discord_msg(config, title, message, color=0x00FF00, fields=None):
027:     """Send enhanced Discord notification with optional structured fields.
028: 
029:     Args:
030:         config: Configuration dictionary with Discord webhook URL
031:         title: Notification title
032:         message: Main message body
033:         color: Embed color (default: green)
034:         fields: Optional list of {"name": str, "value": str, "inline": bool} dicts
035:     """
036:     webhook_url = config.get("discord", {}).get("webhook_url")
037:     if not webhook_url:
038:         return
039: 
040:     embed = {
041:         "title": title,
042:         "description": message,
043:         "color": color,
044:         "timestamp": datetime.utcnow().isoformat(),
045:     }
046: 
047:     if fields:
048:         embed["fields"] = fields
049: 
050:     payload = {"embeds": [embed]}
051: 
052:     try:
053:         requests.post(webhook_url, json=payload, timeout=5)
054:     except Exception as e:
055:         logger.error(f"Discord notice failed: {e}")
056: 
057: 
058: def run_hybrid_alpha_execution(dry_run=True):
059:     config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
060:     with open(config_path, "r", encoding="utf-8") as f:
061:         config = yaml.safe_load(f) or {}
062: 
063:     # 1. Initialize API and Scanner
064:     kis = KISAPIWrapper(config["kis"])
065:     scanner = KRStockScanner(kis)
066:     executor = OrderExecutor(kis)
067:     trade_limiter = TradeLimitCounter(
068:         limits_file=os.path.join(current_dir, "data", "trade_limits.json"),
069:         max_daily_trades=10,
070:     )
071: 
072:     print(f"\n--- [MODE: {'MOCK' if kis.is_mock else 'REAL'}] Hybrid Alpha Engine ---")
073:     if dry_run:
074:         print("💡 DRY RUN MODE: No orders will be executed.")
075: 
076:     # Check trade limit
077:     remaining_trades = trade_limiter.get_remaining("hybrid_alpha")
078:     print(f"📊 Remaining trades today: {remaining_trades}/10")
079:     if remaining_trades == 0:
080:         print("⚠️  DAILY TRADE LIMIT REACHED (10 trades). No more trades allowed today.")
081:         send_discord_msg(
082:             config,
083:             "⚠️ [Hybrid Alpha] Trade Limit Reached",
084:             "일일 거래 한도 도달 (10건). 오늘은 더 이상 거래하지 않습니다.",
085:             color=0xFFA500,
086:         )
087:         return
088: 
089:     # 2. Get Current Holdings
090:     holdings = executor.get_balance()
091:     held_tickers = {h["pdno"]: h for h in holdings}
092:     print(f"📊 Current Holdings: {len(holdings)} stocks")
093: 
094:     # 3. Market Scan
095:     print("🔍 Scanning Market for Candidates (Top 200)...")
096:     candidates = scanner.scan_full_market(limit=200)
097:     top_5 = candidates[:5]
098:     top_5_tickers = [c["ticker"] for c in top_5]
099: 
100:     # Calculate Target Allocation (Global Equity * 0.5)
101:     total_equity = executor.get_total_equity()
102:     target_kr_equity = total_equity * 0.5
103:     target_count = 5
104:     allocation_per_stock = target_kr_equity / target_count
105: 
106:     print(f"📊 Total Global Equity: {total_equity:,.0f}원")
107:     print(
108:         f"📊 Target KR Allocation (50%): {target_kr_equity:,.0f}원 (Per Stock: {allocation_per_stock:,.0f}원)"
109:     )
110: 
111:     cash = executor.get_kr_cash()
112:     print(f"💰 Available Cash (KR): {cash:,.0f}원")
113: 
114:     # Execute Strategy
115:     sell_count = 0
116:     buy_count = 0
117:     print("\n=== SELL PHASE ===")
118:     # 4. SELL Logic: Full Sell or Partial Sell for Rebalancing
119:     for ticker, h in held_tickers.items():
120:         name = h.get("prdt_name", ticker)
121:         current_qty = int(h.get("hldg_qty", 0))
122:         curr_price = float(h.get("prpr", 0))
123: 
124:         # Case A: Ticker no longer in Top 5 -> FULL SELL
125:         if ticker not in top_5_tickers:
126:             print(f"📉 [FULL SELL] {name} ({ticker}): Signal lost or ranked out.")
127:             if not dry_run:
128:                 # Check trade limit before executing
129:                 if not trade_limiter.check_and_increment("hybrid_alpha"):
130:                     print(f"   ⚠️  Cannot sell {name}: Daily trade limit reached")
131:                     send_discord_msg(
132:                         config,
133:                         "⚠️ [Hybrid Alpha] Trade Limit Blocked",
134:                         f"{name} 매도 차단: 일일 거래 한도 도달",
135:                         color=0xFFA500,
136:                     )
137:                     continue
138: 
139:                 res = executor.create_order(
140:                     ticker, side="SELL", qty=current_qty, ord_type="01"
141:                 )
142:                 if res and res.get("rt_cd") == "0":
143:                     sell_amt = current_qty * curr_price
144:                     msg = f"**{name}** ({ticker})\n수량: {current_qty}주 (전량 청산)\n단가: ~{curr_price:,}원\n매도금액: {sell_amt:,.0f}원"
145:                     print(f"   ✅ FULL SELL Success: {name} ({current_qty}주)")
146:                     send_discord_msg(
147:                         config, "📉 [Hybrid Alpha] 전량 매도", msg, color=0xFF0000
148:                     )
149:                     sell_count += 1
150:             else:
151:                 print(f"   [DRY RUN] Would sell all {current_qty} shares.")
152:                 sell_count += 1
153: 
154:         # Case B: Ticker in Top 5 but exceeds allocation -> PARTIAL SELL
155:         else:
156:             target_qty = int(allocation_per_stock // curr_price)
157:             if current_qty > target_qty * 1.1:  # Allow 10% buffer to avoid micro-trades
158:                 sell_qty = current_qty - target_qty
159:                 print(
160:                     f"📉 [PARTIAL SELL] {name} ({ticker}): Reducing weight to 50% split. {current_qty} -> {target_qty}"
161:                 )
162:                 if not dry_run:
163:                     # Check trade limit
164:                     if not trade_limiter.check_and_increment("hybrid_alpha"):
165:                         print(
166:                             f"   ⚠️  Cannot partial sell {name}: Daily trade limit reached"
167:                         )
168:                         continue
169: 
170:                     res = executor.create_order(
171:                         ticker, side="SELL", qty=sell_qty, ord_type="01"
172:                     )
173:                     if res and res.get("rt_cd") == "0":
174:                         msg = f"**{name}** ({ticker})\n수량: {current_qty}주 → {target_qty}주 (🔻{sell_qty}주 매도)\n단가: ~{curr_price:,}원\n매도금액: {sell_qty * curr_price:,.0f}원"
175:                         print(f"   ✅ PARTIAL SELL Success: {name} ({sell_qty}주)")
176:                         send_discord_msg(
177:                             config,
178:                             "📉 [Hybrid Alpha] 비중 조절 매도",
179:                             msg,
180:                             color=0xFFA500,
181:                         )
182:                         sell_count += 1
183:                 else:
184:                     print(f"   [DRY RUN] Would sell {sell_qty} shares to rebalance.")
185:                     sell_count += 1
186: 
187:     if sell_count > 0:
188:         time.sleep(1)
189: 
190:     # 5. BUY Logic: Top 5 Stocks
191:     cash = executor.get_cash()
192:     print(f"💰 Available Cash (Domestic): {cash:,}원")
193: 
194:     for stock in top_5:
195:         ticker = stock["ticker"]
196:         name = stock["name"]
197:         curr_price = stock["price"]
198: 
199:         target_qty = int(allocation_per_stock // curr_price)
200:         current_qty = int(held_tickers.get(ticker, {}).get("hldg_qty", 0))
201: 
202:         needed_qty = target_qty - current_qty
203: 
204:         if needed_qty > 0:
205:             # Check cash limit
206:             max_qty_by_cash = int(cash // curr_price)
207:             buy_qty = min(needed_qty, max_qty_by_cash)
208: 
209:             if buy_qty > 0:
210:                 invest_amt = buy_qty * curr_price
211: 
212:                 # Position size limit check: max 20% of total equity per order
213:                 max_position_size = total_equity * 0.20
214:                 if invest_amt > max_position_size:
215:                     print(
216:                         f"   ⚠️  Order size {invest_amt:,.0f}원 exceeds 20% limit ({max_position_size:,.0f}원)"
217:                     )
218:                     print("   Reducing order size to comply with risk limits...")
219:                     buy_qty = int(max_position_size // curr_price)
220:                     invest_amt = buy_qty * curr_price
221: 
222:                     if buy_qty <= 0:
223:                         print(
224:                             f"   ⚠️  Cannot buy {name}: Even minimum order exceeds 20% limit"
225:                         )
226:                         send_discord_msg(
227:                             config,
228:                             "⚠️ [Hybrid Alpha] Order Size Blocked",
229:                             f"{name} 매수 차단: 주문 금액이 계좌의 20%를 초과합니다.",
230:                             color=0xFFA500,
231:                         )
232:                         continue
233: 
234:                 print(
235:                     f"🚀 [BUY/ADD] {name} ({ticker}): {buy_qty} shares @ ~{curr_price:,}원"
236:                 )
237:                 if not dry_run:
238:                     # Check trade limit
239:                     if not trade_limiter.check_and_increment("hybrid_alpha"):
240:                         print(f"   ⚠️  Cannot buy {name}: Daily trade limit reached")
241:                         break  # Stop buying if limit reached
242: 
243:                     result = executor.create_order(
244:                         ticker, side="BUY", qty=buy_qty, ord_type="01"
245:                     )
246:                     if result and result.get("rt_cd") == "0":
247:                         cash -= invest_amt
248:                         msg = f"**{name}** ({ticker})\n수량: {buy_qty}주 @ ~{curr_price:,}원\n매수금액: {invest_amt:,.0f}원\n━━━━━━━━━━━━━━\n💰 잔여 현금: {cash:,.0f}원"
249:                         print(f"   ✅ BUY Success: {name} ({buy_qty}주)")
250:                         send_discord_msg(
251:                             config, "🚀 [Hybrid Alpha] 매수 완료", msg, color=0x00FF00
252:                         )
253:                         buy_count += 1
254:                     else:
255:                         print(
256:                             f"   ❌ BUY Failed: {result.get('msg1') if result else 'Error'}"
257:                         )
258:                 else:
259:                     print(f"   [DRY RUN] Would buy {buy_qty} shares.")
260:                 time.sleep(1)
261:             else:
262:                 print(f"⚠️ [BUY SKIP] {name} ({ticker}) - Not enough cash.")
263:         else:
264:             print(f"💎 Holding {name} ({ticker}): Allocation reached or exceeded.")
265: 
266:     print("\n--- [Execution Finished] ---")
267: 
268: 
269: if __name__ == "__main__":
270:     # Default to DRY RUN for safety, but we'll call with False for implementation
271:     run_hybrid_alpha_execution(dry_run=True)
272: ```
---

## 🚀 3. `log_daily_equity.py` (108 lines)
```python
001: # -*- coding: utf-8 -*-
002: """
003: Daily Equity Logger
004: Logs total portfolio equity (KR + US) to CSV for performance tracking.
005: Run this script twice daily: 09:00 (after KR market open) and 23:30 (after US market close).
006: """
007: 
008: import logging
009: import yaml
010: import os
011: import sys
012: from datetime import datetime
013: import csv
014: 
015: # Update path
016: current_dir = os.path.dirname(os.path.abspath(__file__))
017: if current_dir not in sys.path:
018:     sys.path.append(current_dir)
019: 
020: from signal_mailer.kis_api_wrapper import KISAPIWrapper
021: from signal_mailer.order_executor import OrderExecutor
022: 
023: logging.basicConfig(
024:     level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
025: )
026: logger = logging.getLogger("EquityLogger")
027: 
028: 
029: def log_daily_equity():
030:     """Log current total equity to CSV file."""
031:     # Load config
032:     config_path = os.path.join(current_dir, "signal_mailer", "config.yaml")
033:     with open(config_path, "r", encoding="utf-8") as f:
034:         config = yaml.safe_load(f) or {}
035: 
036:     # Initialize
037:     kis = KISAPIWrapper(config["kis"])
038:     executor = OrderExecutor(kis)
039: 
040:     # Get current equity
041:     total_equity_krw = executor.get_total_equity()
042: 
043:     # Get detailed breakdown
044:     kr_holdings = executor.get_balance()
045:     kr_stocks_value = sum(float(h.get("evlu_amt", 0)) for h in kr_holdings)
046:     kr_cash = executor.get_cash()
047: 
048:     us_holdings = executor.get_us_balance()
049:     us_stocks_value_usd = sum(float(h.get("frcr_evlu_amt2", 0)) for h in us_holdings)
050:     us_cash_usd = executor.get_us_cash()
051: 
052:     # Approximate USD to KRW
053:     exch_rate = 1400.0
054:     us_total_usd = us_cash_usd + us_stocks_value_usd
055:     us_total_krw = us_total_usd * exch_rate
056: 
057:     # Get timestamp
058:     now = datetime.now()
059:     date_str = now.strftime("%Y-%m-%d")
060:     time_str = now.strftime("%H:%M")
061: 
062:     # CSV file path
063:     log_file = os.path.join(current_dir, "data", "equity_log.csv")
064:     os.makedirs(os.path.dirname(log_file), exist_ok=True)
065: 
066:     # Check if file exists to write header
067:     file_exists = os.path.exists(log_file)
068: 
069:     # Write to CSV
070:     with open(log_file, "a", newline="", encoding="utf-8") as f:
071:         writer = csv.writer(f)
072: 
073:         if not file_exists:
074:             writer.writerow(
075:                 [
076:                     "Date",
077:                     "Time",
078:                     "Total_KRW",
079:                     "KR_Stocks_KRW",
080:                     "KR_Cash_KRW",
081:                     "US_Stocks_USD",
082:                     "US_Cash_USD",
083:                     "US_Total_KRW",
084:                 ]
085:             )
086: 
087:         writer.writerow(
088:             [
089:                 date_str,
090:                 time_str,
100:                 f"{total_equity_krw:.0f}",
101:                 f"{kr_stocks_value:.0f}",
102:                 f"{kr_cash:.0f}",
103:                 f"{us_stocks_value_usd:.2f}",
104:                 f"{us_cash_usd:.2f}",
105:                 f"{us_total_krw:.0f}",
106:             ]
107:         )
108: 
109:     logger.info(
110:         f"📊 Equity logged: Total {total_equity_krw:,.0f} KRW (KR: {kr_stocks_value + kr_cash:,.0f}, US: {us_total_krw:,.0f})"
111:     )
112:     print(f"✅ Equity log saved to {log_file}")
113: 
114: 
115: if __name__ == "__main__":
116:     log_daily_equity()
117: ```
118: 
119: ---
120: 
121: ## 🧠 4. `signal_mailer/mama_lite_predictor.py` (250 lines)
122: ```python
123: 001: # -*- coding: utf-8 -*-
124: 002: import os
125: 003: import logging
126: 004: import pandas as pd
127: 005: import numpy as np
128: 006: import torch
129: 007: import torch.nn as nn
130: 008: import torch.nn.functional as F
131: 009: from sklearn.preprocessing import StandardScaler
132: 010: from sklearn.cluster import KMeans
133: 011: import yfinance as yf
134: 012: from datetime import datetime, timedelta
135: 013: 
136: 014: # Constants
137: 015: GNN_DATA_DIR = r"d:\gg\data\gnn"
138: 016: WEIGHT_FILE = os.path.join(GNN_DATA_DIR, "gnn_weights.pth")
139: 017: ADJ_FILE = os.path.join(GNN_DATA_DIR, "adjacency_matrix.csv")
140: 018: 
141: 019: # Tickers
142: 020: GNN_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AVGO"]
143: 021: SRL_TICKERS = ["^VIX", "^TNX", "SPY"]  # Macro indicators
144: 022: DEFENSIVE_TICKERS = ["BIL", "TLT"]
145: 023: 
146: 024: logger = logging.getLogger("MAMAPredictor")
147: 025: 
148: 026: 
149: 027: class SimpleGCN(nn.Module):
150: 028:     def __init__(self, in_features, hidden_features, out_features):
151: 029:         super(SimpleGCN, self).__init__()
152: 030:         self.conv1 = nn.Linear(in_features, hidden_features)
153: 031:         self.conv2 = nn.Linear(hidden_features, out_features)
154: 032: 
155: 033:     def forward(self, x, adj):
156: 034:         x = torch.mm(adj, x)
157: 035:         x = F.relu(self.conv1(x))
158: 036:         x = torch.mm(adj, x)
159: 037:         x = self.conv2(x)
160: 038:         return x
161: 039: 
162: 040: 
163: 041: class MAMAPredictor:
164: 042:     def __init__(self):
165: 043:         self.device = torch.device("cpu")
166: 044:         self.adj_norm = self._load_adjacency()
167: 045:         self.gnn_model = self._load_gnn_model()
168: 046:         self.scaler = StandardScaler()
169: 047:         self.kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
170: 048: 
171: 049:     def _load_adjacency(self):
172: 050:         if not os.path.exists(ADJ_FILE):
173: 051:             raise FileNotFoundError(f"Adjacency matrix not found at {ADJ_FILE}")
174: 052:         adj_df = pd.read_csv(ADJ_FILE, index_col=0)
175: 053:         A = torch.tensor(adj_df.values, dtype=torch.float32)
176: 054:         A_hat = A + torch.eye(A.shape[0])
177: 055:         D = torch.diag(torch.sum(A_hat, dim=1))
178: 056:         D_inv_sqrt = torch.pow(D, -0.5)
179: 057:         D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
180: 058:         adj_norm = torch.mm(torch.mm(D_inv_sqrt, A_hat), D_inv_sqrt)
181: 059:         return adj_norm.to(self.device)
182: 060: 
183: 061:     def _load_gnn_model(self):
184: 062:         if not os.path.exists(WEIGHT_FILE):
185: 063:             raise FileNotFoundError(f"Model weights not found at {WEIGHT_FILE}")
186: 064:         model = SimpleGCN(2, 16, 1).to(self.device)
187: 065:         model.load_state_dict(torch.load(WEIGHT_FILE, map_location=self.device))
188: 066:         model.eval()
189: 067:         return model
190: 068: 
191: 069:     def fetch_data(self, lookback_days=365):
192: 070:         """Fetch data for SRL and GNN inference using yfinance."""
193: 071:         end_date = datetime.now()
194: 072:         start_date = end_date - timedelta(days=lookback_days + 100)  # Buffer
195: 073: 
196: 074:         tickers = GNN_TICKERS + SRL_TICKERS + DEFENSIVE_TICKERS
197: 075:         data = yf.download(tickers, start=start_date, end=end_date, progress=False)[
198: 076:             "Close"
199: 077:         ]
200: 078: 
201: 079:         # Renaissance fix: yfinance multi-index columns
202: 080:         if isinstance(data.columns, pd.MultiIndex):
203: 081:             data.columns = data.columns.droplevel(1)  # Drop 'Ticker' level if present
204: 082: 
205: 083:         data = data.ffill().dropna()
206: 084:         return data
207: 085: 
208: 086:     def predict_portfolio(self):
209: 087:         """
210: 088:         Main prediction function.
211: 089:         Returns: Dict[Ticker, Weight]
212: 090:         """
213: 091:         logger.info("Fetching market data...")
214: 092:         df = self.fetch_data(lookback_days=400)  # Need >252 for VIX Z-score
215: 093: 
216: 094:         if df.empty:
217: 095:             logger.error("Failed to fetch data.")
218: 096:             return {}
219: 097: 
220: 098:         # --- Phase 1: SRL Regime Identification ---
221: 099:         # Feature Engineering (Same as mama_lite_srl_engine.py)
222: 100:         # Handle potential missing columns
223: 101:         for col in ["^VIX", "^TNX", "SPY"]:
224: 102:             if col not in df.columns:
225: 103:                 logger.error(f"Missing required ticker: {col}")
226: 104:                 return {}
227: 105: 
228: 106:         df["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df[
229: 107:             "^VIX"
230: 108:         ].rolling(252).std()
231: 109:         df["tnx_mom"] = df["^TNX"].pct_change(20)
232: 110:         df["spy_mom"] = df["SPY"].pct_change(60)
233: 111: 
234: 112:         features = df[["vix_z", "tnx_mom", "spy_mom"]].dropna()
235: 113: 
236: 114:         if features.empty:
237: 115:             logger.error("Not enough data for SRL features.")
238: 116:             return {}
239: 117: 
240: 118:         # Fit SRL Model (Production Note: Ideally load pre-fitted, but for robustness refit on expanding window)
241: 119:         X_srl = self.scaler.fit_transform(features)
242: 120:         regime_labels = self.kmeans.fit_predict(X_srl)
243: 121: 
244: 122:         current_regime = regime_labels[-1]
245: 123: 
246: 124:         # Identify 'Bull' Regime on history
247: 125:         # Bull regime has highest mean SPY return
248: 126:         features_with_ret = features.copy()
249: 127:         features_with_ret["spy_ret"] = df["SPY"].pct_change().reindex(features.index)
250: 128:         features_with_ret["regime"] = regime_labels
251: 129: 
252: 130:         regime_spy_ret = features_with_ret.groupby("regime")["spy_ret"].mean()
253: 131:         bull_regime = regime_spy_ret.idxmax()
254: 132: 
255: 133:         logger.info(f"Current Regime: {current_regime} (Bull Regime ID: {bull_regime})")
256: 134: 
257: 135:         target_weights = {}
258: 136: 
259: 137:         if current_regime == bull_regime:
260: 138:             logger.info("Regime is BULL -> Engaging GNN Selection")
261: 139:             # --- Phase 2: GNN Inference ---
262: 140:             node_feats = []
263: 141: 
264: 142:             # Use data up to 'yesterday' (latest available close)
265: 143:             for t in GNN_TICKERS:
266: 144:                 if t not in df.columns:
267: 145:                     node_feats.append([0.0, 0.0])
268: 146:                     continue
269: 147: 
270: 148:                 # Mom: (Price_t / Price_t-21) - 1
271: 149:                 p_now = df[t].iloc[-1]
272: 150:                 p_prev = df[t].iloc[-22]
273: 151:                 mom = (p_now / p_prev) - 1
274: 152: 
275: 153:                 # Vol: Std of daily returns for last 21 days
276: 154:                 vol = df[t].pct_change().iloc[-21:].std()
277: 155:                 if np.isnan(vol):
278: 156:                     vol = 0.0
279: 157: 
280: 158:                 node_feats.append([mom, vol])
281: 159: 
282: 160:             x_gnn = torch.tensor(node_feats, dtype=torch.float32).to(self.device)
283: 161: 
284: 162:             with torch.no_grad():
285: 163:                 scores = self.gnn_model(x_gnn, self.adj_norm).squeeze()
286: 164: 
287: 165:             # Select Top 3
288: 166:             top_indices = scores.argsort(descending=True)[:3]
289: 167:             top_tickers = [GNN_TICKERS[i] for i in top_indices]
290: 168: 
291: 169:             score_map = {
292: 170:                 GNN_TICKERS[i]: scores[i].item() for i in range(len(GNN_TICKERS))
293: 171:             }
294: 172:             logger.info(f"GNN Scores: {score_map}")
295: 173:             logger.info(f"Selected Top 3: {top_tickers}")
296: 174: 
297: 175:             weight_per_stock = 1.0 / 3.0
298: 176:             for t in top_tickers:
299: 177:                 target_weights[t] = weight_per_stock
300: 178: 
301: 179:         else:
302: 180:             logger.info("Regime is NOT Bull -> Defensive Mode (BIL/TLT)")
303: 181:             # 50/50 BIL/TLT (Simple Defensive)
304: 182:             target_weights["BIL"] = 0.5
305: 183:             target_weights["TLT"] = 0.5
306: 184: 
307: 185:         return target_weights
308: 186: 
309: 187:     def get_current_regime(self):
310: 188:         """
311: 189:         Get current market regime classification.
312: 190:         Returns: str - 'Bull', 'Bear', 'Crisis', or 'Neutral'
313: 191:         """
314: 192:         try:
315: 193:             df = self.fetch_data(lookback_days=400)
316: 194:             if df.empty:
317: 195:                 return "Unknown"
318: 196: 
319: 197:             # Calculate SRL features
320: 198:             for col in ["^VIX", "^TNX", "SPY"]:
321: 199:                 if col not in df.columns:
322: 200:                     return "Unknown"
323: 201: 
324: 202:             df["vix_z"] = (df["^VIX"] - df["^VIX"].rolling(252).mean()) / df[
325: 203:                 "^VIX"
326: 204:             ].rolling(252).std()
327: 205:             df["tnx_mom"] = df["^TNX"].pct_change(20)
328: 206:             df["spy_mom"] = df["SPY"].pct_change(60)
329: 207: 
330: 208:             features = df[["vix_z", "tnx_mom", "spy_mom"]].dropna()
331: 209:             if features.empty:
332: 210:                 return "Unknown"
333: 211: 
334: 212:             # Fit and predict
335: 213:             X_srl = self.scaler.fit_transform(features)
336: 214:             regime_labels = self.kmeans.fit_predict(X_srl)
337: 215:             current_regime = regime_labels[-1]
338: 216: 
339: 217:             # Classify regime based on characteristics
340: 218:             features_with_ret = features.copy()
341: 219:             features_with_ret["spy_ret"] = (
342: 220:                 df["SPY"].pct_change().reindex(features.index)
343: 221:             )
344: 222:             features_with_ret["regime"] = regime_labels
345: 223: 
346: 224:             regime_spy_ret = features_with_ret.groupby("regime")["spy_ret"].mean()
347: 225:             bull_regime = regime_spy_ret.idxmax()
348: 226:             bear_regime = regime_spy_ret.idxmin()
349: 227: 
350: 228:             # Get current VIX Z-score
351: 229:             current_vix_z = features["vix_z"].iloc[-1]
352: 230: 
353: 231:             if current_regime == bull_regime:
354: 232:                 return "Bull"
355: 233:             elif current_regime == bear_regime:
356: 234:                 if current_vix_z > 2.0:  # Extreme fear
357: 235:                     return "Crisis"
358: 236:                 return "Bear"
359: 237:             else:
360: 238:                 return "Neutral"
361: 239: 
362: 240:         except Exception as e:
363: 241:             logger.error(f"Error getting regime: {e}")
364: 242:             return "Unknown"
365: 243: 
366: 244: 
367: 245: if __name__ == "__main__":
368: 246:     logging.basicConfig(level=logging.INFO)
369: 247:     predictor = MAMAPredictor()
370: 248:     weights = predictor.predict_portfolio()
371: 249:     print("Target Portfolio Weights:", weights)
372: 250: ```
---

## 🚀 5. `signal_mailer/mama_lite_rebalancer.py` (142 lines)
```python
123: 001: import logging
124: 002: import yaml
125: 003: import sys
126: 004: import os
127: 005: import time
128: 006: 
129: 007: # Add project root to path
130: 008: sys.path.append(os.getcwd())
131: 009: 
132: 010: from signal_mailer.kis_api_wrapper import KISAPIWrapper
133: 011: from signal_mailer.order_executor import OrderExecutor
134: 012: from signal_mailer.mama_lite_predictor import MAMAPredictor
135: 013: 
136: 014: logging.basicConfig(
137: 015:     level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
138: 016: )
139: 017: logger = logging.getLogger("MAMA-Rebalancer")
140: 018: 
141: 019: 
142: 020: def run_mama_rebalance():
143: 021:     logger.info("Starting MAMA Lite Rebalancing Process...")
144: 022: 
145: 023:     # 1. Load Config
146: 024:     with open("signal_mailer/config.yaml", "r", encoding="utf-8") as f:
147: 025:         config = yaml.safe_load(f)
148: 026: 
149: 027:     kis = KISAPIWrapper(config["kis"])
150: 028:     executor = OrderExecutor(kis)
151: 029:     predictor = MAMAPredictor()
152: 030: 
153: 031:     # 2. Get Target Weights from MAMA Lite
154: 032:     # Current MAMA Lite signal is calculated using daily data via yfinance
155: 033:     logger.info("Predicting target weights via MAMA Lite...")
156: 034:     target_weights = predictor.predict_portfolio()
157: 035:     # Example: {'BIL': 0.5, 'TLT': 0.5}
158: 036: 
159: 037:     if not target_weights:
160: 038:         logger.error("Failed to get target weights from MAMA Lite.")
161: 039:         return
162: 040: 
163: 041:     logger.info(f"Target Weights: {target_weights}")
164: 042: 
165: 043:     # 3. Get Current Portfolio Status
166: 044:     logger.info("Fetching current US holdings and cash...")
167: 045:     current_holdings = executor.get_us_balance()  # List[Dict]
168: 046:     # Current holdings output1 fields: 'ovrs_pdno' (ticker), 'ovrs_cblc_qty' (qty), etc.
169: 047: 
170: 048:     # KIS Mock US Balance summary often found in output2 or separate cash call
171: 049:     # Let's use get_us_cash which uses inquire-psbl-order
172: 050:     usd_cash = executor.get_us_cash()
173: 051:     logger.info(f"Available USD Cash: ${usd_cash:,.2f}")
174: 052: 
175: 053:     # If USD cash is 0, check if we can use KRW (Integrated Funds)
176: 054:     if usd_cash < 10:  # Threshold for "empty"
177: 055:         logger.warning(
178: 056:             "USD Cash is low. Checking KRW cash for Integrated Funds usage..."
179: 057:         )
180: 058:         krw_cash = executor.get_cash()
181: 059:         logger.info(f"OrderExecutor returned KRW Cash: ₩{krw_cash}")
182: 060:         if krw_cash > 100000:
183: 061:             # Approx USD conversion for calculation (not official rate)
184: 062:             usd_cash = krw_cash / 1400.0
185: 063:             logger.info(f"Using estimated Integrated USD: ${usd_cash:,.2f}")
186: 064:         else:
187: 065:             logger.error("Insufficient funds (Both USD and KRW) in mock account.")
188: 066:             return
189: 067: 
190: 068:     # 4. Calculate Orders
191: 069:     # Filter out tickers already held (if any)
192: 070:     # Using a set for comparison
193: 071:     held_tickers_set = {
194: 072:         h.get("ovrs_pdno")
195: 073:         for h in current_holdings
196: 074:         if h.get("ovrs_cblc_qty", "0") != "0"
197: 075:     }
198: 076:     logger.info(f"Current US Holdings: {held_tickers_set}")
199: 077:     tickers_to_buy = {
200: 078:         t: w for t, w in target_weights.items() if t not in held_tickers_set and w > 0
201: 079:     }
202: 080: 
203: 081:     if not tickers_to_buy:
204: 082:         logger.info("No new tickers to buy. Already in position or weights are 0.")
205: 083:         return
206: 084: 
207: 085:     # Calculate total weight of tickers we are about to buy
208: 086:     buying_weight_sum = sum(tickers_to_buy.values())
209: 087: 
210: 088:     # Calculate quantity for each ticker
211: 089:     for ticker, weight in tickers_to_buy.items():
212: 090:         time.sleep(1.0)  # Small delay to avoid hammering
213: 091: 
214: 092:         # Try to infer exchange
215: 093:         exchange = "NAS"
216: 094:         if ticker in ["BIL", "GLD", "SPY", "VTI", "COWZ", "BTAL", "PFIX"]:
217: 095:             exchange = "AMS" if ticker in ["BIL", "GLD", "BTAL", "PFIX"] else "NYS"
218: 096: 
219: 097:         logger.info(f"Fetching price for {ticker} on {exchange}...")
220: 098:         price = kis.get_us_current_price(ticker, exchange=exchange)
221: 099:         if not price or price <= 0:
222: 100:             logger.error(f"Could not fetch price for {ticker}. Skipping.")
223: 101:             continue
224: 102: 
225: 103:         # Allocate from the CURRENTly available cash based on relative weight among remaining items
226: 104:         # Use a slightly larger buffer (5%) for fees/slippage just in case
227: 105:         portion = weight / buying_weight_sum
228: 106:         allocation_usd = usd_cash * portion * 0.92
229: 107:         qty = int(allocation_usd / price)
230: 108: 
231: 109:         if qty <= 0:
232: 110:             logger.warning(
233: 111:                 f"Quantity for {ticker} is 0. (Alloc: ${allocation_usd:.2f} / Price: ${price:.2f})"
234: 112:             )
235: 113:             continue
236: 114: 
237: 115:         logger.info(
238: 116:             f"Planned Order: BUY {ticker} | {qty} shares @ ~${price:.2f} (Total: ${price * qty:.2f})"
239: 117:         )
240: 118: 
241: 119:         # 5. Execute Order
242: 120:         # US Order requires limit price. We use current price + 0.1% for Buy orders for high fill probability.
243: 121:         limit_price = round(price * 1.001, 2)
244: 122: 
245: 123:         res = executor.create_us_order(
246: 124:             ticker=ticker,
247: 125:             exchange=exchange,
248: 126:             side="BUY",
249: 127:             qty=qty,
250: 128:             price=limit_price,
251: 129:             ord_type="00",  # Limit
252: 130:         )
253: 131: 
254: 132:         if res and res.get("rt_cd") == "0":
255: 133:             logger.info(f"Successfully placed order for {ticker}.")
256: 134:         else:
257: 135:             logger.error(
258: 136:                 f"Failed to place order for {ticker}: {res.get('msg1') if res else 'Unknown error'}"
259: 137:             )
260: 138: 
261: 139: 
262: 140: if __name__ == "__main__":
263: 141:     run_mama_rebalance()
264: 142: ```
265: 
266: ---
267: 
268: ## 🚀 6. `signal_mailer/order_executor.py` (395 lines)
269: ```python
270: 001: # -*- coding: utf-8 -*-
271: 002: import logging
272: 003: import time
273: 004: from typing import Dict, Any, List, Optional
274: 005: from signal_mailer.kis_api_wrapper import KISAPIWrapper
275: 006: 
276: 007: logger = logging.getLogger(__name__)
277: 008: 
278: 009: 
279: 010: class OrderExecutor:
280: 011:     """
281: 012:     Handles order execution for Korean stocks via KIS API.
282: 013:     Designed for the 'Hybrid Alpha' strategy.
283: 014:     """
284: 015: 
285: 016:     def __init__(self, kis: KISAPIWrapper):
286: 017:         self.kis = kis
287: 018:         self.base_url = kis.base_url
288: 019:         self.cano = kis.cano
289: 020:         self.acnt_prdt_cd = kis.acnt_prdt_cd
290: 021: 
291: 022:     def _get_order_headers(self, tr_id: str) -> Dict[str, str]:
292: 023:         """Prepare headers for order-related TR IDs."""
293: 024:         headers = self.kis.headers.copy()
294: 025:         headers["tr_id"] = tr_id
295: 026:         return headers
296: 027: 
297: 028:     def create_order(
298: 029:         self, ticker: str, side: str, qty: int, price: int = 0, ord_type: str = "01"
299: 030:     ) -> Optional[Dict[str, Any]]:
300: 031:         """
301: 032:         Create a localized market or limit order.
302: 033:         side: 'BUY' or 'SELL'
303: 034:         ord_type: '01' (Market), '00' (Limit)
304: 035:         """
305: 036:         tr_id = "TTTC0802U" if side == "BUY" else "TTTC0801U"
306: 037:         if self.kis.is_mock:
307: 038:             tr_id = "VTTC0802U" if side == "BUY" else "VTTC0801U"
308: 039: 
309: 040:         url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
310: 041: 
311: 042:         body = {
312: 043:             "CANO": self.cano,
313: 044:             "ACNT_PRDT_CD": self.acnt_prdt_cd,
314: 045:             "PDNO": ticker,
315: 046:             "ORD_DVSN": ord_type,
316: 047:             "ORD_QTY": str(qty),
317: 048:             "ORD_UNPR": str(price) if ord_type == "00" else "0",
318: 049:         }
319: 050: 
320: 051:         max_retries = 3
321: 052:         for attempt in range(max_retries):
322: 053:             try:
323: 054:                 r = self.kis.call_post(
324: 055:                     url, headers=self._get_order_headers(tr_id), json=body
325: 056:                 )
326: 057:                 if r.status_code == 200:
327: 058:                     data = r.json()
328: 059:                     if data.get("rt_cd") == "0":
329: 060:                         logger.info(f"[ORDER] {side} {ticker} Success: {qty} shares")
330: 061:                         return data
331: 062:                     else:
332: 063:                         logger.error(
333: 064:                             f"[ORDER] {side} {ticker} Failed: {data.get('msg1')}"
334: 065:                         )
335: 066:                         return data
336: 067:                 else:
337: 068:                     logger.error(
338: 069:                         f"[ORDER] {side} {ticker} API Error: {r.status_code} {r.text}"
339: 070:                     )
340: 071:                     if attempt < max_retries - 1:
341: 072:                         wait_time = (2**attempt) * 0.5  # 0.5s, 1s, 2s
342: 073:                         logger.info(
343: 074:                             f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
344: 075:                         )
345: 076:                         time.sleep(wait_time)
346: 077:                         continue
347: 078:                     return None
348: 079:             except Exception as e:
349: 080:                 logger.error(f"[ORDER] Execution error for {ticker}: {e}")
350: 081:                 if attempt < max_retries - 1:
351: 082:                     wait_time = (2**attempt) * 0.5
352: 083:                     logger.info(
353: 084:                         f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
354: 085:                     )
355: 086:                     time.sleep(wait_time)
356: 087:                     continue
357: 088:                 return None
358: 089:         return None
359: 090: 
360: 091:     def get_balance(self) -> List[Dict[str, Any]]:
361: 092:         """Fetch current stock holdings."""
362: 093:         tr_id = "TTTC8434R"  # Real
363: 094:         params = {
364: 095:             "CANO": self.cano,
365: 096:             "ACNT_PRDT_CD": self.acnt_prdt_cd,
366: 097:             "AFHR_FLG": "N",
367: 098:             "OFR_FLG": "N",
368: 099:             "INQR_DVSN": "01",
369: 100:             "UNPR_DVSN": "01",
370: 101:             "FUND_STTL_ICLD_YN": "N",
371: 102:             "FRLI_GVOFF_SYS_DVSN": "00",
372: 103:             "PRCS_DVSN": "01",
373: 104:             "CTX_AREA_FK100": "",
374: 105:             "CTX_AREA_NK100": "",
375: 106:         }
376: 107: 
377: 108:         if self.kis.is_mock:
378: 109:             tr_id = "VTTC8434R"  # Mock
379: 110:             params = {
380: 111:                 "CANO": self.cano,
381: 112:                 "ACNT_PRDT_CD": self.acnt_prdt_cd,
382: 113:                 "AFHR_FLPR_YN": "N",
383: 114:                 "OFL_YN": "N",
384: 115:                 "INQR_DVSN": "02",
385: 116:                 "UNPR_DVSN": "01",
386: 117:                 "FUND_STTL_ICLD_YN": "N",
387: 118:                 "FNCG_AMT_AUTO_RDPT_YN": "N",
388: 119:                 "PRCS_DVSN": "00",
389: 120:                 "CTX_AREA_FK100": "",
390: 121:                 "CTX_AREA_NK100": "",
391: 122:             }
392: 123: 
393: 124:         url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
394: 125: 
395: 126:         try:
396: 127:             headers = self.kis.headers.copy()
397: 128:             headers["tr_id"] = tr_id
398: 129:             r = self.kis.call_get(url, headers=headers, params=params)
399: 130:             if r.status_code == 200:
400: 131:                 data = r.json()
401: 132:                 return data.get("output1", [])
402: 133:             return []
403: 134:         except Exception as e:
404: 135:             logger.error(f"[BALANCE] Inquiry error: {e}")
405: 136:             return []
406: 137: 
407: 138:     def get_cash(self) -> int:
408: 139:         """Get available cash for trading."""
409: 140:         tr_id = "TTTC8434R"  # Real
410: 141:         params = {
411: 142:             "CANO": self.cano,
412: 143:             "ACNT_PRDT_CD": self.acnt_prdt_cd,
413: 144:             "AFHR_FLG": "N",
414: 145:             "OFR_FLG": "N",
415: 146:             "INQR_DVSN": "01",
416: 147:             "UNPR_DVSN": "01",
417: 148:             "FUND_STTL_ICLD_YN": "N",
418: 149:             "FRLI_GVOFF_SYS_DVSN": "00",
419: 150:             "PRCS_DVSN": "01",
420: 151:             "CTX_AREA_FK100": "",
421: 152:             "CTX_AREA_NK100": "",
422: 153:         }
423: 154: 
424: 155:         if self.kis.is_mock:
425: 156:             tr_id = "VTTC8434R"  # Mock
426: 157:             params = {
427: 158:                 "CANO": self.cano,
428: 159:                 "ACNT_PRDT_CD": self.acnt_prdt_cd,
429: 160:                 "AFHR_FLPR_YN": "N",
430: 161:                 "OFL_YN": "N",
431: 162:                 "INQR_DVSN": "02",
432: 163:                 "UNPR_DVSN": "01",
433: 164:                 "FUND_STTL_ICLD_YN": "N",
434: 165:                 "FNCG_AMT_AUTO_RDPT_YN": "N",
435: 166:                 "PRCS_DVSN": "00",
436: 167:                 "CTX_AREA_FK100": "",
437: 168:                 "CTX_AREA_NK100": "",
438: 169:             }
439: 170: 
440: 171:         url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
441: 172:         try:
442: 173:             headers = self.kis.headers.copy()
443: 174:             headers["tr_id"] = tr_id
444: 175:             logger.info(f"[CASH-Probing] URL: {url} | Params: {params} | TR: {tr_id}")
445: 176:             r = self.kis.call_get(url, headers=headers, params=params)
446: 177:             if r.status_code == 200:
447: 178:                 data = r.json()
448: 179:                 # output2 contains summary data including cash
449: 180:                 summary_list = data.get("output2", [])
450: 181:                 if not summary_list:
451: 182:                     logger.warning(f"[CASH] output2 is empty. Data: {data}")
452: 183:                     return 0
453: 184:                 summary = summary_list[0]
454: 185:                 # Try multiple keys for cash in mock/real summary
455: 186:                 v1 = summary.get("dnca_tot_amt")
456: 187:                 v2 = summary.get("tot_evlu_amt")
457: 188:                 v3 = summary.get("nass_amt")
458: 189:                 logger.debug(f"[CASH] Raw keys: dnca={v1}, tot={v2}, nass={v3}")
459: 190:                 try:
460: 191:                     cash = int(float(v1 or v2 or v3 or 0))
461: 192:                 except (ValueError, TypeError):
462: 193:                     cash = 0
463: 194:                 return cash
464: 195:             else:
465: 196:                 logger.error(f"[CASH] API Error: {r.status_code} {r.text}")
466: 197:                 return 0
467: 198:         except Exception as e:
468: 199:             logger.error(f"[CASH] Inquiry error: {e}")
469: 200:             return 0
470: 201: 
471: 202:     def create_us_order(
472: 203:         self,
473: 204:         ticker: str,
474: 205:         exchange: str,
475: 206:         side: str,
476: 207:         qty: int,
477: 208:         price: float = 0,
478: 209:         ord_type: str = "00",
479: 210:     ) -> Optional[Dict[str, Any]]:
480: 211:         """
481: 212:         Create a US stock market or limit order.
482: 213:         side: 'BUY' or 'SELL'
483: 214:         ord_type: '00' (Limit - Specify Price), 'LOO', 'LOC', 'MOO', 'MOC' etc.
484: 215:         NOTE: KIS Overseas API does NOT support simple '01' (Market) for US stocks easily.
485: 216:               Usually requires '00' with price.
486: 217:               However, for 'Market' equivalent, we might need to use specific codes or price logic.
487: 218:               For now, we will stick to LIMIT orders (00) as default for safety.
488: 219:         """
489: 220:         tr_id = (
490: 221:             "JTTT1002U" if side == "BUY" else "JTTT1006U"
491: 222:         )  # Real (US specific TR may vary, check JTTT1002U=Buy)
492: 223:         # Actually standard: JTTT1002U (Buy), JTTT1006U (Sell)
493: 224: 
494: 225:         if self.kis.is_mock:
495: 226:             tr_id = "VTTT1002U" if side == "BUY" else "VTTT1006U"
496: 227: 
497: 228:         url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
498: 229: 
499: 230:         body = {
500: 231:             "CANO": self.cano,
501: 232:             "ACNT_PRDT_CD": self.acnt_prdt_cd,
502: 233:             "OVRS_EXCG_CD": exchange.upper(),
503: 234:             "PDNO": ticker,
504: 235:             "ORD_DVSN": ord_type,  # 00: Limit
505: 236:             "ORD_QTY": str(qty),
506: 237:             "OVRS_ORD_UNPR": str(price),
507: 238:             "ORD_SVR_DVSN_CD": "0",  # 0: Normal
508: 239:         }
509: 240: 
510: 241:         max_retries = 3
511: 242:         for attempt in range(max_retries):
512: 243:             try:
513: 244:                 r = self.kis.call_post(
514: 245:                     url, headers=self._get_order_headers(tr_id), json=body
515: 246:                 )
516: 247:                 if r.status_code == 200:
517: 248:                     data = r.json()
518: 249:                     if data.get("rt_cd") == "0":
519: 250:                         logger.info(
520: 251:                             f"[US-ORDER] {side} {ticker} Success: {qty} shares @ {price}"
521: 252:                         )
522: 253:                         return data
523: 254:                     else:
524: 255:                         logger.error(
525: 256:                             f"[US-ORDER] {side} {ticker} Failed: {data.get('msg1')}"
526: 257:                         )
527: 258:                         return data
528: 259:                 else:
529: 260:                     logger.error(
530: 261:                         f"[US-ORDER] {side} {ticker} API Error: {r.status_code} {r.text}"
531: 262:                     )
532: 263:                     if attempt < max_retries - 1:
533: 264:                         wait_time = (2**attempt) * 0.5  # 0.5s, 1s, 2s
534: 265:                         logger.info(
535: 266:                             f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
536: 267:                         )
537: 268:                         time.sleep(wait_time)
538: 269:                         continue
539: 270:                     return None
540: 271:             except Exception as e:
541: 272:                 logger.error(f"[US-ORDER] Execution error for {ticker}: {e}")
542: 273:                 if attempt < max_retries - 1:
543: 274:                     wait_time = (2**attempt) * 0.5
544: 275:                     logger.info(
545: 276:                         f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})"
546: 277:                     )
547: 278:                     time.sleep(wait_time)
548: 279:                     continue
549: 280:                 return None
550: 281:         return None
551: 282: 
552: 283:     def get_us_balance(self) -> List[Dict[str, Any]]:
553: 284:         """Fetch US stock holdings by probing multiple exchanges."""
554: 285:         tr_id = "TTTS3012R"
555: 286:         if self.kis.is_mock:
556: 287:             tr_id = "VTTS3012R"
557: 288: 
558: 289:         url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
559: 290:         exchanges = ["AMS", "NAS", "NYS"]
560: 291:         all_holdings = []
561: 292: 
562: 293:         for ex in exchanges:
563: 294:             params = {
564: 295:                 "CANO": self.cano,
565: 296:                 "ACNT_PRDT_CD": self.acnt_prdt_cd,
566: 297:                 "OVRS_EXCG_CD": ex,
567: 298:                 "TR_CRCY_CD": "USD",
568: 299:                 "CTX_AREA_FK200": "",
569: 300:                 "CTX_AREA_NK200": "",
570: 301:             }
571: 302:             try:
572: 303:                 headers = self.kis.headers.copy()
573: 304:                 headers["tr_id"] = tr_id
574: 305:                 r = self.kis.call_get(url, headers=headers, params=params)
575: 306:                 if r.status_code == 200:
576: 307:                     data = r.json()
577: 308:                     if data.get("rt_cd") == "0":
578: 309:                         all_holdings.extend(data.get("output1", []))
579: 310:             except Exception as e:
580: 311:                 logger.error(f"[US-BALANCE] Error for {ex}: {e}")
581: 312: 
582: 313:         return all_holdings
583: 314: 
584: 315:     def get_us_cash(self) -> float:
585: 316:         """Get available US cash (Order Possible Amount)."""
586: 317:         # TR for US Buy Possible: TTTS3007R (Real), VTTS3007R (Mock)
587: 318:         tr_id = "TTTS3007R"
588: 319:         if self.kis.is_mock:
589: 320:             tr_id = "VTTS3007R"
590: 321: 
591: 322:         url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order"
592: 323: 
593: 324:         params = {
594: 325:             "CANO": self.cano,
595: 326:             "ACNT_PRDT_CD": self.acnt_prdt_cd,
596: 327:             "OVRS_EXCG_CD": "NAS",
597: 328:             "OVRS_ORD_UNPR": "0",
598: 329:             "ITEM_CD": "AAPL",
599: 330:         }
600: 331: 
601: 332:         try:
602: 333:             headers = self.kis.headers.copy()
603: 334:             headers["tr_id"] = tr_id
604: 335:             r = self.kis.call_get(url, headers=headers, params=params)
605: 336:             if r.status_code == 200:
606: 337:                 data = r.json()
607: 338:                 output = data.get("output")
608: 339:                 if output:
609: 340:                     # 'frcr_ord_psbl_amt1' is usually the orderable amount in foreign currency
610: 341:                     val = output.get("frcr_ord_psbl_amt1") or output.get(
611: 342:                         "ovrs_reva_mny1", 0
612: 343:                     )
613: 344:                     return float(val)
614: 345:             return 0.0
615: 346:         except Exception as e:
616: 347:             logger.error(f"[US-CASH] Inquiry error: {e}")
617: 348:             return 0.0
618: 349: 
619: 350:     def get_total_equity(self) -> float:
620: 351:         """Calculate total equity across Korea and US assets in KRW."""
621: 352:         # 1. Domestic (KRW)
622: 353:         # Use tot_evlu_amt from output2 which includes (Stock Value + Cash)
623: 354:         tr_id = "TTTC8434R"
624: 355:         if self.kis.is_mock:
625: 356:             tr_id = "VTTC8434R"
626: 357: 
627: 358:         params = {
628: 359:             "CANO": self.cano,
629: 360:             "ACNT_PRDT_CD": self.acnt_prdt_cd,
630: 361:             "AFHR_FLPR_YN": "N",
631: 362:             "OFL_YN": "N",
632: 363:             "INQR_DVSN": "02",
633: 364:             "UNPR_DVSN": "01",
634: 365:             "FUND_STTL_ICLD_YN": "N",
635: 366:             "FNCG_AMT_AUTO_RDPT_YN": "N",
636: 367:             "PRCS_DVSN": "01" if not self.kis.is_mock else "00",
637: 368:             "CTX_AREA_FK100": "",
638: 369:             "CTX_AREA_NK100": "",
639: 370:         }
640: 371: 
641: 372:         url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
642: 373:         dom_total_krw = 0.0
643: 374:         try:
644: 375:             headers = self.kis.headers.copy()
645: 376:             headers["tr_id"] = tr_id
646: 377:             r = self.kis.call_get(url, headers=headers, params=params)
647: 378:             if r.status_code == 200:
648: 379:                 data = r.json()
649: 380:                 summary_list = data.get("output2", [])
650: 381:                 if summary_list:
651: 382:                     dom_total_krw = float(summary_list[0].get("tot_evlu_amt", 0))
652: 383:         except Exception as e:
653: 384:             logger.error(f"[TOTAL-EQUITY-KR] Error: {e}")
654: 385: 
655: 386:         # 2. Overseas (in USD)
656: 387:         us_cash_usd = float(self.get_us_cash())
657: 388:         us_holdings = self.get_us_balance()
658: 389:         us_val_usd = sum(float(h.get("frcr_evlu_amt2", 0)) for h in us_holdings)
659: 390: 
660: 391:         # 3. Sum up (Approx exchange rate 1400)
661: 392:         exch_rate = 1400.0
662: 393:         total_krw = dom_total_krw + (us_cash_usd + us_val_usd) * exch_rate
663: 394:         return total_krw
664: 395: ```
 ---

## 🚀 7. `signal_mailer/kis_api_wrapper.py` (361 lines)
```python
001: import logging
002: import time
003: import os
004: import requests
005: from collections import deque
006: from typing import Dict, Any, Optional, List
007: 
008: logger = logging.getLogger(__name__)
009: 
010: 
011: class RateLimiter:
012:     def __init__(self, max_calls: int, period: float = 1.0):
013:         self.max_calls = max_calls
014:         self.period = period
015:         self.calls = deque()
016: 
017:     def wait(self) -> None:
018:         while True:
019:             now = time.time()
020:             while self.calls and now - self.calls[0] > self.period:
021:                 self.calls.popleft()
022:             if len(self.calls) < self.max_calls:
023:                 self.calls.append(now)
024:                 return
025:             time.sleep(0.01)
026: 
027: 
028: class KISAPIWrapper:
029:     """
030:     Modular wrapper for Korea Investment & Securities (KIS) Open API.
031:     Adapted from kis_bot_v7_4.py with standardized structure.
032:     """
033: 
034:     def __init__(self, config: Dict[str, Any]):
035:         self.app_key = config.get("app_key")
036:         self.app_secret = config.get("app_secret")
037:         self.cano = config.get("account_no")
038:         self.acnt_prdt_cd = config.get("account_prod_code", "01")
039:         self.is_mock = config.get("is_mock", False)
040:         self.base_url = config.get("url_base")
041: 
042:         self.access_token = ""
043:         self.headers = {}
044: 
045:         # Rate Limiters (Mock: 2 GET/sec, 1 POST/sec | Real: 20 GET/sec, 20 POST/sec)
046:         if self.is_mock:
047:             self.limiter_get = RateLimiter(2, 1.0)
048:             self.limiter_post = RateLimiter(1, 1.0)
049:         else:
050:             self.limiter_get = RateLimiter(20, 1.0)
051:             self.limiter_post = RateLimiter(20, 1.0)
052: 
053:         self._auth()
054: 
055:     def _auth(self) -> None:
056:         """Authenticate and retrieve access token with local caching."""
057:         import json  # Local import to avoid any issues
058: 
059:         # Use different cache files for mock vs real to avoid collisions
060:         env_suffix = "mock" if self.is_mock else "real"
061:         cache_path = os.path.join(os.getcwd(), f".kis_token_cache_{env_suffix}.json")
062: 
063:         # 1. Try Loading from Cache
064:         if os.path.exists(cache_path):
065:             try:
066:                 with open(cache_path, "r") as f:
067:                     cache = json.load(f)
068:                     # Expiry check (23 hours)
069:                     if time.time() - cache.get("timestamp", 0) < 23 * 3600:
070:                         self.access_token = cache.get("access_token")
071:                         self.headers = {
072:                             "content-type": "application/json",
073:                             "authorization": f"Bearer {self.access_token}",
074:                             "appkey": self.app_key,
075:                             "appsecret": self.app_secret,
076:                             "custtype": "P",
077:                         }
078:                         logger.debug(f"[KIS] Loaded {env_suffix} token from cache")
079:                         return
080:             except Exception:
081:                 pass
082: 
083:         # 2. Issue New Token
084:         url = f"{self.base_url}/oauth2/tokenP"
085:         body = {
086:             "grant_type": "client_credentials",
087:             "appkey": self.app_key,
088:             "appsecret": self.app_secret,
089:         }
090:         self.limiter_post.wait()
091:         try:
092:             r = requests.post(
093:                 url, json=body, headers={"Content-Type": "application/json"}, timeout=10
094:             )
095:             if r.status_code == 200:
096:                 data = r.json()
097:                 self.access_token = data.get("access_token")
098:                 # Save to Cache
099:                 with open(cache_path, "w") as f:
100:                     json.dump(
101:                         {"access_token": self.access_token, "timestamp": time.time()}, f
102:                     )
103: 
104:                 self.headers = {
105:                     "content-type": "application/json",
106:                     "authorization": f"Bearer {self.access_token}",
107:                     "appkey": self.app_key,
108:                     "appsecret": self.app_secret,
109:                     "custtype": "P",
110:                 }
111:                 env_suffix = "mock" if self.is_mock else "real"
112:                 logger.debug(f"[KIS] New {env_suffix} token issued and cached")
113:             else:
114:                 logger.error(f"[KIS] Auth Failed Status {r.status_code}: {r.text}")
115:         except Exception as e:
116:             logger.error(f"[KIS] Auth Exception: {e}")
117: 
118:     def get_current_price(self, ticker: str) -> Optional[Dict[str, Any]]:
119:         """Inquire current price for a domestic stock."""
120:         url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
121:         headers = self.headers.copy()
122:         headers["tr_id"] = "FHKST01010100"
123:         params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker}
124:         self.limiter_get.wait()
125:         try:
126:             r = requests.get(url, headers=headers, params=params, timeout=5)
127:             if r.status_code == 200:
128:                 data = r.json()
129:                 if data.get("rt_cd") != "0":
130:                     logger.error(f"[KIS] Price API Error: {data.get('msg1')}")
131:                 return data.get("output")
132:             return None
133:         except Exception as e:
134:             logger.error(f"[KIS] Price Inquiry Error for {ticker}: {e}")
135:             return None
136: 
137:     def get_ohlcv_recent(self, ticker: str) -> Optional[Dict[str, Any]]:
138:         """Inquire daily OHLCV for recent days (Domestic)."""
139:         url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
140:         headers = self.headers.copy()
141:         headers["tr_id"] = "FHKST01010400"
142:         params = {
143:             "FID_COND_MRKT_DIV_CODE": "J",
144:             "FID_INPUT_ISCD": ticker,
145:             "FID_PERIOD_DIV_CODE": "D",
146:             "FID_ORG_ADJ_PRC": "1",
147:         }
148:         self.limiter_get.wait()
149:         try:
150:             r = requests.get(url, headers=headers, params=params, timeout=5)
151:             if r.status_code == 200:
152:                 data = r.json()
153:                 if data.get("rt_cd") != "0":
154:                     logger.error(
155:                         f"[KIS] OHLCV API Error for {ticker}: {data.get('msg1')}"
156:                     )
157:                 return data
158:             return None
159:         except Exception as e:
160:             logger.error(f"[KIS] OHLCV Inquiry Error for {ticker}: {e}")
161:             return None
162: 
163:     def get_us_current_price(
164:         self, ticker: str, exchange: str = "NAS"
165:     ) -> Optional[float]:
166:         """
167:         Inquire current price for a US stock.
168:         exchange: NAS (Nasdaq), NYS (NYSE), AMS (Amex)
169:         """
170:         url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price"
171:         headers = self.headers.copy()
172:         headers["tr_id"] = (
173:             "HHDFS00000300"  # Real/Mock same for quotation usually, check docs if fails
174:         )
175: 
176:         # Mapping for safety, though user should pass correct code
177:         excd = exchange.upper()
178: 
179:         params = {
180:             "AUTH": "",
181:             "EXCD": excd,
182:             "SYMB": ticker,
183:         }
184:         self.limiter_get.wait()
185:         try:
186:             r = requests.get(url, headers=headers, params=params, timeout=5)
187:             if r.status_code == 200:
188:                 data = r.json()
189:                 if data.get("rt_cd") != "0":
190:                     logger.error(
191:                         f"[KIS-US] Price API Error for {ticker}: {data.get('msg1')}"
192:                     )
193:                     return None
194: 
195:                 output = data.get("output")
196:                 if output:
197:                     last_str = str(output.get("last", "0"))
198:                     if not last_str.strip():
199:                         return None
200:                     return float(last_str)
201:             return None
202:         except Exception as e:
203:             logger.error(f"[KIS-US] Price Inquiry Error for {ticker}: {e}")
204:             return None
205: 
206:     def get_us_ohlcv_recent(
207:         self, ticker: str, exchange: str = "NAS"
208:     ) -> Optional[Dict[str, Any]]:
209:         """Inquire daily OHLCV for recent days (US)."""
210:         url = f"{self.base_url}/uapi/overseas-price/v1/quotations/dailyprice"
211:         headers = self.headers.copy()
212:         headers["tr_id"] = "HHDFS76240000"
213: 
214:         excd = exchange.upper()
215:         params = {
216:             "AUTH": "",
217:             "EXCD": excd,
218:             "SYMB": ticker,
219:             "GUBN": "0",  # 0: Daily, 1: Weekly, 2: Monthly
220:             "BYMD": "",  # Empty for most recent
221:             "MODP": "1",  # 1: Adjusted Price
222:         }
223:         self.limiter_get.wait()
224:         try:
225:             r = requests.get(url, headers=headers, params=params, timeout=5)
226:             if r.status_code == 200:
227:                 data = r.json()
228:                 if data.get("rt_cd") != "0":
229:                     logger.error(
230:                         f"[KIS-US] OHLCV API Error for {ticker}: {data.get('msg1')}"
231:                     )
232:                 return data
233:             return None
234:         except Exception as e:
235:             logger.error(f"[KIS-US] OHLCV Inquiry Error for {ticker}: {e}")
236:             return None
237: 
238:     def get_intraday_bars(
239:         self, ticker: str, period: str = "5"
240:     ) -> Optional[List[Dict[str, Any]]]:
241:         """
242:         국내 주식 당일 분봉 조회
243: 
244:         TR_ID: FHKST03010200
245: 
246:         Args:
247:             ticker: 종목코드 (예: "005930")
248:             period: 분봉 주기 ("1", "3", "5", "10", "15", "30")
249: 
250:         Returns:
251:             [
252:                 {
253:                     "stck_bsop_date": "20260130",
254:                     "stck_cntg_hour": "0905",
255:                     "stck_prpr": "60200",
256:                     "stck_oprc": "60000",
257:                     "stck_hgpr": "60500",
258:                     "stck_lwpr": "59800",
259:                     "cntg_vol": "12345"
260:                 },
261:                 ...
262:             ]
263:         """
264:         url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
265:         headers = self.headers.copy()
266:         headers["tr_id"] = "FHKST03010200"
267: 
268:         params = {
269:             "FID_COND_MRKT_DIV_CODE": "J",
270:             "FID_INPUT_ISCD": ticker,
271:             "FID_PERIOD_DIV_CODE": period,
272:             "FID_INPUT_HOUR_1": "090000",  # 조회 시작 시간
273:         }
274: 
275:         self.limiter_get.wait()
276:         try:
277:             r = requests.get(url, headers=headers, params=params, timeout=10)
278:             if r.status_code == 200:
279:                 data = r.json()
280:                 if data.get("rt_cd") == "0":
281:                     return data.get("output2", [])
282:                 else:
283:                     logger.error(
284:                         f"[KIS] Intraday bars error for {ticker}: {data.get('msg1')}"
285:                     )
286:             return None
287:         except Exception as e:
288:             logger.error(f"[KIS] Intraday bars error for {ticker}: {e}")
289:             return None
290: 
291:     def get_us_intraday_bars(
292:         self, ticker: str, exchange: str = "NAS", period: str = "1"
293:     ) -> Optional[List[Dict[str, Any]]]:
294:         """
295:         미국 주식 당일 분봉 조회
296: 
300:             ticker: 종목코드 (예: "AAPL")
301:             exchange: 거래소 ("NAS", "NYS", "AMS")
302:             period: 분봉 주기 (KIS는 "1"만 제공)
303: 
304:         Returns:
305:             [
306:                 {
307:                     "xymd": "20260129",
308:                     "xhms": "093000",
309:                     "open": "150.25",
310:                     "high": "150.50",
311:                     "low": "150.10",
312:                     "last": "150.30",
313:                     "evol": "12345"
314:                 },
315:                 ...
316:             ]
317:         """
318:         url = f"{self.base_url}/uapi/overseas-stock/v1/quotations/inquire-time-itemchartprice"
319:         headers = self.headers.copy()
320:         headers["tr_id"] = "HHDFS76950200"
321: 
322:         params = {
323:             "AUTH": "",
324:             "EXCD": exchange.upper(),
325:             "SYMB": ticker,
326:             "NMIN": period,
327:             "PINC": "0",
328:             "NEXT": "",
329:             "NREC": "120",  # 최대 조회 건수
330:         }
331: 
332:         self.limiter_get.wait()
333:         try:
334:             r = requests.get(url, headers=headers, params=params, timeout=10)
335:             if r.status_code == 200:
336:                 data = r.json()
337:                 if data.get("rt_cd") == "0":
338:                     return data.get("output2", [])
339:                 else:
340:                     logger.error(
341:                         f"[KIS-US] Intraday bars error for {ticker}: {data.get('msg1')}"
342:                     )
343:             return None
344:         except Exception as e:
345:             logger.error(f"[KIS-US] Intraday bars error for {ticker}: {e}")
346:             return None
347: 
348:     def call_get(
349:         self, url: str, headers: Dict[str, str], params: Dict[str, Any]
350:     ) -> requests.Response:
351:         """Rate-limited GET request."""
352:         self.limiter_get.wait()
353:         return requests.get(url, headers=headers, params=params, timeout=10)
354: 
355:     def call_post(
356:         self, url: str, headers: Dict[str, str], json: Dict[str, Any]
357:     ) -> requests.Response:
358:         """Rate-limited POST request."""
359:         self.limiter_post.wait()
360:         return requests.post(url, headers=headers, json=json, timeout=10)
361: ```
362: 
363: ---
364: 
365: ## 🚀 8. `signal_mailer/kr_stock_scanner.py` (121 lines)
366: ```python
367: 001: # -*- coding: utf-8 -*-
368: 002: import logging
369: 003: import json
370: 004: import os
371: 005: import pandas as pd
372: 006: from concurrent.futures import ThreadPoolExecutor, as_completed
373: 007: from typing import List, Dict, Any, Optional
374: 008: from signal_mailer.kis_api_wrapper import KISAPIWrapper
375: 009: 
376: 010: logger = logging.getLogger(__name__)
377: 011: 
378: 012: 
379: 013: class KRStockScanner:
380: 014:     """
381: 015:     Live Scanner for KOSPI/KOSDAQ stocks using Hybrid Alpha logic.
382: 016:     Logic: (Close > SMA_5) AND (ROC_1 > 0)
383: 017:     DEPENDENCY: KIS API Only (pykrx dependency removed for reliability)
384: 018:     """
385: 019: 
386: 020:     def __init__(self, kis_wrapper: KISAPIWrapper):
387: 021:         self.kis = kis_wrapper
388: 022:         self.universe_path = os.path.join(
389: 023:             os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
390: 024:             "data",
391: 025:             "kr_universe.json",
392: 026:         )
393: 027:         self.universe: List[Dict[str, str]] = []
394: 028:         self._load_universe()
395: 029: 
396: 030:     def _load_universe(self) -> None:
397: 031:         """Load the pre-generated stock universe."""
398: 032:         if os.path.exists(self.universe_path):
399: 033:             try:
400: 034:                 with open(self.universe_path, "r", encoding="utf-8") as f:
401: 035:                     self.universe = json.load(f)
402: 036:                 logger.info(f"Loaded {len(self.universe)} tickers from universe cache.")
403: 037:             except Exception as e:
404: 038:                 logger.error(f"Failed to load universe: {e}")
405: 039:         else:
406: 040:             logger.warning(
407: 041:                 f"Universe file not found at {self.universe_path}. Scan might be limited."
408: 042:             )
409: 043: 
410: 044:     def get_universe(self, market: str = "ALL") -> List[str]:
411: 045:         """Fetch all active tickers for KOSPI and KOSDAQ."""
412: 046:         # This method is now deprecated or needs re-evaluation given the new universe loading.
413: 047:         # For now, returning tickers from the loaded universe.
414: 048:         return [item["ticker"] for item in self.universe]
415: 049: 
416: 050:     def scan_full_market(self, limit: int = 200) -> List[Dict[str, Any]]:
417: 051:         """
418: 052:         Scan the market using the cached universe and KIS API.
419: 053:         """
420: 054:         # 1. Select Tickers from Universe
421: 055:         active_tickers = self.universe[:limit]
422: 056:         if not active_tickers:
423: 057:             logger.error("No active tickers to scan.")
424: 058:             return []
425: 059: 
426: 060:         logger.info(
427: 061:             f"Scanning {len(active_tickers)} tickers using Hybrid Alpha (KIS Only)..."
428: 062:         )
429: 063: 
430: 064:         # 2. Parallel Logic Check
431: 065:         candidates = []
432: 066:         with ThreadPoolExecutor(max_workers=15) as executor:
433: 067:             future_to_info = {
434: 068:                 executor.submit(self._check_logic, item): item
435: 069:                 for item in active_tickers
436: 070:             }
437: 071:             for future in as_completed(future_to_info):
438: 072:                 res = future.result()
439: 073:                 if res:
440: 074:                     candidates.append(res)
441: 075: 
442: 076:         # 3. Sort by ROC_1 descending
443: 077:         candidates.sort(key=lambda x: x["roc_1"], reverse=True)
444: 078:         return candidates
445: 079: 
446: 080:     def _check_logic(self, item: Dict[str, str]) -> Optional[Dict[str, Any]]:
447: 081:         """Check logic for a ticker info dictionary."""
448: 082:         ticker = item["ticker"]
449: 083:         name = item["name"]
450: 084:         try:
451: 085:             # 1. Get recent OHLCV
452: 086:             res = self.kis.get_ohlcv_recent(ticker)
453: 087:             if not res or "output" not in res:
454: 088:                 return None
455: 089: 
456: 090:             output = res["output"]
457: 091:             if not isinstance(output, list) or len(output) < 5:
458: 092:                 # Handle cases where KIS returns different structure or fewer bars
459: 093:                 return None
460: 094: 
461: 095:             # output is usually sorted by date desc
462: 096:             today = output[0]
463: 097:             prev = output[1]
464: 098: 
465: 099:             curr_price = float(today["stck_clpr"])
466: 100:             prev_price = float(prev["stck_clpr"])
467: 101: 
468: 102:             # SMA_5 calculation
469: 103:             last_5 = [float(x["stck_clpr"]) for x in output[:5]]
470: 104:             sma_5 = sum(last_5) / 5
471: 105: 
472: 106:             # Logic: Close > SMA_5 AND Close > PrevClose (ROC_1 > 0)
473: 107:             if curr_price > sma_5 and curr_price > prev_price:
474: 108:                 return {
475: 109:                     "ticker": ticker,
476: 110:                     "name": name,
477: 111:                     "price": curr_price,
478: 112:                     "sma_5": sma_5,
479: 113:                     "roc_1": (curr_price - prev_price)
480: 114:                     / (prev_price if prev_price != 0 else 1),
481: 115:                     "dist_sma": (curr_price - sma_5) / (sma_5 if sma_5 != 0 else 1),
482: 116:                 }
483: 117:         except Exception as e:
484: 118:             logger.debug(f"Error checking {ticker}: {e}")
485: 119: 
486: 120:         return None
487: 121: ```
 *(...세션 연결 중)*
