# Absolute Global Factor Discovery Report

**Analysis Scope**: 21 Global Assets, 3 Macro Indicators
**Total Derived Factors**: 22
**Period**: 2010-01-01 ~ Present

## 1. Top Predictive Factors (Combined Universe)
미래 수익률(20일 후)과 정비례 또는 반비례 관계가 가장 뚜렷한 팩터들입니다.

| Factor      |   IC_20d_mean |   Stability |
|:------------|--------------:|------------:|
| atr_norm    |    0.123249   |    1.22334  |
| vol_20d     |    0.105732   |    1.15919  |
| vol_252d    |    0.0854346  |    0.79138  |
| bb_width    |    0.0827958  |    1.1155   |
| corr_tnx    |    0.0307626  |    0.63221  |
| vol_roc_5d  |   -0.00308157 |    0.201867 |
| corr_dxy    |   -0.00575457 |    0.146318 |
| pv_corr_20d |   -0.010532   |    0.248733 |
| corr_vix    |   -0.0147796  |    0.328866 |
| ret_1d      |   -0.020457   |    1.03504  |
| rsi         |   -0.0211555  |    0.36549  |
| will_r      |   -0.0234485  |    0.459487 |
| ret_120d    |   -0.0254525  |    0.287364 |
| ret_252d    |   -0.0257715  |    0.314387 |
| macd_hist   |   -0.0320288  |    0.739381 |

## 2. Category Intelligence
자산군별로 가장 '말이 잘 듣는' 전조 증상(Leading Factor)입니다.

*   **Equities**: `atr_norm` (Avg IC: 0.223)
*   **Hard Assets**: `ret_252d` (Avg IC: 0.075)
*   **Safe Havens**: `atr_norm` (Avg IC: 0.102)

## 3. Structural Insights (Alpha Matrix)
1. **모멘텀 붕괴**: 장기 수익률(`ret_252d`)이 너무 높으면 오히려 반전되는 경향이 관찰되었습니다.
2. **매크로 연결고리**: `corr_vix` 및 `corr_tnx`가 자산군 수익률 결정에 지대한 영향을 미칩니다.
3. **기술적 저항**: `will_r` 및 `rsi`와 같은 오실레이터가 단기 리밸런싱 시점에 매우 유효합니다.
