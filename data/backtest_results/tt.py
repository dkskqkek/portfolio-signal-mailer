import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import yfinance as yf

# ---------------------------------------------------------
# 전역 변수를 통한 모듈 캐싱 (Lazy Loading 최적화)
# ---------------------------------------------------------
_MAMA_PREDICTOR = None


def _get_mama_predictor():
    """
    MAMA 예측기를 최초 1회만 로드하고 이후에는 재사용합니다.
    매일 반복되는 import 오버헤드를 방지합니다.
    """
    global _MAMA_PREDICTOR
    if _MAMA_PREDICTOR is None:
        try:
            from mama_lite_predictor import MAMAPredictor

            _MAMA_PREDICTOR = MAMAPredictor()
        except ImportError:
            # For debugging/fallback if file is missing
            print("Warning: 'mama_lite_predictor.py' not found. Using dummy predictor.")

            class DummyPredictor:
                def predict(self, date, data):
                    return {"SPY": 1.0}

            _MAMA_PREDICTOR = DummyPredictor()

    return _MAMA_PREDICTOR


# ---------------------------------------------------------
# 전략 함수 정의
# ---------------------------------------------------------
def mama_lite_strategy(date, data, current_holdings):
    """
    MAMA Lite 전략: 예측기를 통해 매수/매도 신호를 생성
    """
    # 1. 데이터 유효성 검증 (Silent Failure 방지)
    if data is None or data.empty:
        return {}

    # 2. 예측기 로드 (Cached)
    predictor = _get_mama_predictor()

    # 3. 예측 실행
    try:
        # predictor 내부 로직에 따라 target_weights 반환
        return predictor.predict(date, data)
    except Exception as e:
        # 로직 에러는 숨기지 않고 로그를 남기거나 상위로 전파
        print(f"[Error] Strategy Execution Failed on {date}: {e}")
        return {}  # 비상시에만 빈 딕셔너리 리턴 (혹은 raise)


def sixty_forty_strategy(date, data, current_holdings):
    """
    고전적인 60/40 전략 (주식 60%, 채권 40%)
    """
    return {"SPY": 0.6, "AGG": 0.4}


class BacktestEngine:
    def __init__(self, start_date, end_date, strategy=None, initial_capital=10000):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.strategy = strategy
        self.initial_capital = initial_capital

    def run(self, data_feed):
        """
        백테스트 실행 메인 루프
        data_feed: 날짜별 가격 데이터가 포함된 DataFrame (Index: Date)
        """
        # 데이터 기간 필터링
        dates = data_feed.loc[self.start_date : self.end_date].index

        portfolio = []
        current_holdings = {}
        capital = self.initial_capital

        # 전략 주입 (Dependency Injection)
        strategy = self.strategy
        if strategy is None:
            strategy = mama_lite_strategy

        # 날짜별 루프
        for date in dates:
            # 1. 현재 자산 가치 평가
            # (이번 예시에서는 간단히 당일 종가 기준 평가라고 가정, 실제로는 전일 종가 + 당일 변동 반영 등 필요)
            daily_value = capital  # 현금

            # 보유 주식 가치 합산 (여기서는 간소화된 시뮬레이션을 위해 생략하거나 더미 로직 사용)
            # 실제 백테스팅에서는 체결 가격 관리 필요

            # 2. 전략 실행 (Target Weights 수신)
            historical_data = data_feed.loc[:date]
            target_weights = strategy(date, historical_data, current_holdings)

            # 3. 리밸런싱 로직 (매수/매도 실행 - 생략, 결과만 기록)
            # 실제로는 target_weights에 맞춰 capital을 주식으로 변환

            # 시뮬레이션을 위해 SPY 단순 보유 또는 랜덤 수익률 가정 대신
            # 여기서는 'SPY' 컬럼이 있으면 그걸 추종한다고 가정
            daily_return = 0.0
            if "SPY" in data_feed.columns:
                # 당일 SPY 등락률 적용 (간이 시뮬레이션)
                # 실제로는 target_weights * asset_returns
                if date != dates[0]:
                    try:
                        # MultiIndex handle
                        if isinstance(data_feed.columns, pd.MultiIndex):
                            spy_ret = (
                                data_feed.xs("Close", level=0, axis=1)["SPY"]
                                .pct_change()
                                .loc[date]
                            )
                        else:
                            spy_ret = data_feed["SPY"].pct_change().loc[date]
                        daily_return = spy_ret
                    except:
                        pass

            # 자산 가치 업데이트 (단순화: SPY 100% 보유 가정 테스트)
            capital = capital * (1 + daily_return)
            daily_value = capital

            # 결과 기록
            portfolio.append({"Date": date, "Total Value": daily_value})

        # DataFrame 변환
        df_result = pd.DataFrame(portfolio).set_index("Date")

        # 일별 수익률 계산
        df_result["Daily_Return"] = df_result["Total Value"].pct_change().fillna(0)

        # 성과 지표 계산
        metrics = self.calculate_metrics(df_result)

        return metrics, df_result

    def calculate_metrics(self, df):
        """
        17개 핵심 성과 지표 계산
        """
        returns = df["Daily_Return"]
        total_days = len(df)
        years = total_days / 252.0

        # --- 1. 수익률 관련 ---
        total_return = (df["Total Value"].iloc[-1] / df["Total Value"].iloc[0]) - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Real Return (인플레 3% 가정)
        real_return = ((1 + cagr) / (1 + 0.03)) - 1

        # --- 2. 리스크 지표 ---
        volatility = returns.std() * np.sqrt(252)
        risk_free_rate = 0.04
        excess_return = cagr - risk_free_rate

        sharpe = excess_return / volatility if volatility != 0 else 0

        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252)
        sortino = (
            excess_return / downside_vol
            if downside_vol != 0 and not np.isnan(downside_vol)
            else 0
        )

        # MDD & MDD Duration
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        mdd = drawdown.min()

        is_underwater = drawdown < 0
        mdd_duration = 0
        current_duration = 0
        for u in is_underwater:
            if u:
                current_duration += 1
                mdd_duration = max(mdd_duration, current_duration)
            else:
                current_duration = 0

        # Recovery Period
        mdd_idx = drawdown.idxmin()
        recovery_period = 0
        if mdd_idx != drawdown.index[-1]:
            post_mdd = cumulative_returns.loc[mdd_idx:]
            recovered = post_mdd[post_mdd >= peak.loc[mdd_idx]]
            if not recovered.empty:
                recovery_period = (recovered.index[0] - mdd_idx).days
            else:
                recovery_period = (df.index[-1] - mdd_idx).days  # Not recovered yet

        calmar = cagr / abs(mdd) if mdd != 0 else 0
        ulcer_index = np.sqrt(np.mean(drawdown**2)) * 100

        # --- 3. 거래 성과 ---
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0
        avg_win = winning_days.mean() if not winning_days.empty else 0
        avg_loss = losing_days.mean() if not losing_days.empty else 0
        gross_profit = winning_days.sum()
        gross_loss = abs(losing_days.sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        xirr = cagr  # Approximation

        return {
            "Total Return": total_return,
            "CAGR": cagr,
            "Real Return": real_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": mdd,
            "MDD Duration (Days)": mdd_duration,
            "Recovery Period (Days)": recovery_period,
            "Calmar Ratio": calmar,
            "Ulcer Index": ulcer_index,
            "Win Rate": win_rate,
            "Avg Win": avg_win,
            "Avg Loss": avg_loss,
            "Win/Loss Ratio": win_loss_ratio,
            "Profit Factor": profit_factor,
            "XIRR": xirr,
        }


if __name__ == "__main__":
    # Test execution
    print("Testing BacktestEngine...")

    # Download sample data
    symbols = ["SPY", "AGG"]
    data = yf.download(symbols, start="2020-01-01", progress=False)

    # Handle data structure
    if "Adj Close" in data.columns:
        data = data["Adj Close"]
    elif "Close" in data.columns:
        data = data["Close"]

    engine = BacktestEngine("2021-01-01", "2023-12-31")
    metrics, df_res = engine.run(data)

    print("\nBacktest Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    df_res["Total Value"].plot(title="Backtest Result")
    plt.savefig("d:/gg/data/backtest_results/tt_backtest.png")
    print("Chart saved to d:/gg/data/backtest_results/tt_backtest.png")
