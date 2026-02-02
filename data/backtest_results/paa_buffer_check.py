import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 다운로드
# PAA를 위해 여러 자산이 필요하며, 버퍼 전략 비교를 위해 SPY를 기준으로 삼습니다.
symbols = ["SPY", "VEA", "VWO", "BND"]  # PAA 공격 자산 예시
print("Downloading data...")
# auto_adjust=True is often better for backtesting
data = yf.download(symbols, start="2020-01-01", progress=False)

# Data column handling (from previous robust fix)
if "Adj Close" in data.columns:
    data = data["Adj Close"]
elif "Close" in data.columns:
    print("'Adj Close' not found, using 'Close'")
    data = data["Close"]
else:
    # Handle MultiIndex case where levels might be different
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            data = data.xs("Adj Close", level=0, axis=1)
        elif "Close" in data.columns.get_level_values(0):
            data = data.xs("Close", level=0, axis=1)
        else:
            print(
                "Warning: Could not find 'Adj Close' or 'Close' in columns via direct check. Using raw download."
            )
            # Fallback might be needed or data is already correct


# Handle single symbol case just in case
if isinstance(data, pd.Series):
    data = data.to_frame()


def calculate_paa_signal(data, window=252):
    """PAA Breadth 신호 계산 (12개월 모멘텀 점수 기반)"""
    # 12개월(약 252일) 모멘텀: 현재가 / 12개월 전 가격 - 1
    momentum = data.pct_change(window)

    # Breadth: 모멘텀이 0보다 큰 자산의 개수
    breadth = (momentum > 0).sum(axis=1)
    total_assets = len(data.columns)

    # 안전 자산 비중 결정 (단순화: 절반 미만이 양수면 현금화 신호)
    # 실제 PAA는 (bad_assets / total_assets * protection_factor) 로 계산함
    paa_signal = np.where(breadth < (total_assets / 2), 0, 1)
    return pd.Series(paa_signal, index=data.index)


def calculate_buffer_ma_signal(price, ma_window=185, buffer=0.03):
    """
    185일 이평선 + 3% 버퍼 신호 계산
    (상단 밴드 돌파 시 매수, 하단 밴드 이탈 시 매도, 그 외 유지)
    """
    ma = price.rolling(window=ma_window).mean()

    current_state = 1  # 1: 매수(보유), 0: 매도(현금) 초기 상태 가정

    # Optimizing loop access using numpy arrays for speed
    price_values = price.values
    ma_values = ma.values
    signal_values = np.zeros(len(price))

    for i in range(len(price)):
        if np.isnan(ma_values[i]):
            signal_values[i] = 0
            continue

        upper_band = ma_values[i] * (1 + buffer)
        lower_band = ma_values[i] * (1 - buffer)

        if price_values[i] > upper_band:
            current_state = 1
        elif price_values[i] < lower_band:
            current_state = 0

        signal_values[i] = current_state

    return pd.Series(signal_values, index=price.index)


# 2. 신호 계산
if "SPY" not in data.columns:
    print("Error: SPY data not found. Columns:", data.columns)
else:
    spy_price = data["SPY"]
    paa_sig = calculate_paa_signal(data)
    buffer_sig = calculate_buffer_ma_signal(spy_price)

    # 3. 시각화
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(spy_price, label="SPY Price", color="gray", alpha=0.5)
    plt.title("Market Price & Strategy Signals")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.fill_between(
        paa_sig.index,
        0,
        paa_sig,
        alpha=0.3,
        label="PAA Signal (1=In, 0=Out)",
        color="blue",
    )
    plt.fill_between(
        buffer_sig.index,
        0,
        buffer_sig,
        alpha=0.3,
        label="185MA+3% Buffer (1=In, 0=Out)",
        color="red",
    )
    plt.ylim(-0.1, 1.1)
    plt.title("Signal Timing Comparison")
    plt.legend(loc="center right")

    plt.tight_layout()
    print("Plotting done. Saving plot...")
    plt.savefig("d:/gg/data/backtest_results/paa_buffer_result.png")
    print("Saved plot to d:/gg/data/backtest_results/paa_buffer_result.png")
    plt.show()
