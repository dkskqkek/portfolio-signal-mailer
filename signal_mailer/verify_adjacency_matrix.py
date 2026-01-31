"""
Adjacency Matrix 검증 스크립트 (v3.1 Week 2)
현재 인접 행렬과 실제 상관계수 비교 분석
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# GNN 티커
GNN_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AVGO"]


def load_adjacency_matrix():
    """현재 adjacency matrix 로드"""
    adj = pd.read_csv("d:/gg/data/gnn/adjacency_matrix.csv", index_col=0)
    return adj


def calculate_correlation_matrix(lookback_days=252):
    """실제 상관계수 계산 (최근 1년)"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 50)

    print(f"📥 Downloading data from {start_date.date()} to {end_date.date()}...")
    data = yf.download(GNN_TICKERS, start=start_date, end=end_date, progress=False)[
        "Close"
    ]

    # 일별 수익률
    returns = data.pct_change().dropna()

    # 상관계수 행렬
    corr = returns.corr()

    return corr


def analyze_differences(adj, corr):
    """차이점 분석"""
    print("\n" + "=" * 80)
    print("ADJACENCY MATRIX vs CORRELATION MATRIX 비교")
    print("=" * 80)

    # 1. 연결 vs 비연결 비교
    print("\n📊 연결 패턴 분석\n")

    for ticker in GNN_TICKERS:
        # Adjacency에서 연결된 티커들
        connected = adj.loc[ticker][adj.loc[ticker] == 1].index.tolist()
        if ticker in connected:
            connected.remove(ticker)  # 자기 자신 제거

        # 상관계수 높은 티커들 (>0.5)
        high_corr = corr.loc[ticker][corr.loc[ticker] > 0.5].index.tolist()
        if ticker in high_corr:
            high_corr.remove(ticker)

        print(f"{ticker:6s}:")
        print(f"  현재 연결:      {connected}")
        print(f"  높은 상관(>0.5): {high_corr}")

        # 누락된 연결 찾기
        missing = set(high_corr) - set(connected)
        if missing:
            print(f"  ⚠️  누락 가능:    {list(missing)}")

        # 불필요한 연결 찾기
        unnecessary = set(connected) - set(high_corr)
        if unnecessary:
            print(f"  ⚠️  약한 연결:    {list(unnecessary)}")
        print()

    # 2. TSLA 특별 분석
    print("\n🚗 TSLA 연결 상태 상세 분석\n")
    tsla_connections = adj.loc["TSLA"][adj.loc["TSLA"] == 1].index.tolist()
    tsla_correlations = corr.loc["TSLA"].sort_values(ascending=False)

    print(f"현재 TSLA 연결: {tsla_connections}")
    print(f"\nTSLA 상관계수 (상위 5개):")
    for ticker, corr_val in tsla_correlations.head(6).items():
        status = "✓ 연결됨" if ticker in tsla_connections else "✗ 미연결"
        print(f"  {ticker:6s}: {corr_val:.3f}  {status}")

    # 3. 전체 통계
    print("\n📈 전체 통계\n")

    # Adjacency에서 연결 수
    total_connections = (
        adj.sum().sum() - len(GNN_TICKERS)
    ) / 2  # 양방향이므로 /2, 자기연결 제외
    possible_connections = len(GNN_TICKERS) * (len(GNN_TICKERS) - 1) / 2

    print(f"현재 연결 수:     {int(total_connections)}/{int(possible_connections)}")
    print(f"연결 밀도:        {total_connections / possible_connections:.1%}")

    # 상관계수 >0.5인 쌍 수
    high_corr_pairs = 0
    for i in range(len(GNN_TICKERS)):
        for j in range(i + 1, len(GNN_TICKERS)):
            if corr.iloc[i, j] > 0.5:
                high_corr_pairs += 1

    print(f"높은 상관(>0.5):  {high_corr_pairs}/{int(possible_connections)}")
    print(f"상관 밀도:        {high_corr_pairs / possible_connections:.1%}")


def visualize_comparison(adj, corr):
    """시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Adjacency Matrix
    sns.heatmap(
        adj,
        annot=True,
        fmt="g",
        cmap="YlGnBu",
        ax=axes[0],
        square=True,
        cbar_kws={"label": "Connected"},
    )
    axes[0].set_title("Current Adjacency Matrix (Binary)", fontsize=14, weight="bold")

    # Correlation Matrix
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        vmin=-0.5,
        vmax=1.0,
        ax=axes[1],
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    axes[1].set_title(
        "Actual Correlation Matrix (252 days)", fontsize=14, weight="bold"
    )

    plt.tight_layout()
    plt.savefig("d:/gg/adjacency_vs_correlation.png", dpi=150, bbox_inches="tight")
    print(f"\n📊 Visualization saved: d:/gg/adjacency_vs_correlation.png")

    return fig


def suggest_improvements(adj, corr, threshold=0.5):
    """개선 제안"""
    print("\n" + "=" * 80)
    print("💡 ADJACENCY MATRIX 개선 제안")
    print("=" * 80)

    suggestions = []

    for i, ticker1 in enumerate(GNN_TICKERS):
        for j, ticker2 in enumerate(GNN_TICKERS):
            if i >= j:  # 대각선 및 하삼각 제외
                continue

            current_connection = adj.loc[ticker1, ticker2]
            correlation = corr.loc[ticker1, ticker2]

            # 높은 상관인데 연결 안 됨
            if correlation > threshold and current_connection == 0:
                suggestions.append(
                    {
                        "type": "ADD",
                        "pair": f"{ticker1}-{ticker2}",
                        "correlation": correlation,
                        "reason": f"상관계수 {correlation:.3f} > {threshold}",
                    }
                )

            # 낮은 상관인데 연결됨
            elif correlation < threshold and current_connection == 1:
                suggestions.append(
                    {
                        "type": "REMOVE",
                        "pair": f"{ticker1}-{ticker2}",
                        "correlation": correlation,
                        "reason": f"상관계수 {correlation:.3f} < {threshold}",
                    }
                )

    if suggestions:
        print(f"\n추천 임계값: {threshold}")

        add_suggestions = [s for s in suggestions if s["type"] == "ADD"]
        remove_suggestions = [s for s in suggestions if s["type"] == "REMOVE"]

        if add_suggestions:
            print(f"\n➕ 연결 추가 제안 ({len(add_suggestions)}개):")
            for s in sorted(add_suggestions, key=lambda x: -x["correlation"])[:5]:
                print(f"  {s['pair']:15s}  correlation: {s['correlation']:.3f}")

        if remove_suggestions:
            print(f"\n➖ 연결 제거 제안 ({len(remove_suggestions)}개):")
            for s in sorted(remove_suggestions, key=lambda x: x["correlation"])[:5]:
                print(f"  {s['pair']:15s}  correlation: {s['correlation']:.3f}")
    else:
        print("\n✅ 현재 Adjacency Matrix가 상관계수를 잘 반영하고 있습니다!")

    return suggestions


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ADJACENCY MATRIX 검증 (v3.1 Week 2)")
    print("=" * 80)

    # 1. 데이터 로드
    adj = load_adjacency_matrix()
    corr = calculate_correlation_matrix(lookback_days=252)

    # 2. 차이점 분석
    analyze_differences(adj, corr)

    # 3. 개선 제안
    suggestions = suggest_improvements(adj, corr, threshold=0.5)

    # 4. 시각화
    fig = visualize_comparison(adj, corr)

    print("\n" + "=" * 80)
    print("검증 완료!")
    print("=" * 80)
