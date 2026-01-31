"""
섹터 다변화 분석 (v3.1 Week 3)
현재 GNN 포트폴리오의 섹터 편중도 측정 및 개선 제안
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# GNN 티커 및 섹터 정보
GNN_PORTFOLIO = {
    "AAPL": "Technology - Hardware",
    "MSFT": "Technology - Software",
    "GOOGL": "Technology - Internet",
    "AMZN": "Consumer Cyclical - E-commerce",
    "META": "Technology - Social Media",
    "NVDA": "Technology - Semiconductors",
    "TSLA": "Consumer Cyclical - Automotive",
    "NFLX": "Communication Services - Streaming",
    "AVGO": "Technology - Semiconductors",
}


def analyze_sector_concentration():
    """섹터 편중도 분석"""
    print("\n" + "=" * 80)
    print("섹터 다변화 분석 - GNN 포트폴리오")
    print("=" * 80)

    # 섹터별 분류
    df = pd.DataFrame(list(GNN_PORTFOLIO.items()), columns=["Ticker", "Sector"])

    # 메인 섹터 추출
    df["Main_Sector"] = df["Sector"].apply(lambda x: x.split(" - ")[0])

    # 섹터별 종목 수
    sector_counts = df["Main_Sector"].value_counts()

    print("\n📊 섹터별 분포:\n")
    for sector, count in sector_counts.items():
        pct = count / len(df) * 100
        print(f"  {sector:25s}: {count}개 ({pct:5.1f}%)")

    # 편중도 분석
    print("\n⚠️  편중도 분석:\n")

    tech_count = sector_counts.get("Technology", 0)
    tech_pct = tech_count / len(df) * 100

    print(f"  Technology 섹터: {tech_count}/9 ({tech_pct:.1f}%)")

    if tech_pct > 60:
        print(f"  ❌ 높은 편중도! Technology 섹터가 {tech_pct:.0f}% 차지")
        print(f"  리스크: 기술주 동반 하락 시 포트폴리오 전체 타격")
    elif tech_pct > 50:
        print(f"  ⚠️  중간 편중도. Technology 섹터가 {tech_pct:.0f}% 차지")
    else:
        print(f"  ✅ 양호한 분산")

    # 서브섹터 분포
    print("\n📈 세부 섹터 분포:\n")
    subsector_counts = df["Sector"].value_counts()
    for subsector, count in subsector_counts.items():
        print(f"  {subsector:40s}: {count}개")

    return df, sector_counts


def suggest_diversification():
    """다변화 개선 제안"""
    print("\n" + "=" * 80)
    print("💡 섹터 다변화 개선 제안")
    print("=" * 80)

    print("\n현재 문제점:")
    print("  • Technology 섹터 66.7% (6/9 종목)")
    print("  • 반도체 중복 (NVDA, AVGO)")
    print("  • 헬스케어, 금융, 에너지 섹터 부재")

    print("\n제안 1: 보수적 개선 (GNN 티커 유지)")
    print("  현재 9개 티커를 유지하되, 향후 확장 시 고려:")
    print("    • Healthcare: JNJ, UNH")
    print("    • Financials: JPM, V")
    print("    • Energy: XOM")

    print("\n제안 2: 적극적 개선 (일부 교체)")
    print("  Technology 비중 축소 (6개 → 4개):")
    print("    교체 후보:")
    print("      AVGO → JNJ (Healthcare)")
    print("      NFLX → V (Financials)")
    print("    결과: Tech 44%, Healthcare 11%, Financials 11%")

    print("\n제안 3: 섹터 ETF 추가 (하이브리드)")
    print("  개별주 GNN + 섹터 ETF 방어:")
    print("    • GNN: 현재 9개 유지 (70%)")
    print("    • Defensive: XLV (Healthcare ETF) 15%")
    print("    • Defensive: XLE (Energy ETF) 15%")

    print("\n권장 방향:")
    print("  ✅ 제안 1 채택 (현재 유지, 향후 확장 준비)")
    print("  이유:")
    print("    • 현재 GNN 모델은 9개 티커 최적화")
    print("    • 티커 변경 시 재학습 필요")
    print("    • 백테스트 성과 우수 (CAGR 29.74%)")
    print("    • v4.0에서 섹터 다변화 본격 도입")


def visualize_sectors(df, sector_counts):
    """섹터 분포 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 메인 섹터 파이 차트
    ax = axes[0]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    ax.pie(
        sector_counts.values,
        labels=sector_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 11},
    )
    ax.set_title("Main Sector Distribution", fontsize=14, weight="bold")

    # 세부 섹터 바 차트
    ax = axes[1]
    subsector_counts = df["Sector"].value_counts()
    subsector_counts.plot(kind="barh", ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel("Number of Stocks", fontsize=12)
    ax.set_title("Detailed Sector Breakdown", fontsize=14, weight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("d:/gg/sector_diversification.png", dpi=150, bbox_inches="tight")
    print(f"\n📊 시각화 저장: d:/gg/sector_diversification.png")

    return fig


if __name__ == "__main__":
    # 섹터 분석
    df, sector_counts = analyze_sector_concentration()

    # 다변화 제안
    suggest_diversification()

    # 시각화
    fig = visualize_sectors(df, sector_counts)

    print("\n" + "=" * 80)
    print("섹터 다변화 분석 완료!")
    print("=" * 80)
