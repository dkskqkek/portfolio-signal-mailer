import json
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Define ETF Holdings Data (Approximate Top 10-15 Weights as of Jan 2026)
# Normalized to sum to ~1.0 roughly, or just use top components
# "Residual" (Others) is important for VT. VT has 9000 stocks.
# We focus on the "Active Drivers" (Top 50-100 Global Stocks).

etf_holdings = {
    "SPY": {  # S&P 500
        "MSFT": 0.07,
        "AAPL": 0.07,
        "NVDA": 0.06,
        "AMZN": 0.035,
        "META": 0.025,
        "GOOGL": 0.02,
        "GOOG": 0.02,
        "BRK.B": 0.017,
        "TSLA": 0.015,
        "LLY": 0.013,
        "AVGO": 0.013,
        "JPM": 0.012,
        "XOM": 0.010,
        "UNH": 0.010,
        "V": 0.010,
    },
    "QQQ": {  # Nasdaq 100
        "NVDA": 0.086,
        "AAPL": 0.070,
        "MSFT": 0.064,
        "AMZN": 0.048,
        "TSLA": 0.038,
        "GOOGL": 0.037,
        "META": 0.037,
        "GOOG": 0.034,
        "WMT": 0.030,
        "AVGO": 0.030,
        "COST": 0.024,
        "NFLX": 0.020,
    },
    "VGT": {  # Tech
        "NVDA": 0.180,
        "AAPL": 0.150,
        "MSFT": 0.125,
        "AVGO": 0.050,
        "PLTR": 0.020,
        "AMD": 0.020,
        "ORCL": 0.020,
        "CSCO": 0.015,
        "IBM": 0.014,
        "MU": 0.014,
    },
    "SCHD": {  # Dividend
        "LMT": 0.047,
        "CVX": 0.042,
        "BMY": 0.041,
        "HD": 0.041,
        "COP": 0.041,
        "MO": 0.040,
        "TXN": 0.040,
        "MRK": 0.040,
        "KO": 0.039,
        "AMGN": 0.038,
        "ABBV": 0.035,
        "PEP": 0.035,
    },
    "COWZ": {  # Cash Cow
        "NEM": 0.026,
        "XOM": 0.022,
        "GILD": 0.022,
        "CVX": 0.022,
        "MO": 0.021,
        "CMCSA": 0.021,
        "MRK": 0.021,
        "AMGN": 0.020,
        "DIS": 0.020,
        "ACN": 0.020,
    },
    "XLV": {  # Healthcare
        "LLY": 0.148,
        "JNJ": 0.092,
        "ABBV": 0.067,
        "UNH": 0.055,
        "MRK": 0.047,
        "TMO": 0.041,
        "ABT": 0.035,
        "AMGN": 0.032,
        "ISRG": 0.032,
        "GILD": 0.029,
        "PFE": 0.025,
    },
    "VXUS": {  # Intl
        "TSM": 0.030,
        "TCEHY": 0.012,
        "ASML": 0.011,
        "005930.KS": 0.010,  # Samsung
        "BABA": 0.008,
        "RHHBY": 0.007,
        "AZN": 0.007,
        "HSBC": 0.007,
        "NVO": 0.007,
        "SHEL": 0.007,
        "SAP": 0.006,
        "TM": 0.006,
    },
    "GLD": {  # Gold
        "GOLD_BULLION": 1.0
    },
    "VT": {  # Global Benchmark
        "NVDA": 0.041,
        "AAPL": 0.038,
        "MSFT": 0.034,
        "AMZN": 0.021,
        "GOOGL": 0.018,
        "AVGO": 0.015,
        "GOOG": 0.014,
        "META": 0.014,
        "TSLA": 0.012,
        "TSM": 0.011,
        "LLY": 0.010,
        "BRK.B": 0.009,
        "JPM": 0.008,
        "XOM": 0.007,
        "UNH": 0.006,
        "V": 0.006,
        "WMT": 0.005,
        "MA": 0.005,
        "PG": 0.005,
        "JNJ": 0.005,
        "HD": 0.004,
        "COST": 0.004,
        "ABBV": 0.004,
        "MRK": 0.004,
    },
}

# Add Korean ETF Proxies
# KODEX S&P500 -> SPY
# KODEX Nasdaq100 -> QQQ
# KODEX Korea Value -> Generic Korea (use partial VXUS or simplified Samsung?)
# Let's map Korea specific to Samsung + Hynix or just leave as "Diverse Korea"
# For now, map 'EWY' style exposure?
# Let's treat "Korea ETFs" as 20% Samsung, 5% Hynix, 75% Other for simplicity?
etf_holdings["KOREA_proxy"] = {
    "005930.KS": 0.20,
    "000660.KS": 0.05,
    "005380.KS": 0.02,
    "POSCO": 0.02,
    "NAVER": 0.02,
    "KAKAO": 0.01,
    "KB": 0.02,
    "SHINHAN": 0.02,
}

# 2. Load Portfolio State
try:
    with open("d:/gg/data/portfolio_state.json", "r", encoding="utf-8") as f:
        state = json.load(f)
except Exception as e:
    print(f"Error loading state: {e}")
    exit()

# 3. Build User Portfolio Vector
user_vector = {}
total_val = 0

# Ticker to Holdings Key Map
asset_map = {
    "KODEX US S&P500": "SPY",
    "KODEX US Nasdaq100": "QQQ",
    "KODEX Korea Value": "KOREA_proxy",  # Rough proxy
    "ACE Gold": "GLD",
    "TIGER US Div+7%": "SCHD",
    "TIGER Korea Top10": "KOREA_proxy",
    "GOOGL (Alphabet)": "SINGLE:GOOGL",
    "VGT (Tech ETF)": "VGT",
    "COWZ (Cash Cow)": "COWZ",
    "XLV (Healthcare)": "XLV",
    "CVX (Chevron)": "SINGLE:CVX",
    "VXUS (Intl)": "VXUS",
    "GLDM (Gold)": "GLD",
}

# Calculate Total Value First
for acct in state["accounts"]:
    for asset in acct["assets"]:
        total_val += asset["value"]

print(f"Total Portfolio Value: {total_val:,.0f} KRW")

# Aggregate Holdings
for acct in state["accounts"]:
    for asset in acct["assets"]:
        name = asset["ticker"]
        val = asset["value"]
        weight_in_pf = val / total_val

        mapping_key = asset_map.get(name, "SPY")  # Default to SPY if unknown

        if mapping_key.startswith("SINGLE:"):
            stock = mapping_key.split(":")[1]
            user_vector[stock] = user_vector.get(stock, 0) + weight_in_pf
        else:
            # It's an ETF
            holdings = etf_holdings.get(mapping_key, {})
            # We assume the ETF holdings sum to < 1.0 (Top 10 only)
            # This is "Known Exposure". The rest is "Unknown Alpha/Beta".
            for stock, w_in_etf in holdings.items():
                effective_w = weight_in_pf * w_in_etf
                user_vector[stock] = user_vector.get(stock, 0) + effective_w

# 4. Compare with VT
vt_vector = etf_holdings["VT"]

# Create Union of Keys
all_stocks = set(user_vector.keys()) | set(vt_vector.keys())

# Build Vectors
vec_user = []
vec_vt = []
labels = []

for stock in all_stocks:
    vec_user.append(user_vector.get(stock, 0.0))
    vec_vt.append(vt_vector.get(stock, 0.0))
    labels.append(stock)

# Cosine Similarity
from numpy.linalg import norm

v_u = np.array(vec_user)
v_v = np.array(vec_vt)

# Handle zero vector case
if norm(v_u) == 0 or norm(v_v) == 0:
    sim = 0
else:
    sim = np.dot(v_u, v_v) / (norm(v_u) * norm(v_v))

print(f"\n🧩 **Holdings-Based Similarity (vs VT)**: {sim:.3f}")

# Top Overweights / Underweights
diffs = []
for i, stock in enumerate(labels):
    u_w = v_u[i]
    v_w = v_v[i]
    diff = u_w - v_w
    diffs.append({"Stock": stock, "User": u_w, "VT": v_w, "Diff": diff})

df = pd.DataFrame(diffs)
print("\n[Top Overweights (vs VT)]")
print(
    df.sort_values("Diff", ascending=False)
    .head(10)[["Stock", "User", "VT", "Diff"]]
    .to_string()
)

print("\n[Top Underweights (vs VT)]")
print(
    df.sort_values("Diff", ascending=True)
    .head(5)[["Stock", "User", "VT", "Diff"]]
    .to_string()
)

# Write Report
md_lines = []
md_lines.append("# 내 포트폴리오 vs VT (전세계 주식) 구성 유사도 분석")
md_lines.append(f"**분석 방식**: ETF 내부 보유 종목(Look-through) 기반 정밀 분석")
md_lines.append(f"**일자**: {datetime.now().strftime('%Y-%m-%d')}")
md_lines.append("")
md_lines.append(f"## 🧩 유사도 결과: `{sim:.3f}`")
if sim > 0.8:
    md_lines.append("- 귀하의 포트폴리오는 **VT(전세계 시장 평균)와 매우 유사**합니다.")
elif sim > 0.6:
    md_lines.append(
        "- 귀하의 포트폴리오는 **기본적으로 시장을 따르지만, 특정 섹터(기술주 등)에 집중**되어 있습니다."
    )
else:
    md_lines.append(
        "- 귀하의 포트폴리오는 **VT와 상당히 다른(액티브한) 구성**을 가지고 있습니다."
    )

md_lines.append("")
md_lines.append("## ⚖️ 주요 종목 비중 비교 (Top Holdings)")
md_lines.append("| 종목 | 내 비중 | VT 비중 | 차이 (Active Share) |")
md_lines.append("| :--- | :--- | :--- | :--- |")

for _, row in df.sort_values("User", ascending=False).head(15).iterrows():
    diff_str = (
        f"+{row['Diff'] * 100:.1f}%" if row["Diff"] > 0 else f"{row['Diff'] * 100:.1f}%"
    )
    md_lines.append(
        f"| {row['Stock']} | {row['User'] * 100:.1f}% | {row['VT'] * 100:.1f}% | {diff_str} |"
    )

md_lines.append("")
md_lines.append("## 🔍 분석 인사이트")
md_lines.append("### 1. 기술주 집중 (High Tech Concentration)")
top_tech = df[df["Stock"].isin(["NVDA", "AAPL", "MSFT", "GOOGL", "AVGO"])]["User"].sum()
vt_tech = df[df["Stock"].isin(["NVDA", "AAPL", "MSFT", "GOOGL", "AVGO"])]["VT"].sum()
md_lines.append(
    f"- 주요 빅테크 종목 비중 합계: **{top_tech * 100:.1f}%** (VT: {vt_tech * 100:.1f}%)"
)
md_lines.append("- VGT와 QQQ의 영향으로 VT 대비 기술주 비중이 상당히 높습니다.")

md_lines.append("### 2. 가치주/배당주 보완")
top_val = df[df["Stock"].isin(["CVX", "XOM", "LLY", "UNH", "JNJ"])]["User"].sum()
md_lines.append(f"- 헬스케어/에너지 가치주 비중: **{top_val * 100:.1f}%**")
md_lines.append("- SCHD, COWZ, XLV가 성장주 쏠림을 완화해주고 있습니다.")

md_lines.append("### 3. 금(Gold) 자산")
gold_w = user_vector.get("GOLD_BULLION", 0)
md_lines.append(f"- 금 비중: **{gold_w * 100:.1f}%**")
md_lines.append("- VT에는 없는 대체 자산으로, 포트폴리오의 안정성을 높이는 요소입니다.")

with open("d:/gg/research/vt_similarity_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("Report saved to d:/gg/research/vt_similarity_report.md")
