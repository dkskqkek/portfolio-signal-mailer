import json
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# 1. Load Portfolio State
try:
    with open("d:/gg/data/portfolio_state.json", "r", encoding="utf-8") as f:
        state = json.load(f)
except FileNotFoundError:
    print("Error: portfolio_state.json not found.")
    exit()

# 2. Define Ticker Mapping
ticker_map = {
    "KODEX US S&P500": {"primary": "360750.KS", "proxy": "SPY"},
    "KODEX US Nasdaq100": {"primary": "379810.KS", "proxy": "QQQ"},
    "KODEX Korea Value": {"primary": "223190.KS", "proxy": "EWY"},
    "ACE Gold": {"primary": "411060.KS", "proxy": "GLD"},
    "TIGER US Div+7%": {"primary": "458760.KS", "proxy": "SCHD"},
    "TIGER Korea Top10": {"primary": "364960.KS", "proxy": "EWY"},
    "GOOGL (Alphabet)": {"primary": "GOOGL"},
    "VGT (Tech ETF)": {"primary": "VGT"},
    "COWZ (Cash Cow)": {"primary": "COWZ"},
    "XLV (Healthcare)": {"primary": "XLV"},
    "CVX (Chevron)": {"primary": "CVX"},
    "VXUS (Intl)": {"primary": "VXUS"},
    "GLDM (Gold)": {"primary": "GLDM"},
}

# 3. Aggregate Portfolio
asset_values = {}
total_portfolio_value = 0

print("--- Portfolio Composition ---")
for acct in state["accounts"]:
    for asset in acct["assets"]:
        name = asset["ticker"]
        val = asset["value"]

        # Identify Key
        key = name

        if key not in asset_values:
            asset_values[key] = 0
        asset_values[key] += val
        total_portfolio_value += val

print(f"Total Value: {total_portfolio_value:,.0f} KRW")

# 4. Fetch Data
print("\n--- Fetching Market Data ---")
fetch_list = []
asset_ticker_lookup = {}

for name in asset_values.keys():
    map_info = ticker_map.get(name)
    primary = map_info.get("primary") if map_info else name.split(" ")[0]
    proxy = map_info.get("proxy") if map_info else None

    # INTELLIGENT SELECTION:
    if "US" in name and proxy:
        fetch_list.append(proxy)
        asset_ticker_lookup[name] = proxy
    else:
        fetch_list.append(primary)
        asset_ticker_lookup[name] = primary

fetch_list.append("SPY")
fetch_list.append("VT")  # Add VT
fetch_list = list(set(fetch_list))

start_date = datetime.now() - timedelta(days=365 * 2)  # 2 Years
print(f"Downloading {len(fetch_list)} tickers...")
data = yf.download(
    fetch_list, start=start_date, progress=False, group_by="ticker", auto_adjust=True
)

# Flatten MultiIndex if needed
if isinstance(data.columns, pd.MultiIndex):
    try:
        data = data.xs("Close", level=1, axis=1)
    except KeyError:
        if "Close" in data.columns.levels[1]:
            data = data.xs("Close", level=1, axis=1)
        else:
            print(
                "Warning: Could not find Close column in MultiIndex. Using first level."
            )
            data = data.droplevel(1, axis=1)
else:
    pass

# Check structure
if isinstance(data, pd.Series):
    data = data.to_frame()

# Fallback Logic
final_asset_map = {}
valid_cols = list(data.columns)

for name in asset_values.keys():
    primary = asset_ticker_lookup[name]
    used_ticker = primary

    if primary not in valid_cols or data[primary].isnull().sum() > len(data) * 0.9:
        map_info = ticker_map.get(name)
        proxy = map_info.get("proxy") if map_info else None
        if proxy:
            # print(f"⚠️  Missing/Empty {primary}. Falling back to proxy: {proxy}")
            if proxy not in data.columns:
                proxy_data = yf.download(
                    proxy, start=start_date, progress=False, auto_adjust=True
                )
                if (
                    isinstance(proxy_data, pd.DataFrame)
                    and "Close" in proxy_data.columns
                ):
                    data[proxy] = proxy_data["Close"]
                elif isinstance(proxy_data, pd.Series):
                    data[proxy] = proxy_data

            used_ticker = proxy
        else:
            print(f"❌ Data failure: {name} ({primary})")
            used_ticker = None

    if used_ticker and used_ticker in data.columns:
        final_asset_map[name] = used_ticker

# 5. Process and Calculate
print("\n--- Calculating Metrics ---")
data = data.ffill()
returns = data.pct_change()
returns = returns.replace([np.inf, -np.inf], np.nan)

spy_ret = returns["SPY"].copy()
vt_ret = returns["VT"].copy() if "VT" in returns.columns else spy_ret  # Fallback

portfolio_ret_series = pd.Series(0.0, index=returns.index)
total_weight = 0
details = []

for name, val in asset_values.items():
    ticker = final_asset_map.get(name)
    if not ticker or ticker not in returns.columns:
        continue

    weight = val / total_portfolio_value
    ret_col = returns[ticker].fillna(0.0)

    portfolio_ret_series += ret_col * weight
    total_weight += weight

    # Pairwise Corr vs SPY (standard)
    pair_df = pd.DataFrame({"A": ret_col, "B": spy_ret}).dropna()
    c = pair_df["A"].corr(pair_df["B"]) if len(pair_df) > 10 else 0.0

    details.append({"Asset": name, "Ticker": ticker, "Weight": weight, "Corr": c})

if total_weight > 0:
    portfolio_ret_series = portfolio_ret_series / total_weight

# Final Portfolio vs SPY
final_df = pd.DataFrame(
    {"Port": portfolio_ret_series, "SPY": spy_ret, "VT": vt_ret}
).dropna()

# Beta vs SPY
beta_spy = final_df["Port"].cov(final_df["SPY"]) / final_df["SPY"].var()
corr_spy = final_df["Port"].corr(final_df["SPY"])

# Beta vs VT
beta_vt = final_df["Port"].cov(final_df["VT"]) / final_df["VT"].var()
corr_vt = final_df["Port"].corr(final_df["VT"])

# Report Writing
report_content = []
report_content.append(f"# 포트폴리오 베타(Beta) 및 상관계수 분석")
report_content.append(f"**기준일**: {datetime.now().strftime('%Y-%m-%d')}")
report_content.append(f"**총 자산 가치**: {total_portfolio_value:,.0f} KRW")
report_content.append("")
report_content.append(f"## 📊 핵심 지표 (Key Metrics)")
report_content.append(f"| 지표 | vs SPY (미국 시장) | vs VT (전세계 시장) |")
report_content.append(f"| :--- | :--- | :--- |")
report_content.append(f"| **베타 (Beta)** | `{beta_spy:.2f}` | `{beta_vt:.2f}` |")
report_content.append(f"| **상관계수 (Corr)** | `{corr_spy:.3f}` | `{corr_vt:.3f}` |")
report_content.append("")

report_content.append("## 💡 베타(Beta) 해석")
report_content.append(
    f"- **SPY 베타 ({beta_spy:.2f})**: 미국 시장이 1% 움직일 때, 귀하의 포트폴리오는 약 **{beta_spy:.2f}%** 움직입니다."
)
if beta_spy < 0.9:
    report_content.append("  - 시장보다 변동성이 낮습니다 (방어적 성향).")
elif beta_spy > 1.1:
    report_content.append("  - 시장보다 변동성이 높습니다 (공격적/레버리지 성향).")
else:
    report_content.append("  - 시장과 유사한 변동성을 가집니다.")

report_content.append(
    f"- **VT 베타 ({beta_vt:.2f})**: 전세계 시장이 1% 움직일 때, 귀하의 포트폴리오는 약 **{beta_vt:.2f}%** 움직입니다."
)

report_content.append("")
report_content.append(f"## 📉 자산별 상세 (Asset Breakdown)")
report_content.append(f"| 자산명 | 티커 | 비중 | SPY 상관 계수 |")
report_content.append(f"| :--- | :--- | :--- | :--- |")

for d in sorted(details, key=lambda x: x["Weight"], reverse=True):
    name_clean = d["Asset"].replace("|", "")
    report_content.append(
        f"| {name_clean} | {d['Ticker']} | {d['Weight'] * 100:.1f}% | {d['Corr']:.2f} |"
    )

with open("d:/gg/research/beta_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_content))

print("✅ 분석 완료. 보고서가 생성되었습니다: beta_report.md")
