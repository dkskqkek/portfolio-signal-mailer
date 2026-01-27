# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def compare_cash_proxies():
    tickers = ["BIL", "USFR", "SGOV", "BOXX", "ICSH", "PULS"]
    start_date = "2023-01-01"  # BOXX 상장 이후 안정화 기간 고려
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Downloading data for {tickers}...")
    # yfinance returns a MultiIndex DataFrame when multiple tickers are downloaded.
    # We access the 'Adj Close' level.
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if 'Adj Close' in raw_data.columns:
            data = raw_data['Adj Close']
        else:
            # Fallback for some yfinance versions or single column results
            data = raw_data['Close']
    except Exception as e:
        print(f"Failed to download data: {e}")
        return
    
    # 누적 수익률 계산
    returns = data.pct_change().dropna()
    cum_returns = (1 + returns).cumprod()
    
    # 성과 지표 산출
    metrics = []
    for ticker in tickers:
        series = cum_returns[ticker]
        daily_ret = returns[ticker]
        
        cagr = (series.iloc[-1] ** (252 / len(series)) - 1) * 100
        vol = daily_ret.std() * np.sqrt(252) * 100
        mdd = (series / series.cummax() - 1).min() * 100
        sharpe = (cagr - 4.0) / vol if vol > 0 else 0  # 무위험 수익률 4% 가정
        
        metrics.append({
            "Ticker": ticker,
            "CAGR (%)": cagr,
            "Vol (%)": vol,
            "MDD (%)": mdd,
            "Sharpe": sharpe,
            "Final Value": series.iloc[-1]
        })
    
    df_metrics = pd.DataFrame(metrics)
    print("\n[Cash Proxy Comparison Summary]")
    print(df_metrics.sort_values(by="CAGR (%)", ascending=False).to_markdown())
    
    # 결과 저장
    df_metrics.to_csv("cash_proxy_comparison_results.csv", index=False)
    
    # 시각화
    plt.figure(figsize=(12, 7))
    for ticker in tickers:
        plt.plot(cum_returns.index, cum_returns[ticker], label=ticker, linewidth=1.5)
        
    plt.title(f"Cash Proxy Aggressive Comparison (Since {start_date})", fontsize=14)
    plt.ylabel("Cumulative Returns (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("cash_proxy_comparison_plot.png")
    print("\n✓ Plot saved as cash_proxy_comparison_plot.png")

if __name__ == "__main__":
    compare_cash_proxies()
