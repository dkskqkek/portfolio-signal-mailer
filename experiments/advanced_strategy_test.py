# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Config
INITIAL_CAPITAL_KRW = 100_000_000
DIVIDEND_TAX_RATE = 0.154
TICKERS = {
    'SCHD': 'SCHD', 'QQQ': 'QQQ', 'GLD': 'GLD', 'JEPI': 'JEPI',
    'KOSPI': '^KS200', 'SIGNAL': 'SPY', 'FX': 'KRW=X'
}

def fetch_data():
    print("Fetching data...")
    price_data, div_data = {}, {}
    for key, ticker in TICKERS.items():
        try:
            print(f"Loading {ticker}...")
            t = yf.Ticker(ticker)
            hist = t.history(period="5y")
            if hist.empty:
                hist = t.history(period="max")
            if hist.empty: continue
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
            price_data[key] = hist['Close'].groupby(hist.index).last()
            div_data[key] = hist['Dividends'].groupby(hist.index).sum() if 'Dividends' in hist.columns else pd.Series(0, index=hist.index)
            print(f"Loaded {ticker}: {len(hist)} rows")
        except Exception as e: print(f"Error {ticker}: {e}")
    
    if not price_data: return None, None
    df = pd.DataFrame(price_data).fillna(method='ffill').dropna()
    for k in price_data.keys():
        if k not in div_data: div_data[k] = pd.Series(0, index=df.index)
        else: div_data[k] = div_data[k].reindex(df.index).fillna(0)
    return df, div_data

def calculate_signal(df):
    spy = df['SIGNAL']
    log_ret = np.log(spy / spy.shift(1).replace(0, np.nan))
    ma, vol = log_ret.rolling(20).mean(), log_ret.rolling(20).std()
    signals, h_ma, h_vol = [], [], []
    for i in range(len(df)):
        if i < 20:
            signals.append(0)
            if not np.isnan(ma.iloc[i]): h_ma.append(ma.iloc[i]); h_vol.append(vol.iloc[i])
            continue
        sig = 1 if len(h_ma) > 20 and (ma.iloc[i] < np.percentile(h_ma, 25) or vol.iloc[i] > np.percentile(h_vol, 75)) else 0
        signals.append(sig); h_ma.append(ma.iloc[i]); h_vol.append(vol.iloc[i])
    df['is_danger'] = signals
    return df

class Portfolio:
    def __init__(self, weights):
        self.weights = weights
        self.cash = INITIAL_CAPITAL_KRW
        self.holdings = {}
        self.events = []

    def run(self, df, div_data, is_dynamic=False):
        history = []
        fx_s = df['FX']
        prev_sig = 0
        # Initial
        p0, fx0 = df.iloc[0], fx_s.iloc[0]
        total = self.cash
        for t, w in self.weights.items():
            p = p0[t] * (fx0 if t != 'KOSPI' else 1)
            self.holdings[t] = (total * w) / p if p > 0 else 0
        self.cash = 0

        for i in range(len(df)):
            prices, fx, sig = df.iloc[i], fx_s.iloc[i], df['is_danger'].iloc[i]
            # Dividends
            for t, sh in self.holdings.items():
                if div_data[t].iloc[i] > 0:
                    self.holdings[t] += (div_data[t].iloc[i] * sh * (1-DIVIDEND_TAX_RATE)) / prices[t]
            # Switch
            if is_dynamic and sig != prev_sig:
                tgt, old = ('JEPI', 'QQQ') if sig == 1 else ('QQQ', 'JEPI')
                if old in self.holdings:
                    val = self.holdings.pop(old) * prices[old]
                    self.holdings[tgt] = val / prices[tgt] if prices[tgt] > 0 else 0
                    self.events.append((df.index[i], f"Switch to {tgt}"))
            prev_sig = sig
            # Value
            v = 0
            for t, sh in self.holdings.items():
                v += sh * prices[t] * (fx if t != 'KOSPI' else 1)
            history.append(v)
        return history

def main():
    df, div_data = fetch_data()
    if df is None: return
    df = calculate_signal(df)
    v1 = Portfolio({'SCHD': 0.5, 'QQQ': 0.45, 'GLD': 0.05}).run(df, div_data)
    s2 = Portfolio({'SCHD': 0.38, 'QQQ': 0.38, 'GLD': 0.05, 'KOSPI': 0.19})
    v2 = s2.run(df, div_data, is_dynamic=True)
    df['V1'], df['V2'] = v1, v2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for ax in [ax1, ax2]:
        ax.plot(df.index, df['V1'], label='Static', color='gray', alpha=0.3)
        ax.plot(df.index, df['V2'], label='Dynamic', color='black')
    
    buys = [x[0] for x in s2.events if 'QQQ' in x[1]]
    if buys: ax1.scatter(buys, df['V2'].loc[buys], color='red', marker='^', s=80, label='Buy QQQ (Red)')
    ax1.set_title("Buy Signals (Red Markers)")
    
    sells = [x[0] for x in s2.events if 'JEPI' in x[1]]
    if sells: ax2.scatter(sells, df['V2'].loc[sells], color='blue', marker='v', s=80, label='Sell QQQ (Blue)')
    ax2.set_title("Sell Signals (Blue Markers)")
    
    for ax in [ax1, ax2]: ax.legend(); ax.grid(True, alpha=0.2)
    plt.tight_layout(); plt.savefig('advanced_comparison.png')

    with open('strategies_comparison.md', 'w', encoding='utf-8') as f:
        f.write("# Statistics\n")
        for n, c in [("Static", "V1"), ("Dynamic", "V2")]:
            f.write(f"- {n}: Return {(df[c].iloc[-1]/df[c].iloc[0]-1)*100:.1f}%, MDD {((df[c]-df[c].cummax())/df[c].cummax()).min()*100:.1f}%\n")
    print("Done.")

if __name__ == "__main__": main()
