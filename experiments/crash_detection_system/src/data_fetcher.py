"""
Data Fetcher Module - Handles data collection and preprocessing
파일 목적: yfinance, pandas_datareader를 통한 데이터 수집
주요 기능: Price data, VIX, Macro indicators 로드
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    데이터 수집 및 전처리를 담당하는 클래스
    
    Responsibility:
    - yfinance를 통한 OHLC 데이터 수집
    - VIX (공포지수) 및 확장 VIX 데이터
    - 신용스프레드 프록시 (HYG vs IEF)
    - 기술적 지표의 기초 데이터 제공
    """
    
    def __init__(self, output_dir: str = "./data"):
        """
        Initialize DataFetcher with logging and cache configuration.
        
        Args:
            output_dir: 데이터 캐시 디렉토리
        """
        self.output_dir = output_dir
        self.cache: Dict = {}
        logger.info(f"DataFetcher initialized with output_dir={output_dir}")
    
    def fetch_price_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from yfinance with validation.
        
        Args:
            ticker: Stock ticker (e.g., 'SPY')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', etc.)
        
        Returns:
            DataFrame with columns: [Open, High, Low, Close, Volume, Adj Close]
        
        Raises:
            ValueError: if data is empty or invalid
        """
        try:
            logger.info(f"Fetching {ticker} data from {start_date} to {end_date}")
            
            # 캐시 확인 (멱등성 확보)
            cache_key = f"{ticker}_{start_date}_{end_date}_{interval}"
            if cache_key in self.cache:
                logger.info(f"Using cached data for {cache_key}")
                return self.cache[cache_key].copy()
            
            # yfinance에서 데이터 수집
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            # 데이터 검증 (Data Integrity)
            if data.empty:
                raise ValueError(f"No data fetched for {ticker}")
            
            # yfinance 최신 버전은 MultiIndex를 사용할 수 있음 - 플래튼하게
            if isinstance(data.columns, pd.MultiIndex):
                # MultiIndex 컬럼을 flatten
                data.columns = data.columns.droplevel(1)  # 티커 레벨 제거
            
            if data.isnull().sum().sum() > 0:
                logger.warning(f"NaN values detected in {ticker}. Forward filling...")
                data = data.ffill().bfill()
            
            # 컬럼명 표준화 (필요한 컬럼만 선택)
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            available_cols = [col for col in required_cols if col in data.columns]
            
            if len(available_cols) < 4:
                raise ValueError(f"Missing essential OHLC columns. Available: {available_cols}")
            
            data = data[available_cols]
            
            logger.info(f"Successfully fetched {len(data)} rows for {ticker}")
            logger.info(f"Columns: {list(data.columns)}")
            
            # 캐시 저장 (멱등성 확보: 동일한 요청시 캐시 재사용)
            self.cache[cache_key] = data.copy()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            raise
    
    def fetch_vix_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch VIX (^VIX) and VVIX (^VVIX) data.
        
        Returns:
            DataFrame with columns: [VIX, VVIX]
        """
        try:
            logger.info(f"Fetching VIX and VVIX data from {start_date} to {end_date}")
            
            cache_key = f"VIX_{start_date}_{end_date}"
            if cache_key in self.cache:
                logger.info(f"Using cached VIX data")
                return self.cache[cache_key].copy()
            
            # VIX와 VVIX 수집
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Adj Close']
            vvix = yf.download('^VVIX', start=start_date, end=end_date, progress=False)['Adj Close']
            
            vix_data = pd.DataFrame({
                'VIX': vix,
                'VVIX': vvix
            }).dropna()
            
            if vix_data.empty:
                raise ValueError("No VIX/VVIX data available")
            
            logger.info(f"VIX data shape: {vix_data.shape}")
            self.cache[cache_key] = vix_data.copy()
            
            return vix_data
            
        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            # Fallback: synthetic VIX proxy using SPY volatility
            logger.warning("Using SPY volatility as VIX proxy")
            return self._create_vix_proxy(start_date, end_date)
    
    def _create_vix_proxy(
        self,
        start_date: str,
        end_date: str,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Create VIX proxy using SPY historical volatility.
        
        VIX 프록시 생성: SPY의 20일 수익률 표준편차 * 100
        이는 VIX가 불가용할 때의 폴백 메커니즘입니다.
        """
        logger.info("Creating VIX proxy from SPY volatility")
        
        try:
            spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
            
            # MultiIndex 처리
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_data.columns = spy_data.columns.droplevel(1)
            
            # 'Adj Close' 또는 'Close' 선택
            if 'Adj Close' in spy_data.columns:
                price_col = spy_data['Adj Close']
            elif 'Close' in spy_data.columns:
                price_col = spy_data['Close']
            else:
                price_col = spy_data.iloc[:, -1]  # 마지막 컬럼
            
            log_returns = np.log(price_col / price_col.shift(1))
            
            vix_proxy = log_returns.rolling(window=window).std() * np.sqrt(252) * 100
            vvix_proxy = log_returns.rolling(window=window).std().rolling(window=5).std() * np.sqrt(252) * 100
            
            result = pd.DataFrame({
                'VIX': vix_proxy,
                'VVIX': vvix_proxy
            }).dropna()
            
            logger.info(f"VIX proxy created with shape: {result.shape}")
            return result
        
        except Exception as e:
            logger.error(f"Error creating VIX proxy: {e}")
            # Fallback to constant VIX
            fallback_vix = pd.DataFrame({
                'VIX': pd.Series(20.0, index=pd.date_range(start=start_date, end=end_date, freq='D')),
                'VVIX': pd.Series(10.0, index=pd.date_range(start=start_date, end=end_date, freq='D'))
            })
            return fallback_vix
    
    def fetch_credit_spread_proxy(
        self,
        start_date: str,
        end_date: str
    ) -> pd.Series:
        """
        Fetch HYG (High Yield Bond) vs IEF (Treasury) spread proxy.
        
        Return:
            Series of HYG/IEF ratio (higher ratio = wider spread = more risk)
        
        설명: 신용 스프레드는 하이일드 채권과 국채 간의 가격 차이를 나타냅니다.
        비율이 높을수록 리스크 프리미엄이 높다는 의미입니다.
        """
        try:
            logger.info("Fetching credit spread proxy (HYG vs IEF)")
            
            cache_key = f"CREDIT_SPREAD_{start_date}_{end_date}"
            if cache_key in self.cache:
                return self.cache[cache_key].copy()
            
            hyg = yf.download('HYG', start=start_date, end=end_date, progress=False)
            ief = yf.download('IEF', start=start_date, end=end_date, progress=False)
            
            # MultiIndex 처리
            if isinstance(hyg.columns, pd.MultiIndex):
                hyg.columns = hyg.columns.droplevel(1)
            if isinstance(ief.columns, pd.MultiIndex):
                ief.columns = ief.columns.droplevel(1)
            
            # 'Adj Close' 또는 'Close' 선택
            hyg_col = 'Adj Close' if 'Adj Close' in hyg.columns else ('Close' if 'Close' in hyg.columns else hyg.columns[-1])
            ief_col = 'Adj Close' if 'Adj Close' in ief.columns else ('Close' if 'Close' in ief.columns else ief.columns[-1])
            
            hyg_prices = hyg[hyg_col]
            ief_prices = ief[ief_col]
            
            # HYG/IEF 비율 (높을수록 스프레드 증가)
            spread = hyg_prices / ief_prices
            spread = spread.dropna()
            
            logger.info(f"Credit spread proxy computed, shape: {spread.shape}")
            self.cache[cache_key] = spread.copy()
            
            return spread
            
        except Exception as e:
            logger.error(f"Error fetching credit spread: {e}")
            return None
    
    def fetch_market_breadth_proxy(
        self,
        start_date: str,
        end_date: str
    ) -> pd.Series:
        """
        Create market breadth proxy using top 100 US stocks.
        
        McClellan Summation Index 대체 지표
        진출주 / 후퇴주의 비율을 통해 시장 폭(breadth) 측정
        """
        try:
            logger.info("Computing market breadth proxy")
            
            # 대표 종목들 (proxy)
            ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
                          'TSLA', 'JPM', 'JNJ', 'V', 'WMT']
            
            returns_dict = {}
            for ticker in ticker_list:
                try:
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    returns = data['Adj Close'].pct_change()
                    returns_dict[ticker] = returns
                except:
                    logger.warning(f"Could not fetch {ticker}")
                    continue
            
            returns_df = pd.DataFrame(returns_dict).fillna(0)
            
            # 양수 수익률 비율 (breadth indicator)
            breadth = (returns_df > 0).sum(axis=1) / len(returns_df.columns)
            
            logger.info(f"Market breadth proxy computed, shape: {breadth.shape}")
            return breadth
            
        except Exception as e:
            logger.error(f"Error computing breadth proxy: {e}")
            return None
    
    def create_comprehensive_dataset(
        self,
        ticker: str = 'SPY',
        start_date: str = '2010-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive dataset combining price, VIX, and macro factors.
        
        Responsibility:
        - 모든 데이터 소스를 통합하여 하나의 DataFrame으로 반환
        - 데이터 무결성 보증 (Full Context)
        - NaN 값 처리 및 로깅
        
        Returns:
            DataFrame with columns: [Open, High, Low, Close, Volume, Adj Close, 
                                     VIX, VVIX, Credit_Spread, Breadth]
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Creating comprehensive dataset for {ticker} ({start_date} to {end_date})")
        
        try:
            # Price data
            price_data = self.fetch_price_data(ticker, start_date, end_date)
            
            # VIX data
            vix_data = self.fetch_vix_data(start_date, end_date)
            
            # Credit spread
            spread_data = self.fetch_credit_spread_proxy(start_date, end_date)
            
            # Market breadth
            breadth_data = self.fetch_market_breadth_proxy(start_date, end_date)
            
            # Merge all data
            result = price_data.copy()
            
            if vix_data is not None:
                result = result.join(vix_data, how='left')
            
            if spread_data is not None:
                result['Credit_Spread'] = spread_data
            
            if breadth_data is not None:
                result['Breadth'] = breadth_data
            
            # Forward fill then backward fill for NaN
            result = result.ffill().bfill()
            
            # 데이터 검증
            nan_count = result.isnull().sum().sum()
            if nan_count > 0:
                logger.warning(f"Remaining NaN values after filling: {nan_count}")
                # Additional filling with mean
                for col in result.columns:
                    if result[col].isnull().any():
                        result[col].fillna(result[col].mean(), inplace=True)
            else:
                logger.info("All NaN values successfully filled (Data Integrity OK)")
            
            logger.info(f"Comprehensive dataset created: shape={result.shape}, columns={list(result.columns)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dataset: {e}")
            raise


if __name__ == "__main__":
    # Test DataFetcher
    logging.basicConfig(level=logging.INFO)
    
    fetcher = DataFetcher(output_dir="./data")
    
    # 테스트: 최근 1년 데이터
    start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    
    df = fetcher.create_comprehensive_dataset('SPY', start, end)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
    print(f"\nData Info:\n{df.info()}")
