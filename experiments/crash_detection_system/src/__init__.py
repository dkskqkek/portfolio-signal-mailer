"""
Early Crash Detection System Package
"""

from .data_fetcher import DataFetcher
from .signal_processor import SignalProcessor
from .strategy import Strategy, SignalType
from .main import CrashDetectionPipeline

__all__ = [
    'DataFetcher',
    'SignalProcessor',
    'Strategy',
    'SignalType',
    'CrashDetectionPipeline'
]

__version__ = '1.0.0'
