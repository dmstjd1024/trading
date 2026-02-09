"""
전략 패키지

새 전략을 추가하려면:
1. Strategy 클래스를 상속하여 새 전략 파일 생성
2. 이 파일에서 import 후 STRATEGIES 딕셔너리에 등록
"""

from .base import Strategy
from .golden_cross import GoldenCrossStrategy
from .rsi_strategy import RSIStrategy
from .bollinger_band import BollingerBandStrategy
from .macd_strategy import MACDStrategy
from .composite_strategy import CompositeStrategy
from .ai_strategy import AICompositeStrategy

STRATEGIES = {
    "golden_cross": GoldenCrossStrategy,
    "rsi": RSIStrategy,
    "bollinger_band": BollingerBandStrategy,
    "macd": MACDStrategy,
    "composite": CompositeStrategy,
    "ai_composite": AICompositeStrategy,
}

__all__ = [
    "Strategy",
    "GoldenCrossStrategy",
    "RSIStrategy",
    "BollingerBandStrategy",
    "MACDStrategy",
    "CompositeStrategy",
    "AICompositeStrategy",
    "STRATEGIES",
]
