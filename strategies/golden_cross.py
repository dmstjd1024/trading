"""
골든크로스 / 데드크로스 전략

- 단기 이동평균선이 장기 이동평균선을 상향 돌파 → 매수
- 단기 이동평균선이 장기 이동평균선을 하향 돌파 → 매도
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from models import Signal, Position
from .base import Strategy


class GoldenCrossStrategy(Strategy):

    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window

    @property
    def name(self) -> str:
        return f"GoldenCross({self.short_window}/{self.long_window})"

    def on_init(self, data: pd.DataFrame) -> None:
        super().on_init(data)
        # 이동평균선 계산
        data[f"ma_{self.short_window}"] = data["close"].rolling(self.short_window).mean()
        data[f"ma_{self.long_window}"] = data["close"].rolling(self.long_window).mean()

    def on_candle(
        self,
        index: int,
        row: pd.Series,
        position: Position,
        data: pd.DataFrame,
    ) -> Tuple[Signal, float]:
        # 이동평균선이 계산되지 않은 초기 구간은 HOLD
        if index < self.long_window:
            return Signal.HOLD, 0.0

        ma_short = row[f"ma_{self.short_window}"]
        ma_long = row[f"ma_{self.long_window}"]

        prev = data.iloc[index - 1]
        prev_ma_short = prev[f"ma_{self.short_window}"]
        prev_ma_long = prev[f"ma_{self.long_window}"]

        # 골든크로스: 단기선이 장기선을 상향 돌파
        if prev_ma_short <= prev_ma_long and ma_short > ma_long:
            if not position.is_open:
                return Signal.BUY, 1.0  # 자본금 100% 매수

        # 데드크로스: 단기선이 장기선을 하향 돌파
        if prev_ma_short >= prev_ma_long and ma_short < ma_long:
            if position.is_open:
                return Signal.SELL, 1.0  # 전량 매도

        return Signal.HOLD, 0.0
