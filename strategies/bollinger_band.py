"""
볼린저밴드 전략

- 가격이 하단밴드 이탈 후 복귀 → 매수
- 가격이 상단밴드 이탈 후 복귀 → 매도
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from models import Signal, Position
from .base import Strategy


class BollingerBandStrategy(Strategy):

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
    ):
        super().__init__()
        self.period = period
        self.num_std = num_std

    @property
    def name(self) -> str:
        return f"BollingerBand({self.period}, {self.num_std})"

    def on_init(self, data: pd.DataFrame) -> None:
        super().on_init(data)
        # 볼린저밴드 계산
        data["bb_middle"] = data["close"].rolling(window=self.period).mean()
        rolling_std = data["close"].rolling(window=self.period).std()
        data["bb_upper"] = data["bb_middle"] + (rolling_std * self.num_std)
        data["bb_lower"] = data["bb_middle"] - (rolling_std * self.num_std)

    def on_candle(
        self,
        index: int,
        row: pd.Series,
        position: Position,
        data: pd.DataFrame,
    ) -> Tuple[Signal, float]:
        if index < self.period + 1:
            return Signal.HOLD, 0.0

        close = row["close"]
        lower = row["bb_lower"]
        upper = row["bb_upper"]

        prev = data.iloc[index - 1]
        prev_close = prev["close"]
        prev_lower = prev["bb_lower"]
        prev_upper = prev["bb_upper"]

        if pd.isna(lower) or pd.isna(upper):
            return Signal.HOLD, 0.0

        # 하단밴드 이탈 후 복귀 시 매수
        if prev_close <= prev_lower and close > lower:
            if not position.is_open:
                return Signal.BUY, 1.0

        # 상단밴드 이탈 후 복귀 시 매도
        if prev_close >= prev_upper and close < upper:
            if position.is_open:
                return Signal.SELL, 1.0

        return Signal.HOLD, 0.0
