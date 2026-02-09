"""
RSI (Relative Strength Index) 전략

- RSI가 과매도 구간(30 이하) 진입 후 반등 → 매수
- RSI가 과매수 구간(70 이상) 진입 후 하락 → 매도
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from models import Signal, Position
from .base import Strategy


class RSIStrategy(Strategy):

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return f"RSI({self.period}, {self.oversold}/{self.overbought})"

    def on_init(self, data: pd.DataFrame) -> None:
        super().on_init(data)
        # RSI 계산
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, float("inf"))
        data["rsi"] = 100 - (100 / (1 + rs))

    def on_candle(
        self,
        index: int,
        row: pd.Series,
        position: Position,
        data: pd.DataFrame,
    ) -> Tuple[Signal, float]:
        if index < self.period + 1:
            return Signal.HOLD, 0.0

        rsi = row["rsi"]
        prev_rsi = data.iloc[index - 1]["rsi"]

        if pd.isna(rsi) or pd.isna(prev_rsi):
            return Signal.HOLD, 0.0

        # 과매도 구간에서 반등 시 매수
        if prev_rsi <= self.oversold and rsi > self.oversold:
            if not position.is_open:
                return Signal.BUY, 1.0

        # 과매수 구간에서 하락 시 매도
        if prev_rsi >= self.overbought and rsi < self.overbought:
            if position.is_open:
                return Signal.SELL, 1.0

        return Signal.HOLD, 0.0
