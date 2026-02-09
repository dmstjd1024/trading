"""
MACD (Moving Average Convergence Divergence) 전략

- MACD 선이 시그널 선을 상향 돌파 → 매수
- MACD 선이 시그널 선을 하향 돌파 → 매도
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from models import Signal, Position
from .base import Strategy


class MACDStrategy(Strategy):

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    @property
    def name(self) -> str:
        return f"MACD({self.fast_period}/{self.slow_period}/{self.signal_period})"

    def on_init(self, data: pd.DataFrame) -> None:
        super().on_init(data)
        # EMA 계산
        ema_fast = data["close"].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data["close"].ewm(span=self.slow_period, adjust=False).mean()

        # MACD 라인 = 단기 EMA - 장기 EMA
        data["macd"] = ema_fast - ema_slow

        # 시그널 라인 = MACD의 EMA
        data["macd_signal"] = data["macd"].ewm(span=self.signal_period, adjust=False).mean()

        # 히스토그램 = MACD - 시그널
        data["macd_hist"] = data["macd"] - data["macd_signal"]

    def on_candle(
        self,
        index: int,
        row: pd.Series,
        position: Position,
        data: pd.DataFrame,
    ) -> Tuple[Signal, float]:
        # 충분한 데이터가 쌓일 때까지 대기
        min_period = self.slow_period + self.signal_period
        if index < min_period:
            return Signal.HOLD, 0.0

        macd = row["macd"]
        signal = row["macd_signal"]

        prev = data.iloc[index - 1]
        prev_macd = prev["macd"]
        prev_signal = prev["macd_signal"]

        if pd.isna(macd) or pd.isna(signal):
            return Signal.HOLD, 0.0

        # MACD가 시그널 선을 상향 돌파 → 매수
        if prev_macd <= prev_signal and macd > signal:
            if not position.is_open:
                return Signal.BUY, 1.0

        # MACD가 시그널 선을 하향 돌파 → 매도
        if prev_macd >= prev_signal and macd < signal:
            if position.is_open:
                return Signal.SELL, 1.0

        return Signal.HOLD, 0.0
