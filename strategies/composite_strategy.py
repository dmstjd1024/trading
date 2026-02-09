"""
복합 전략 (Composite Strategy)

MACD 신호 + 추세 필터 + RSI 필터를 조합합니다.

매수 조건:
- MACD 골든크로스 발생
- 단기 MA > 장기 MA (상승 추세 확인)
- RSI < 65 (과매수 아님)

매도 조건:
- MACD 데드크로스 발생
- 또는 RSI > 75 (과매수)
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from models import Signal, Position
from .base import Strategy


class CompositeStrategy(Strategy):

    def __init__(
        self,
        ma_short: int = 10,
        ma_long: int = 30,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ):
        super().__init__()
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    @property
    def name(self) -> str:
        return "Composite(MA+RSI+MACD)"

    def on_init(self, data: pd.DataFrame) -> None:
        super().on_init(data)

        # 이동평균선
        data["ma_short"] = data["close"].rolling(self.ma_short).mean()
        data["ma_long"] = data["close"].rolling(self.ma_long).mean()

        # RSI
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, float("inf"))
        data["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = data["close"].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = data["close"].ewm(span=self.macd_slow, adjust=False).mean()
        data["macd"] = ema_fast - ema_slow
        data["macd_signal"] = data["macd"].ewm(span=self.macd_signal, adjust=False).mean()

    def on_candle(
        self,
        index: int,
        row: pd.Series,
        position: Position,
        data: pd.DataFrame,
    ) -> Tuple[Signal, float]:
        min_period = max(self.ma_long, self.macd_slow + self.macd_signal) + 1
        if index < min_period:
            return Signal.HOLD, 0.0

        # 현재 값
        ma_short = row["ma_short"]
        ma_long = row["ma_long"]
        rsi = row["rsi"]
        macd = row["macd"]
        macd_sig = row["macd_signal"]

        # 이전 값
        prev = data.iloc[index - 1]
        prev_macd = prev["macd"]
        prev_macd_sig = prev["macd_signal"]

        if pd.isna(rsi) or pd.isna(macd):
            return Signal.HOLD, 0.0

        # 매수: MACD 골든크로스 + 상승추세 + RSI 적정
        if not position.is_open:
            macd_golden = prev_macd <= prev_macd_sig and macd > macd_sig
            trend_up = ma_short > ma_long
            rsi_ok = rsi < 65

            if macd_golden and trend_up and rsi_ok:
                return Signal.BUY, 1.0

        # 매도: MACD 데드크로스 또는 RSI 과매수
        if position.is_open:
            macd_dead = prev_macd >= prev_macd_sig and macd < macd_sig
            rsi_high = rsi > 75

            if macd_dead or rsi_high:
                return Signal.SELL, 1.0

        return Signal.HOLD, 0.0
