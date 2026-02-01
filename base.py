"""
전략 추상 클래스

모든 전략은 이 클래스를 상속하고 on_candle() 메서드를 구현합니다.
백테스팅 엔진은 각 봉마다 on_candle()을 호출하여 시그널을 받습니다.
"""

from abc import ABC, abstractmethod

import pandas as pd

from models import Signal, Position


class Strategy(ABC):
    """
    전략 추상 클래스

    새 전략을 만들려면:
    1. 이 클래스를 상속
    2. name 프로퍼티 구현
    3. on_candle() 메서드 구현
    4. (선택) on_init()에서 지표 초기화
    """

    def __init__(self):
        self._data: pd.DataFrame | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """전략 이름"""
        ...

    def on_init(self, data: pd.DataFrame) -> None:
        """
        백테스팅 시작 전 호출. 지표 계산 등 전처리를 여기서 합니다.

        Args:
            data: 전체 OHLCV 데이터 (index=date, columns=open,high,low,close,volume)
        """
        self._data = data

    @abstractmethod
    def on_candle(
        self,
        index: int,
        row: pd.Series,
        position: Position,
        data: pd.DataFrame,
    ) -> tuple[Signal, float]:
        """
        각 봉(캔들)마다 호출되어 매매 시그널을 반환합니다.

        Args:
            index: 현재 봉의 인덱스 (0부터 시작)
            row: 현재 봉 데이터 (open, high, low, close, volume + 지표 컬럼)
            position: 현재 포지션 정보
            data: 전체 데이터 (과거 데이터 참조용, data.iloc[:index+1]로 현재까지만 사용 권장)

        Returns:
            (signal, ratio) 튜플
            - signal: Signal.BUY / Signal.SELL / Signal.HOLD
            - ratio: 매수 시 자본금 대비 비율 (0.0~1.0), 매도 시 보유수량 대비 비율
        """
        ...

    def on_finish(self) -> None:
        """백테스팅 완료 후 호출 (선택적 오버라이드)"""
        pass
