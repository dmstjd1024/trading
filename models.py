"""
데이터 모델 정의
트레이딩 시스템에서 사용하는 핵심 데이터 구조
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Signal(Enum):
    """매매 시그널"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(Enum):
    """주문 유형"""
    MARKET = "MARKET"   # 시장가
    LIMIT = "LIMIT"     # 지정가


@dataclass
class Candle:
    """봉 데이터 (OHLCV)"""
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __repr__(self) -> str:
        return (
            f"Candle({self.date.strftime('%Y-%m-%d')} "
            f"O:{self.open:,.0f} H:{self.high:,.0f} "
            f"L:{self.low:,.0f} C:{self.close:,.0f} V:{self.volume:,})"
        )


@dataclass
class Trade:
    """개별 거래 기록"""
    date: datetime
    signal: Signal
    price: float
    quantity: int
    commission: float = 0.0
    tax: float = 0.0
    slippage: float = 0.0

    @property
    def total_cost(self) -> float:
        """실제 체결 비용 (수수료+세금+슬리피지 포함)"""
        if self.signal == Signal.BUY:
            return self.price * self.quantity + self.commission + self.slippage
        else:  # SELL
            return self.price * self.quantity - self.commission - self.tax - self.slippage


@dataclass
class Position:
    """현재 포지션"""
    code: str
    quantity: int = 0
    avg_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.quantity > 0

    def update_buy(self, price: float, quantity: int) -> None:
        """매수 시 평균단가 갱신"""
        total_value = self.avg_price * self.quantity + price * quantity
        self.quantity += quantity
        self.avg_price = total_value / self.quantity if self.quantity > 0 else 0

    def update_sell(self, quantity: int) -> None:
        """매도 시 수량 차감"""
        self.quantity -= quantity
        if self.quantity <= 0:
            self.quantity = 0
            self.avg_price = 0.0


@dataclass
class BacktestResult:
    """백테스팅 결과"""
    strategy_name: str
    stock_code: str
    period: str
    initial_capital: float
    final_capital: float
    trades: list[Trade] = field(default_factory=list)
    daily_equity: list[tuple[datetime, float]] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        """총 수익률 (%)"""
        return (self.final_capital - self.initial_capital) / self.initial_capital * 100

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def win_trades(self) -> int:
        """수익 거래 수 (매도 기준)"""
        sells = [t for t in self.trades if t.signal == Signal.SELL]
        buys = [t for t in self.trades if t.signal == Signal.BUY]
        wins = 0
        for i, sell in enumerate(sells):
            if i < len(buys) and sell.price > buys[i].price:
                wins += 1
        return wins

    @property
    def win_rate(self) -> float:
        """승률 (%)"""
        sells = [t for t in self.trades if t.signal == Signal.SELL]
        if not sells:
            return 0.0
        return self.win_trades / len(sells) * 100

    def summary(self) -> str:
        """결과 요약 문자열"""
        lines = [
            f"{'='*50}",
            f"  백테스팅 결과: {self.strategy_name}",
            f"{'='*50}",
            f"  종목코드: {self.stock_code}",
            f"  기간: {self.period}",
            f"  초기자본: {self.initial_capital:>14,.0f}원",
            f"  최종자본: {self.final_capital:>14,.0f}원",
            f"  총수익률: {self.total_return:>13.2f}%",
            f"  거래횟수: {self.trade_count:>10}회",
            f"  승률:     {self.win_rate:>13.1f}%",
            f"{'='*50}",
        ]
        return "\n".join(lines)
