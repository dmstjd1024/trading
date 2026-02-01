"""
백테스팅 엔진

전략을 과거 데이터에 적용하여 성과를 시뮬레이션합니다.
수수료, 슬리피지, 거래세를 반영합니다.
"""

import math

import pandas as pd

from config import BacktestConfig, backtest_config
from models import Signal, Trade, Position, BacktestResult
from strategies.base import Strategy


class BacktestEngine:
    """백테스팅 엔진"""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or backtest_config

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        stock_code: str = "UNKNOWN",
    ) -> BacktestResult:
        """
        백테스팅 실행

        Args:
            strategy: 실행할 전략 인스턴스
            data: OHLCV DataFrame (index=date)
            stock_code: 종목코드 (결과 표시용)

        Returns:
            BacktestResult
        """
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")

        # 데이터 복사 (전략이 컬럼을 추가할 수 있으므로)
        df = data.copy()

        # 전략 초기화
        strategy.on_init(df)

        # 상태 변수
        capital = self.config.initial_capital
        position = Position(code=stock_code)
        trades: list[Trade] = []
        daily_equity: list[tuple] = []

        print(f"\n[백테스트] {strategy.name} 시작")
        print(f"  종목: {stock_code}")
        print(f"  기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  초기자본: {capital:,.0f}원")
        print(f"  데이터: {len(df)}봉\n")

        for i in range(len(df)):
            row = df.iloc[i]
            current_date = df.index[i]

            # 전략에 시그널 요청
            signal, ratio = strategy.on_candle(i, row, position, df)

            if signal == Signal.BUY and not position.is_open and ratio > 0:
                trade = self._execute_buy(
                    date=current_date,
                    price=row["close"],
                    capital=capital,
                    ratio=ratio,
                    position=position,
                )
                if trade:
                    capital -= trade.total_cost
                    trades.append(trade)

            elif signal == Signal.SELL and position.is_open and ratio > 0:
                trade = self._execute_sell(
                    date=current_date,
                    price=row["close"],
                    position=position,
                    ratio=ratio,
                )
                if trade:
                    capital += trade.total_cost
                    trades.append(trade)

            # 일별 평가금액 기록
            equity = capital
            if position.is_open:
                equity += position.quantity * row["close"]
            daily_equity.append((current_date, equity))

        # 전략 마무리
        strategy.on_finish()

        # 최종 평가
        final_capital = capital
        if position.is_open:
            last_price = df.iloc[-1]["close"]
            final_capital += position.quantity * last_price

        period_str = f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}"

        result = BacktestResult(
            strategy_name=strategy.name,
            stock_code=stock_code,
            period=period_str,
            initial_capital=self.config.initial_capital,
            final_capital=final_capital,
            trades=trades,
            daily_equity=daily_equity,
        )

        return result

    def _execute_buy(
        self,
        date,
        price: float,
        capital: float,
        ratio: float,
        position: Position,
    ) -> Trade | None:
        """매수 실행"""
        # 슬리피지 적용
        exec_price = price * (1 + self.config.slippage_rate)

        # 매수 가능 금액
        available = capital * min(ratio, 1.0)

        # 수수료 고려한 매수 가능 수량
        quantity = math.floor(available / (exec_price * (1 + self.config.commission_rate)))
        if quantity <= 0:
            return None

        commission = exec_price * quantity * self.config.commission_rate
        slippage_cost = price * quantity * self.config.slippage_rate

        position.update_buy(exec_price, quantity)

        trade = Trade(
            date=date,
            signal=Signal.BUY,
            price=exec_price,
            quantity=quantity,
            commission=commission,
            slippage=slippage_cost,
        )

        print(f"  📈 매수 {date.strftime('%Y-%m-%d')} | {exec_price:,.0f}원 × {quantity:,}주 | 수수료: {commission:,.0f}원")
        return trade

    def _execute_sell(
        self,
        date,
        price: float,
        position: Position,
        ratio: float,
    ) -> Trade | None:
        """매도 실행"""
        # 슬리피지 적용
        exec_price = price * (1 - self.config.slippage_rate)

        # 매도 수량
        quantity = math.floor(position.quantity * min(ratio, 1.0))
        if quantity <= 0:
            return None

        commission = exec_price * quantity * self.config.commission_rate
        tax = exec_price * quantity * self.config.tax_rate
        slippage_cost = price * quantity * self.config.slippage_rate

        # 수익률 계산
        pnl = (exec_price - position.avg_price) * quantity - commission - tax - slippage_cost
        pnl_pct = (exec_price / position.avg_price - 1) * 100

        position.update_sell(quantity)

        trade = Trade(
            date=date,
            signal=Signal.SELL,
            price=exec_price,
            quantity=quantity,
            commission=commission,
            tax=tax,
            slippage=slippage_cost,
        )

        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"  📉 매도 {date.strftime('%Y-%m-%d')} | {exec_price:,.0f}원 × {quantity:,}주 | {emoji} {pnl_pct:+.2f}%")
        return trade
