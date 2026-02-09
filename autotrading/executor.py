"""
전략 실행기
전략을 실행하고 주문을 처리합니다.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from config import kis_config, autotrading_config
from api_client import KISClient
from models import Signal, OrderType, TradingLog
from strategies import STRATEGIES
from .log_manager import LogManager


class StrategyExecutor:
    """전략 실행기"""

    def __init__(self):
        self.client = KISClient()
        self.log_manager = LogManager()

    def execute_with_screening(
        self,
        strategy_name: str,
        top_n: int = 10,
        market: str = "0000",
    ) -> List[dict]:
        """
        스크리닝으로 종목을 자동 선정한 후 전략 실행

        Args:
            strategy_name: 전략 이름
            top_n: 스크리닝 상위 N개 종목
            market: 대상 시장

        Returns:
            실행 결과 로그 리스트
        """
        try:
            from screener import StockScreener
            screener = StockScreener()
            stock_codes = screener.get_top_codes(top_n=top_n, market=market)

            if not stock_codes:
                return [{
                    "timestamp": datetime.now().isoformat(),
                    "stock_code": "-",
                    "strategy_name": strategy_name,
                    "signal": "-",
                    "price": 0,
                    "quantity": 0,
                    "status": "failed",
                    "message": "스크리닝 결과 없음",
                }]

            print(f"[자동매매] 스크리닝 완료: {len(stock_codes)}개 종목 선정")
            return self.execute_strategy(strategy_name, stock_codes)

        except Exception as e:
            return [{
                "timestamp": datetime.now().isoformat(),
                "stock_code": "-",
                "strategy_name": strategy_name,
                "signal": "-",
                "price": 0,
                "quantity": 0,
                "status": "failed",
                "message": f"스크리닝 오류: {str(e)}",
            }]

    def execute_strategy(
        self,
        strategy_name: str,
        stock_codes: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        전략 실행

        Args:
            strategy_name: 전략 이름
            stock_codes: 대상 종목 코드 리스트

        Returns:
            실행 결과 로그 리스트
        """
        results = []

        # 전략 클래스 가져오기
        if strategy_name not in STRATEGIES:
            return [{
                "timestamp": datetime.now().isoformat(),
                "stock_code": "-",
                "strategy_name": strategy_name,
                "signal": "-",
                "price": 0,
                "quantity": 0,
                "status": "failed",
                "message": f"알 수 없는 전략: {strategy_name}",
            }]

        strategy_class = STRATEGIES[strategy_name]

        for stock_code in stock_codes:
            result = self._execute_for_stock(strategy_class, strategy_name, stock_code)
            results.append(result)

            # 로그 저장
            self.log_manager.write_log_dict(result)

        return results

    def _execute_for_stock(
        self,
        strategy_class,
        strategy_name: str,
        stock_code: str,
    ) -> dict:
        """개별 종목 전략 실행"""
        timestamp = datetime.now()

        try:
            # 데이터 로드 (최근 60일)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

            candles = self.client.get_daily_candles(
                stock_code=stock_code,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
            )

            if not candles:
                return self._create_result(
                    timestamp, stock_code, strategy_name,
                    "failed", "데이터 로드 실패",
                )

            # DataFrame 생성
            df = pd.DataFrame([
                {
                    "date": c.date,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ])
            df.set_index("date", inplace=True)

            # 전략 초기화 및 시그널 생성
            strategy = strategy_class()
            strategy.on_init(df)

            # 마지막 봉에서 시그널 확인
            last_idx = len(df) - 1
            last_row = df.iloc[last_idx]

            # 현재 포지션 확인
            from models import Position
            holdings = self.client.get_holdings()
            position = Position(code=stock_code)

            for h in holdings:
                if h.stock_code == stock_code:
                    position.quantity = h.quantity
                    position.avg_price = h.avg_price
                    break

            signal, ratio = strategy.on_candle(last_idx, last_row, position, df)

            # 현재가 조회
            current_price_info = self.client.get_current_price(stock_code)
            current_price = current_price_info["price"]

            # 시그널에 따라 주문 실행
            if signal == Signal.HOLD or ratio <= 0:
                return self._create_result(
                    timestamp, stock_code, strategy_name,
                    "executed", f"HOLD - 현재가: {current_price:,}원",
                    signal="HOLD", price=current_price,
                )

            elif signal == Signal.BUY and not position.is_open:
                # 매수 주문
                return self._execute_buy(
                    timestamp, stock_code, strategy_name,
                    current_price, ratio,
                )

            elif signal == Signal.SELL and position.is_open:
                # 매도 주문
                return self._execute_sell(
                    timestamp, stock_code, strategy_name,
                    current_price, position, ratio,
                )

            else:
                return self._create_result(
                    timestamp, stock_code, strategy_name,
                    "skipped", f"조건 불충족 - 시그널: {signal.value}, 포지션: {'있음' if position.is_open else '없음'}",
                    signal=signal.value, price=current_price,
                )

        except Exception as e:
            return self._create_result(
                timestamp, stock_code, strategy_name,
                "failed", f"오류: {str(e)}",
            )

    def _execute_buy(
        self,
        timestamp: datetime,
        stock_code: str,
        strategy_name: str,
        price: int,
        ratio: float,
    ) -> dict:
        """매수 실행"""
        try:
            # 계좌 잔고 확인
            balance = self.client.get_account_balance()
            available_cash = balance.cash_balance

            # 투자 금액 계산
            max_ratio = autotrading_config.max_position_ratio
            invest_amount = available_cash * min(ratio, max_ratio)

            # 매수 수량 계산
            quantity = math.floor(invest_amount / price)

            if quantity <= 0:
                return self._create_result(
                    timestamp, stock_code, strategy_name,
                    "skipped", f"매수 가능 수량 없음 (잔고: {available_cash:,.0f}원)",
                    signal="BUY", price=price,
                )

            # 주문 실행
            result = self.client.place_order(
                stock_code=stock_code,
                order_type=OrderType.MARKET,
                is_buy=True,
                quantity=quantity,
            )

            return self._create_result(
                timestamp, stock_code, strategy_name,
                "executed", f"매수 주문 완료 - 주문번호: {result['order_no']}",
                signal="BUY", price=price, quantity=quantity,
            )

        except Exception as e:
            return self._create_result(
                timestamp, stock_code, strategy_name,
                "failed", f"매수 오류: {str(e)}",
                signal="BUY", price=price,
            )

    def _execute_sell(
        self,
        timestamp: datetime,
        stock_code: str,
        strategy_name: str,
        price: int,
        position,
        ratio: float,
    ) -> dict:
        """매도 실행"""
        try:
            # 매도 수량 계산
            quantity = math.floor(position.quantity * min(ratio, 1.0))

            if quantity <= 0:
                return self._create_result(
                    timestamp, stock_code, strategy_name,
                    "skipped", f"매도 가능 수량 없음 (보유: {position.quantity}주)",
                    signal="SELL", price=price,
                )

            # 주문 실행
            result = self.client.place_order(
                stock_code=stock_code,
                order_type=OrderType.MARKET,
                is_buy=False,
                quantity=quantity,
            )

            return self._create_result(
                timestamp, stock_code, strategy_name,
                "executed", f"매도 주문 완료 - 주문번호: {result['order_no']}",
                signal="SELL", price=price, quantity=quantity,
            )

        except Exception as e:
            return self._create_result(
                timestamp, stock_code, strategy_name,
                "failed", f"매도 오류: {str(e)}",
                signal="SELL", price=price,
            )

    def _create_result(
        self,
        timestamp: datetime,
        stock_code: str,
        strategy_name: str,
        status: str,
        message: str,
        signal: str = "-",
        price: int = 0,
        quantity: int = 0,
    ) -> dict:
        """결과 딕셔너리 생성"""
        return {
            "timestamp": timestamp.isoformat(),
            "stock_code": stock_code,
            "strategy_name": strategy_name,
            "signal": signal,
            "price": price,
            "quantity": quantity,
            "status": status,
            "message": message,
        }
