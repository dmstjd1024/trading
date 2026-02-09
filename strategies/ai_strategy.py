"""
AI 복합 전략 (AI Composite Strategy)

LLM 시장 분석 + ML 가격 예측 + 기술적 확인을 결합하여
최종 매매 신호를 생성합니다.

신호 결합:
  최종 판단 = LLM_outlook(40%) + ML_prediction(40%) + 기술적_확인(20%)

리스크 관리:
  - LLM risk_level "high" → 매수 비율 자동 축소 (최대 30%)
  - ML confidence < 0.6 → 매수 비율 축소
  - 연속 손실 3회 → 자동 쿨다운 (1일 거래 중지)
"""

from __future__ import annotations

from typing import Tuple, Optional

import pandas as pd
import numpy as np

from models import Signal, Position
from strategies.base import Strategy


class AICompositeStrategy(Strategy):
    """LLM + ML + 기술적 분석 복합 전략"""

    def __init__(
        self,
        llm_weight: float = 0.4,
        ml_weight: float = 0.4,
        tech_weight: float = 0.2,
        max_buy_ratio_high_risk: float = 0.3,
        min_confidence: float = 0.6,
        cooldown_after_losses: int = 3,
    ):
        super().__init__()
        self.llm_weight = llm_weight
        self.ml_weight = ml_weight
        self.tech_weight = tech_weight
        self.max_buy_ratio_high_risk = max_buy_ratio_high_risk
        self.min_confidence = min_confidence
        self.cooldown_after_losses = cooldown_after_losses

        # AI 모듈 참조 (on_init에서 설정)
        self._llm_analysis = None   # MarketAnalysis
        self._ml_prediction = None  # PricePrediction

        # 리스크 관리 상태
        self._consecutive_losses = 0
        self._is_cooldown = False
        self._last_buy_price = 0.0

    @property
    def name(self) -> str:
        return "AI Composite(LLM+ML+Tech)"

    def set_ai_signals(self, llm_analysis=None, ml_prediction=None):
        """
        외부에서 AI 분석 결과를 주입합니다.
        백테스트에서는 on_init 시점에 호출하고,
        자동매매에서는 매일 장 시작 전에 호출합니다.
        """
        self._llm_analysis = llm_analysis
        self._ml_prediction = ml_prediction

    def on_init(self, data: pd.DataFrame) -> None:
        """기술적 지표 계산"""
        super().on_init(data)

        close = data["close"]

        # RSI (14일)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, float("inf"))
        data["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        data["macd"] = ema12 - ema26
        data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()

        # 이동평균선
        data["ma5"] = close.rolling(5).mean()
        data["ma20"] = close.rolling(20).mean()
        data["ma60"] = close.rolling(60).mean()

        # Bollinger Band
        bb_ma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        data["bb_upper"] = bb_ma + 2 * bb_std
        data["bb_lower"] = bb_ma - 2 * bb_std

    def on_candle(
        self,
        index: int,
        row: pd.Series,
        position: Position,
        data: pd.DataFrame,
    ) -> Tuple[Signal, float]:
        """각 봉마다 AI 복합 신호 생성"""
        # 지표 안정화 기간
        if index < 60:
            return Signal.HOLD, 0.0

        # 쿨다운 체크
        if self._is_cooldown:
            self._is_cooldown = False  # 다음 봉에서 해제
            return Signal.HOLD, 0.0

        rsi = row.get("rsi", 50)
        macd = row.get("macd", 0)
        macd_sig = row.get("macd_signal", 0)
        ma5 = row.get("ma5", 0)
        ma20 = row.get("ma20", 0)
        close = row.get("close", 0)
        bb_upper = row.get("bb_upper", 0)
        bb_lower = row.get("bb_lower", 0)

        if pd.isna(rsi) or pd.isna(macd):
            return Signal.HOLD, 0.0

        # ── 1) 기술적 점수 계산 (-1 ~ +1) ──
        tech_score = self._compute_tech_score(index, row, data)

        # ── 2) LLM 점수 ──
        llm_score = 0.0
        if self._llm_analysis:
            llm_score = self._llm_analysis.outlook_score

        # ── 3) ML 점수 ──
        ml_score = 0.0
        ml_confidence = 0.5
        if self._ml_prediction:
            ml_score = self._ml_prediction.direction_score
            ml_confidence = self._ml_prediction.probability

        # ── 종합 점수 ──
        combined = (
            self.llm_weight * llm_score
            + self.ml_weight * ml_score
            + self.tech_weight * tech_score
        )

        # ── 매매 판단 ──
        if not position.is_open:
            # 매수 판단
            if combined > 0.3:
                ratio = self._compute_buy_ratio(combined, ml_confidence)
                return Signal.BUY, ratio
            elif combined > 0.15:
                ratio = self._compute_buy_ratio(combined * 0.7, ml_confidence)
                return Signal.BUY, ratio

        elif position.is_open:
            # 매도 판단
            # 강한 매도 신호
            if combined < -0.3:
                self._update_loss_tracking(row, position)
                return Signal.SELL, 1.0

            # RSI 과매수 + 약한 하락 신호
            if rsi > 75 and combined < 0:
                self._update_loss_tracking(row, position)
                return Signal.SELL, 0.7

            # Bollinger Band 상단 돌파 + 음의 종합점수
            if close > bb_upper and combined < 0.1:
                self._update_loss_tracking(row, position)
                return Signal.SELL, 0.5

            # 기술적 매도: MACD 데드크로스
            if index > 0:
                prev = data.iloc[index - 1]
                prev_macd = prev.get("macd", 0)
                prev_sig = prev.get("macd_signal", 0)
                if not pd.isna(prev_macd) and prev_macd >= prev_sig and macd < macd_sig:
                    if combined < 0.1:
                        self._update_loss_tracking(row, position)
                        return Signal.SELL, 0.8

        return Signal.HOLD, 0.0

    def _compute_tech_score(self, index: int, row: pd.Series, data: pd.DataFrame) -> float:
        """기술적 지표 기반 점수 계산 (-1 ~ +1)"""
        score = 0.0
        rsi = row.get("rsi", 50)
        macd = row.get("macd", 0)
        macd_sig = row.get("macd_signal", 0)
        ma5 = row.get("ma5", 0)
        ma20 = row.get("ma20", 0)
        close = row.get("close", 0)

        # RSI 기반 (과매도 = 매수, 과매수 = 매도)
        if not pd.isna(rsi):
            if rsi < 30:
                score += 0.4
            elif rsi < 40:
                score += 0.2
            elif rsi > 70:
                score -= 0.3
            elif rsi > 60:
                score -= 0.1

        # MACD 기반
        if not pd.isna(macd) and not pd.isna(macd_sig):
            if macd > macd_sig:
                score += 0.3
            else:
                score -= 0.3

            # MACD 크로스오버 감지
            if index > 0:
                prev = data.iloc[index - 1]
                prev_macd = prev.get("macd", 0)
                prev_sig = prev.get("macd_signal", 0)
                if not pd.isna(prev_macd):
                    if prev_macd <= prev_sig and macd > macd_sig:
                        score += 0.3  # 골든크로스
                    elif prev_macd >= prev_sig and macd < macd_sig:
                        score -= 0.3  # 데드크로스

        # 이동평균 추세
        if ma5 > 0 and ma20 > 0:
            if ma5 > ma20:
                score += 0.2
            else:
                score -= 0.2

        return max(-1.0, min(1.0, score))

    def _compute_buy_ratio(self, combined_score: float, ml_confidence: float) -> float:
        """매수 비율 계산 (리스크 관리 적용)"""
        # 기본 비율: 종합 점수에 비례
        base_ratio = min(1.0, combined_score * 1.5)

        # LLM 리스크 레벨에 따른 축소
        if self._llm_analysis and self._llm_analysis.risk_level == "high":
            base_ratio = min(base_ratio, self.max_buy_ratio_high_risk)

        # ML 신뢰도 낮으면 축소
        if ml_confidence < self.min_confidence:
            base_ratio *= ml_confidence

        # 연속 손실 시 축소
        if self._consecutive_losses >= self.cooldown_after_losses:
            self._is_cooldown = True
            self._consecutive_losses = 0
            return 0.0

        return max(0.1, min(1.0, base_ratio))

    def _update_loss_tracking(self, row: pd.Series, position: Position):
        """연속 손실 추적"""
        if position.is_open and position.avg_price > 0:
            current_price = row.get("close", 0)
            if current_price < position.avg_price:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

    def on_finish(self) -> None:
        """백테스트 종료 정리"""
        self._consecutive_losses = 0
        self._is_cooldown = False
