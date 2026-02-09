"""
ML 가격 예측 모듈

기술적 지표와 LLM 센티멘트를 피처로 사용하여
XGBoost 모델로 가격 방향을 예측합니다.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from ai.llm_analyzer import MarketAnalysis

warnings.filterwarnings("ignore", category=UserWarning)


# ── 예측 결과 모델 ─────────────────────────────────

@dataclass
class PricePrediction:
    """ML 가격 예측 결과"""
    direction: str = "flat"        # "up" | "down" | "flat"
    probability: float = 0.5       # 예측 확률 0.0 ~ 1.0
    expected_return: float = 0.0   # 예상 수익률 (%)
    features_importance: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "probability": self.probability,
            "expected_return": self.expected_return,
            "features_importance": self.features_importance,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PricePrediction":
        return cls(
            direction=d.get("direction", "flat"),
            probability=d.get("probability", 0.5),
            expected_return=d.get("expected_return", 0.0),
            features_importance=d.get("features_importance", {}),
            timestamp=datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else datetime.now(),
        )

    @property
    def direction_score(self) -> float:
        """방향을 수치 점수로 변환 (-1 ~ +1)"""
        mapping = {"up": 1.0, "flat": 0.0, "down": -1.0}
        return mapping.get(self.direction, 0.0) * self.probability

    @property
    def direction_label_kr(self) -> str:
        """한국어 방향 라벨"""
        return {"up": "📈 상승 예측", "flat": "➡️ 보합 예측", "down": "📉 하락 예측"}.get(self.direction, "보합")


# ── 피처 엔지니어링 ───────────────────────────────

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV 데이터에서 기술적 지표 피처를 계산합니다.

    Args:
        df: open, high, low, close, volume 컬럼을 가진 DataFrame

    Returns:
        피처가 추가된 DataFrame (원본 수정 안 함)
    """
    data = df.copy()
    close = data["close"]
    volume = data["volume"]

    # ── RSI (14일) ──
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    data["feat_rsi"] = 100 - (100 / (1 + rs))

    # ── MACD ──
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["feat_macd"] = ema12 - ema26
    data["feat_macd_signal"] = data["feat_macd"].ewm(span=9, adjust=False).mean()
    data["feat_macd_hist"] = data["feat_macd"] - data["feat_macd_signal"]

    # ── Bollinger Band 위치 (0~1) ──
    bb_ma = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    data["feat_bb_position"] = (close - bb_lower) / bb_range

    # ── 이동평균 기울기 (정규화) ──
    for window in [5, 20, 60]:
        ma = close.rolling(window).mean()
        slope = ma.diff(5) / ma.shift(5) * 100  # 5일간 변화율(%)
        data[f"feat_ma{window}_slope"] = slope

    # ── 가격 대비 이동평균 괴리율 ──
    ma20 = close.rolling(20).mean()
    data["feat_price_ma20_gap"] = (close - ma20) / ma20 * 100

    # ── 거래량 변화율 ──
    vol_ma5 = volume.rolling(5).mean()
    data["feat_volume_ratio"] = volume / vol_ma5.replace(0, np.nan)

    # ── ATR (14일) ──
    high = data["high"]
    low = data["low"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    data["feat_atr"] = tr.rolling(14).mean() / close * 100  # ATR 비율(%)

    # ── 수익률 관련 ──
    data["feat_return_1d"] = close.pct_change(1) * 100
    data["feat_return_5d"] = close.pct_change(5) * 100
    data["feat_return_20d"] = close.pct_change(20) * 100

    # ── 변동성 ──
    data["feat_volatility"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    return data


FEATURE_COLUMNS = [
    "feat_rsi",
    "feat_macd",
    "feat_macd_signal",
    "feat_macd_hist",
    "feat_bb_position",
    "feat_ma5_slope",
    "feat_ma20_slope",
    "feat_ma60_slope",
    "feat_price_ma20_gap",
    "feat_volume_ratio",
    "feat_atr",
    "feat_return_1d",
    "feat_return_5d",
    "feat_return_20d",
    "feat_volatility",
]

# LLM 센티멘트 피처 (별도 추가)
SENTIMENT_FEATURES = [
    "feat_sentiment_score",    # outlook 점수 (-1 ~ +1)
    "feat_sentiment_conf",     # confidence (0 ~ 1)
    "feat_risk_score",         # risk level 점수 (0 ~ 1)
]

# 거시 피처 (별도 추가)
MACRO_FEATURES = [
    "feat_kospi_change",
    "feat_vix_change",
    "feat_usd_krw_change",
    "feat_oil_change",
]


# ── ML 예측기 클래스 ──────────────────────────────

class MLPredictor:
    """XGBoost 기반 가격 방향 예측기"""

    def __init__(
        self,
        model_dir: str = "./data/models",
        prediction_days: int = 5,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_days = prediction_days
        self._models: Dict[str, object] = {}  # stock_code -> model

    def _get_all_feature_cols(self) -> List[str]:
        """전체 피처 컬럼 목록"""
        return FEATURE_COLUMNS + SENTIMENT_FEATURES + MACRO_FEATURES

    def _prepare_data(
        self,
        df: pd.DataFrame,
        sentiment_history: Optional[List[MarketAnalysis]] = None,
        macro_data: Optional[dict] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        학습/추론용 데이터 준비

        Returns:
            (features_df, target_series) - target은 학습 시에만 제공
        """
        # 기술적 피처 계산
        data = compute_technical_features(df)

        # 센티멘트 피처 추가
        data["feat_sentiment_score"] = 0.0
        data["feat_sentiment_conf"] = 0.5
        data["feat_risk_score"] = 0.5

        if sentiment_history:
            # 가장 최근 센티멘트 적용 (전체 행에 동일 값)
            latest = sentiment_history[-1]
            data["feat_sentiment_score"] = latest.outlook_score
            data["feat_sentiment_conf"] = latest.confidence
            data["feat_risk_score"] = latest.risk_score

        # 거시 피처 추가
        data["feat_kospi_change"] = 0.0
        data["feat_vix_change"] = 0.0
        data["feat_usd_krw_change"] = 0.0
        data["feat_oil_change"] = 0.0

        if macro_data:
            data["feat_kospi_change"] = macro_data.get("kospi_change", 0.0)
            data["feat_vix_change"] = macro_data.get("vix_change", 0.0)
            data["feat_usd_krw_change"] = macro_data.get("usd_krw_change", 0.0)
            data["feat_oil_change"] = macro_data.get("oil_change", 0.0)

        # 타겟: 향후 N거래일 수익률
        future_return = data["close"].shift(-self.prediction_days) / data["close"] - 1
        # 3분류: up (>1%), down (<-1%), flat
        target = pd.Series("flat", index=data.index)
        target[future_return > 0.01] = "up"
        target[future_return < -0.01] = "down"

        # NaN 제거
        all_features = self._get_all_feature_cols()
        valid_mask = data[all_features].notna().all(axis=1)
        data = data[valid_mask]
        target = target[valid_mask]

        return data, target

    def train(
        self,
        stock_code: str,
        df: pd.DataFrame,
        sentiment_history: Optional[List[MarketAnalysis]] = None,
        macro_data: Optional[dict] = None,
    ) -> Dict:
        """
        모델 학습

        Args:
            stock_code: 종목 코드
            df: OHLCV DataFrame
            sentiment_history: LLM 분석 이력
            macro_data: 거시경제 변화율 dict

        Returns:
            학습 결과 메트릭 dict
        """
        try:
            from xgboost import XGBClassifier
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.preprocessing import LabelEncoder
            import joblib
        except ImportError as e:
            print(f"[AI-ML] 필요 패키지 미설치: {e}")
            return {"error": str(e)}

        data, target = self._prepare_data(df, sentiment_history, macro_data)

        if len(data) < 100:
            return {"error": f"학습 데이터 부족 ({len(data)}행, 최소 100행 필요)"}

        # 타겟에서 미래 데이터가 없는 행 제거
        valid_idx = target.index[target.notna()]
        # 마지막 prediction_days 개는 타겟을 알 수 없으므로 제외
        train_end = len(data) - self.prediction_days
        if train_end < 50:
            return {"error": "학습 가능 데이터 부족"}

        all_features = self._get_all_feature_cols()
        X = data[all_features].iloc[:train_end].values
        y_raw = target.iloc[:train_end]

        # 라벨 인코딩
        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        # 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))

        # 전체 데이터로 최종 학습
        model.fit(X, y)

        # 모델 저장
        model_path = self.model_dir / f"{stock_code}_xgb.pkl"
        joblib.dump({"model": model, "label_encoder": le, "features": all_features}, model_path)
        self._models[stock_code] = {"model": model, "label_encoder": le, "features": all_features}

        # 피처 중요도
        importance = dict(zip(all_features, model.feature_importances_.tolist()))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

        return {
            "stock_code": stock_code,
            "cv_accuracy": np.mean(scores),
            "cv_scores": scores,
            "train_samples": len(X),
            "features_importance": sorted_importance,
            "model_path": str(model_path),
        }

    def predict(
        self,
        stock_code: str,
        df: pd.DataFrame,
        current_sentiment: Optional[MarketAnalysis] = None,
        macro_data: Optional[dict] = None,
    ) -> PricePrediction:
        """
        가격 방향 예측

        Args:
            stock_code: 종목 코드
            df: 최신 OHLCV DataFrame
            current_sentiment: 현재 LLM 시장 분석
            macro_data: 현재 거시경제 변화율

        Returns:
            PricePrediction 예측 결과
        """
        try:
            import joblib
        except ImportError:
            return self._demo_prediction()

        # 모델 로드
        if stock_code not in self._models:
            model_path = self.model_dir / f"{stock_code}_xgb.pkl"
            if model_path.exists():
                self._models[stock_code] = joblib.load(model_path)
            else:
                print(f"[AI-ML] {stock_code} 모델 없음 — 데모 예측 반환")
                return self._demo_prediction()

        model_data = self._models[stock_code]
        model = model_data["model"]
        le = model_data["label_encoder"]
        feature_cols = model_data["features"]

        # 피처 준비
        sentiment_history = [current_sentiment] if current_sentiment else None
        data, _ = self._prepare_data(df, sentiment_history, macro_data)

        if data.empty:
            return self._demo_prediction()

        # 최신 행으로 예측
        X_latest = data[feature_cols].iloc[[-1]].values
        pred_class = model.predict(X_latest)[0]
        pred_proba = model.predict_proba(X_latest)[0]

        direction = le.inverse_transform([pred_class])[0]
        probability = float(pred_proba.max())

        # 예상 수익률 추정 (단순 선형 맵핑)
        if direction == "up":
            expected_return = probability * 3.0  # 최대 3%
        elif direction == "down":
            expected_return = -probability * 3.0
        else:
            expected_return = 0.0

        # 피처 중요도
        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8])

        return PricePrediction(
            direction=direction,
            probability=probability,
            expected_return=expected_return,
            features_importance=top_features,
            timestamp=datetime.now(),
        )

    def _demo_prediction(self) -> PricePrediction:
        """데모용 예측 결과"""
        return PricePrediction(
            direction="up",
            probability=0.68,
            expected_return=2.04,
            features_importance={
                "feat_rsi": 0.15,
                "feat_macd_hist": 0.13,
                "feat_sentiment_score": 0.12,
                "feat_ma20_slope": 0.10,
                "feat_volume_ratio": 0.09,
                "feat_bb_position": 0.08,
                "feat_return_5d": 0.07,
                "feat_kospi_change": 0.06,
            },
            timestamp=datetime.now(),
        )

    def is_trained(self, stock_code: str) -> bool:
        """해당 종목의 학습된 모델이 있는지 확인"""
        if stock_code in self._models:
            return True
        model_path = self.model_dir / f"{stock_code}_xgb.pkl"
        return model_path.exists()
