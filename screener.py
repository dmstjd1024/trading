"""
멀티팩터 종목 스크리닝 모듈

기술적 팩터(RSI, 거래량, 모멘텀)와 펀더멘탈 팩터(PER, PBR, ROE)를
조합하여 종목을 점수화하고 상위 N개를 선정합니다.

사용법:
    from screener import StockScreener
    screener = StockScreener()
    results = screener.run()
    for r in results:
        print(f"{r['name']} - 점수: {r['total_score']:.1f}")
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

from config import screener_config, ScreenerConfig
from api_client import KISClient


class StockScreener:
    """멀티팩터 종목 스크리닝 엔진"""

    def __init__(self, config: Optional[ScreenerConfig] = None):
        self.config = config or screener_config
        self.client = KISClient()
        self._cache_dir = Path(self.config.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 메인 실행 ──────────────────────────────────

    def run(
        self,
        top_n: Optional[int] = None,
        market: Optional[str] = None,
        tech_weight: Optional[float] = None,
        fund_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        멀티팩터 스크리닝 실행

        Args:
            top_n: 상위 N개 종목 (기본: config.top_n)
            market: 시장 구분 (기본: config.market)
            tech_weight: 기술적 팩터 가중치 (기본: config.tech_weight)
            fund_weight: 펀더멘탈 팩터 가중치 (기본: config.fund_weight)

        Returns:
            스크리닝 결과 리스트 (점수 내림차순)
        """
        top_n = top_n or self.config.top_n
        market = market or self.config.market
        tw = tech_weight if tech_weight is not None else self.config.tech_weight
        fw = fund_weight if fund_weight is not None else self.config.fund_weight

        print(f"\n{'='*60}")
        print(f"  멀티팩터 종목 스크리닝")
        print(f"  시장: {self._market_name(market)} | 기술:{tw:.0%} 펀더멘탈:{fw:.0%}")
        print(f"{'='*60}")

        # 1단계: 후보 종목 수집
        print("\n[1/4] 후보 종목 수집 중...")
        candidates = self._collect_candidates(market)
        print(f"  → {len(candidates)}개 후보 종목")

        if not candidates:
            print("[경고] 후보 종목이 없습니다.")
            return []

        # 2단계: 필터링
        print("\n[2/4] 필터링 적용 중...")
        filtered = self._apply_filters(candidates)
        print(f"  → {len(filtered)}개 종목 통과")

        if not filtered:
            print("[경고] 필터를 통과한 종목이 없습니다.")
            return []

        # 3단계: 팩터 점수 계산
        print(f"\n[3/4] 팩터 점수 계산 중... ({len(filtered)}개 종목)")
        scored = self._calculate_scores(filtered)

        # 4단계: 종합 점수 산출 및 정렬
        print("\n[4/4] 종합 점수 산출 중...")
        results = self._compute_total_score(scored, tw, fw)

        # 상위 N개 선정
        results.sort(key=lambda x: x["total_score"], reverse=True)
        top_results = results[:top_n]

        self._print_results(top_results)
        return top_results

    # ── 1단계: 후보 종목 수집 ──────────────────────

    def _collect_candidates(self, market: str) -> List[Dict[str, Any]]:
        """시가총액 상위 종목을 후보로 수집"""
        try:
            candidates = self.client.get_market_cap_ranking(
                market=market,
                count=50,
            )
            time.sleep(0.5)  # API 호출 제한 준수
            return candidates
        except Exception as e:
            print(f"  [오류] 종목 목록 조회 실패: {e}")
            return []

    # ── 2단계: 필터링 ────────────────────────────

    def _apply_filters(self, candidates: List[Dict]) -> List[Dict]:
        """기본 필터 적용"""
        filtered = []
        for stock in candidates:
            # 시가총액 필터
            market_cap = stock.get("market_cap", 0)
            if market_cap < self.config.min_market_cap:
                continue

            # 거래량 필터
            volume = stock.get("volume", 0)
            if volume < self.config.min_volume:
                continue

            # 가격이 0인 종목 제외
            if stock.get("price", 0) <= 0:
                continue

            filtered.append(stock)

        return filtered

    # ── 3단계: 팩터 점수 계산 ────────────────────

    def _calculate_scores(self, stocks: List[Dict]) -> List[Dict]:
        """각 종목의 기술적/펀더멘탈 팩터 점수 계산"""
        scored = []

        for i, stock in enumerate(stocks):
            code = stock["code"]
            progress = f"[{i+1}/{len(stocks)}]"

            # 기술적 팩터 계산
            tech_scores = self._calc_tech_factors(code, stock)

            # 펀더멘탈 팩터 계산 (캐시 사용)
            fund_scores = self._calc_fund_factors(code, stock)

            stock_result = {
                **stock,
                **tech_scores,
                **fund_scores,
            }
            scored.append(stock_result)

            # API 호출 제한 준수
            if i < len(stocks) - 1:
                time.sleep(0.3)

            # 진행 상황 표시 (10개마다)
            if (i + 1) % 10 == 0:
                print(f"  {progress} {stock.get('name', code)} 처리 완료")

        return scored

    def _calc_tech_factors(self, code: str, stock: Dict) -> Dict[str, float]:
        """기술적 팩터 계산 (일봉 데이터 기반)"""
        result = {
            "rsi": 50.0,
            "volume_ratio": 1.0,
            "momentum_20d": 0.0,
        }

        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")

            candles = self.client.get_daily_candles(code, start_date, end_date)
            if len(candles) < 20:
                return result

            closes = [c.close for c in candles]
            volumes = [c.volume for c in candles]

            # RSI (14일)
            result["rsi"] = self._calc_rsi(closes, period=14)

            # 거래량 비율 (최근 5일 평균 / 20일 평균)
            if len(volumes) >= 20:
                avg_5 = np.mean(volumes[-5:])
                avg_20 = np.mean(volumes[-20:])
                result["volume_ratio"] = avg_5 / avg_20 if avg_20 > 0 else 1.0

            # 20일 수익률 (모멘텀)
            if len(closes) >= 20:
                result["momentum_20d"] = (closes[-1] / closes[-20] - 1) * 100

        except Exception:
            pass  # 데이터 조회 실패 시 기본값 사용

        return result

    def _calc_fund_factors(self, code: str, stock: Dict) -> Dict[str, float]:
        """펀더멘탈 팩터 계산 (재무 데이터 기반, 캐시 지원)"""
        # 시가총액 순위 API에서 이미 PER, PBR을 제공할 수 있음
        result = {
            "per": stock.get("per", 0),
            "pbr": stock.get("pbr", 0),
            "roe": 0.0,
        }

        # 캐시 확인
        cached = self._load_cache(code)
        if cached:
            result["per"] = cached.get("per", result["per"])
            result["pbr"] = cached.get("pbr", result["pbr"])
            result["roe"] = cached.get("roe", result["roe"])
            return result

        # API 조회
        try:
            fin_data = self.client.get_financial_data(code)
            if fin_data.get("roe", 0) != 0:
                result["roe"] = fin_data["roe"]
            # PER/PBR이 0이면 재무 데이터에서 보완
            # (시가총액 API에서 이미 제공되므로 0일 때만 덮어씀)

            # 캐시 저장
            self._save_cache(code, result)
        except Exception:
            pass

        return result

    # ── 4단계: 종합 점수 산출 ────────────────────

    def _compute_total_score(
        self,
        stocks: List[Dict],
        tech_weight: float,
        fund_weight: float,
    ) -> List[Dict]:
        """Z-Score 정규화 후 가중 평균으로 종합 점수 계산"""
        if not stocks:
            return []

        df = pd.DataFrame(stocks)

        # ── 기술적 팩터 정규화 ──

        # RSI 점수: 30~50 구간이 높은 점수 (과매도 = 기회)
        # 50 기준으로 거리가 가까울수록, 30 이하면 보너스
        df["rsi_score"] = df["rsi"].apply(self._rsi_to_score)

        # 거래량 비율: 높을수록 좋음 (관심 증가)
        df["volume_score"] = self._normalize_zscore(df["volume_ratio"])

        # 모멘텀: 적당히 양수가 좋음 (너무 높으면 과매수)
        df["momentum_score"] = df["momentum_20d"].apply(self._momentum_to_score)

        # 기술적 종합 점수
        cfg = self.config
        df["tech_score"] = (
            df["rsi_score"] * cfg.rsi_weight +
            df["volume_score"] * cfg.volume_ratio_weight +
            df["momentum_score"] * cfg.momentum_weight
        )

        # ── 펀더멘탈 팩터 정규화 ──

        # PER: 낮을수록 좋음 (역수 정규화, 음수/0 제외)
        df["per_score"] = df["per"].apply(self._per_to_score)

        # PBR: 낮을수록 좋음 (역수 정규화, 음수/0 제외)
        df["pbr_score"] = df["pbr"].apply(self._pbr_to_score)

        # ROE: 높을수록 좋음
        df["roe_score"] = self._normalize_zscore(df["roe"])

        # 펀더멘탈 종합 점수
        df["fund_score"] = (
            df["per_score"] * cfg.per_weight +
            df["pbr_score"] * cfg.pbr_weight +
            df["roe_score"] * cfg.roe_weight
        )

        # ── 종합 점수 ──
        df["total_score"] = (
            df["tech_score"] * tech_weight +
            df["fund_score"] * fund_weight
        )

        # 0~100 범위로 스케일링
        ts = df["total_score"]
        ts_min, ts_max = ts.min(), ts.max()
        if ts_max > ts_min:
            df["total_score"] = (ts - ts_min) / (ts_max - ts_min) * 100
        else:
            df["total_score"] = 50.0

        return df.to_dict("records")

    # ── 점수 변환 함수들 ──────────────────────────

    @staticmethod
    def _rsi_to_score(rsi: float) -> float:
        """RSI를 점수로 변환 (30~50 구간이 높은 점수)"""
        if rsi <= 0:
            return 50.0
        if rsi <= 30:
            return 90.0  # 과매도 = 매수 기회
        elif rsi <= 50:
            return 70.0 + (50 - rsi) * 1.0  # 30~50 구간은 70~90
        elif rsi <= 70:
            return 50.0 + (70 - rsi) * 1.0  # 50~70 구간은 50~70
        else:
            return max(10.0, 50.0 - (rsi - 70) * 1.3)  # 과매수 = 위험

    @staticmethod
    def _momentum_to_score(momentum: float) -> float:
        """모멘텀을 점수로 변환 (적당한 양수가 좋음)"""
        if momentum <= -10:
            return 30.0  # 급락
        elif momentum <= 0:
            return 50.0 + momentum * 2  # 소폭 하락은 중립~약간 나쁨
        elif momentum <= 10:
            return 50.0 + momentum * 3  # 적당한 상승이 좋음
        elif momentum <= 20:
            return 80.0 - (momentum - 10) * 1  # 과열 시작
        else:
            return max(20.0, 70.0 - (momentum - 20) * 2)  # 과매수

    @staticmethod
    def _per_to_score(per: float) -> float:
        """PER을 점수로 변환 (낮을수록 좋음, 음수/0 제외)"""
        if per <= 0:
            return 30.0  # 적자 기업
        elif per <= 5:
            return 95.0  # 매우 저평가
        elif per <= 10:
            return 85.0
        elif per <= 15:
            return 70.0
        elif per <= 20:
            return 55.0
        elif per <= 30:
            return 40.0
        else:
            return max(10.0, 30.0 - (per - 30) * 0.5)

    @staticmethod
    def _pbr_to_score(pbr: float) -> float:
        """PBR을 점수로 변환 (낮을수록 좋음)"""
        if pbr <= 0:
            return 30.0
        elif pbr <= 0.5:
            return 95.0
        elif pbr <= 1.0:
            return 80.0
        elif pbr <= 1.5:
            return 65.0
        elif pbr <= 2.0:
            return 50.0
        elif pbr <= 3.0:
            return 35.0
        else:
            return max(10.0, 25.0 - (pbr - 3) * 3)

    @staticmethod
    def _normalize_zscore(series: pd.Series) -> pd.Series:
        """Z-Score 정규화 후 0~100 범위로 변환"""
        mean = series.mean()
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series([50.0] * len(series), index=series.index)
        z = (series - mean) / std
        # Z-Score를 0~100으로 변환 (평균=50, ±2σ = 0~100)
        normalized = 50 + z * 25
        return normalized.clip(0, 100)

    @staticmethod
    def _calc_rsi(closes: List[float], period: int = 14) -> float:
        """RSI 계산"""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    # ── 캐시 관리 ────────────────────────────────

    def _load_cache(self, code: str) -> Optional[Dict]:
        """캐시된 재무 데이터 로드"""
        cache_file = self._cache_dir / f"{code}.json"
        try:
            if not cache_file.exists():
                return None
            with open(cache_file) as f:
                data = json.load(f)
            # TTL 확인
            cached_at = datetime.fromisoformat(data.get("_cached_at", "2000-01-01"))
            if datetime.now() - cached_at > timedelta(hours=self.config.cache_ttl_hours):
                return None
            return data
        except Exception:
            return None

    def _save_cache(self, code: str, data: Dict) -> None:
        """재무 데이터 캐시 저장"""
        cache_file = self._cache_dir / f"{code}.json"
        try:
            data["_cached_at"] = datetime.now().isoformat()
            with open(cache_file, "w") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    # ── 출력 ──────────────────────────────────────

    @staticmethod
    def _market_name(market: str) -> str:
        """시장 코드를 한글 이름으로 변환"""
        names = {
            "0000": "전체",
            "0001": "코스피",
            "1001": "코스닥",
            "2001": "코스피200",
        }
        return names.get(market, market)

    @staticmethod
    def _print_results(results: List[Dict]) -> None:
        """결과 출력"""
        print(f"\n{'='*80}")
        print(f"  스크리닝 결과 (상위 {len(results)}개)")
        print(f"{'='*80}")
        print(f"{'순위':>4}  {'종목명':<14} {'현재가':>10} {'등락률':>7} {'총점':>5} {'기술':>5} {'펀더':>5} {'PER':>6} {'PBR':>5} {'RSI':>5}")
        print(f"{'-'*80}")

        for i, r in enumerate(results, 1):
            name = r.get("name", "")[:10]
            price = r.get("price", 0)
            change = r.get("change_rate", 0)
            total = r.get("total_score", 0)
            tech = r.get("tech_score", 0)
            fund = r.get("fund_score", 0)
            per = r.get("per", 0)
            pbr = r.get("pbr", 0)
            rsi = r.get("rsi", 0)

            print(
                f"  {i:>2}.  {name:<14} {price:>9,}원 {change:>+6.2f}% "
                f"{total:>5.1f} {tech:>5.1f} {fund:>5.1f} "
                f"{per:>6.1f} {pbr:>5.2f} {rsi:>5.1f}"
            )

        print(f"{'='*80}")

    def get_top_codes(
        self,
        top_n: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """스크리닝 실행 후 종목 코드만 반환 (자동매매 연동용)"""
        results = self.run(top_n=top_n, **kwargs)
        return [r["code"] for r in results]
