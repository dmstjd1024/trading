"""
경제 데이터 수집 모듈

뉴스, 거시경제 지표, 시장 심리 데이터를 수집하여
AI 분석 엔진에 입력 데이터를 제공합니다.
"""

from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import requests
import pandas as pd


# ── 데이터 모델 ────────────────────────────────────

@dataclass
class NewsItem:
    """뉴스 아이템"""
    title: str
    summary: str
    source: str
    published: datetime
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "published": self.published.isoformat(),
            "url": self.url,
        }


@dataclass
class MacroIndicators:
    """거시경제 지표"""
    kospi: float = 0.0
    kospi_change: float = 0.0
    kosdaq: float = 0.0
    kosdaq_change: float = 0.0
    usd_krw: float = 0.0
    usd_krw_change: float = 0.0
    vix: float = 0.0
    vix_change: float = 0.0
    wti_oil: float = 0.0
    wti_oil_change: float = 0.0
    us_10y_yield: float = 0.0
    us_10y_yield_change: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "코스피": f"{self.kospi:,.2f} ({self.kospi_change:+.2f}%)",
            "코스닥": f"{self.kosdaq:,.2f} ({self.kosdaq_change:+.2f}%)",
            "원/달러": f"{self.usd_krw:,.1f} ({self.usd_krw_change:+.2f}%)",
            "VIX": f"{self.vix:.2f} ({self.vix_change:+.2f}%)",
            "WTI유가": f"${self.wti_oil:.2f} ({self.wti_oil_change:+.2f}%)",
            "미국10년물": f"{self.us_10y_yield:.3f}% ({self.us_10y_yield_change:+.3f}%p)",
            "수집시각": self.timestamp.strftime("%Y-%m-%d %H:%M"),
        }

    def to_prompt_text(self) -> str:
        """LLM 프롬프트용 텍스트 변환"""
        lines = ["[거시경제 지표]"]
        for k, v in self.to_dict().items():
            if k != "수집시각":
                lines.append(f"- {k}: {v}")
        return "\n".join(lines)


@dataclass
class SentimentData:
    """시장 심리 데이터"""
    fear_greed_index: float = 50.0  # 0(극도의 공포) ~ 100(극도의 탐욕)
    fear_greed_label: str = "중립"
    foreign_net_buy: float = 0.0     # 외국인 순매수 (억원)
    institution_net_buy: float = 0.0  # 기관 순매수 (억원)
    individual_net_buy: float = 0.0   # 개인 순매수 (억원)
    volume_surge_count: int = 0       # 거래량 급증 종목 수
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "공포탐욕지수": f"{self.fear_greed_index:.0f} ({self.fear_greed_label})",
            "외국인순매수": f"{self.foreign_net_buy:+,.0f}억원",
            "기관순매수": f"{self.institution_net_buy:+,.0f}억원",
            "개인순매수": f"{self.individual_net_buy:+,.0f}억원",
            "거래량급증종목": f"{self.volume_surge_count}개",
        }

    def to_prompt_text(self) -> str:
        lines = ["[시장 심리]"]
        for k, v in self.to_dict().items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)


# ── 캐시 유틸리티 ──────────────────────────────────

class _FileCache:
    """간단한 파일 기반 캐시"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{h}.json"

    def get(self, key: str, ttl_seconds: int) -> Optional[dict]:
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(data["_cached_at"])
            if (datetime.now() - cached_at).total_seconds() > ttl_seconds:
                return None
            return data.get("value")
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def set(self, key: str, value) -> None:
        path = self._key_path(key)
        data = {"_cached_at": datetime.now().isoformat(), "value": value}
        path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")


# ── 뉴스 수집기 ───────────────────────────────────

class NewsCollector:
    """경제 뉴스 RSS 수집"""

    RSS_FEEDS = {
        "한국경제": "https://www.hankyung.com/feed/economy",
        "매일경제": "https://www.mk.co.kr/rss/30100041/",
        "연합뉴스경제": "https://www.yna.co.kr/rss/economy.xml",
    }

    def __init__(self, cache_dir: str = "./data/ai_cache"):
        self._cache = _FileCache(cache_dir)

    def collect(self, max_items: int = 20) -> List[NewsItem]:
        """최근 경제 뉴스 수집 (캐시 1시간)"""
        cached = self._cache.get("news_headlines", ttl_seconds=3600)
        if cached:
            return [
                NewsItem(
                    title=n["title"],
                    summary=n["summary"],
                    source=n["source"],
                    published=datetime.fromisoformat(n["published"]),
                    url=n.get("url", ""),
                )
                for n in cached[:max_items]
            ]

        all_items: List[NewsItem] = []

        try:
            import feedparser
        except ImportError:
            print("[AI-뉴스] feedparser 미설치. pip install feedparser")
            return self._demo_news()

        for source, url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    pub_date = datetime.now()
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])

                    summary = getattr(entry, "summary", "")
                    if len(summary) > 200:
                        summary = summary[:200] + "..."

                    all_items.append(NewsItem(
                        title=entry.get("title", ""),
                        summary=summary,
                        source=source,
                        published=pub_date,
                        url=entry.get("link", ""),
                    ))
            except Exception as e:
                print(f"[AI-뉴스] {source} RSS 수집 실패: {e}")

        # 최신순 정렬 + 중복 제거
        seen_titles = set()
        unique = []
        for item in sorted(all_items, key=lambda x: x.published, reverse=True):
            if item.title not in seen_titles:
                seen_titles.add(item.title)
                unique.append(item)

        result = unique[:max_items]

        # 캐싱
        if result:
            self._cache.set("news_headlines", [n.to_dict() for n in result])

        return result if result else self._demo_news()

    def _demo_news(self) -> List[NewsItem]:
        """데모용 뉴스 데이터"""
        now = datetime.now()
        return [
            NewsItem("한은 기준금리 동결…경기 회복세 주시", "한국은행이 기준금리를 3.0%로 동결했다.", "한국경제", now, ""),
            NewsItem("코스피 2,600선 회복…외국인 순매수 전환", "외국인이 이틀 연속 순매수하며 시장 반등.", "매일경제", now, ""),
            NewsItem("반도체 수출 호조…삼성전자 목표가 상향", "증권사들이 반도체 업종 전망을 긍정적으로 전환.", "연합뉴스경제", now, ""),
            NewsItem("미 연준 금리인하 시사…글로벌 증시 상승", "연준 의장이 완화적 발언을 내놓으며 시장 환호.", "한국경제", now, ""),
            NewsItem("원/달러 환율 1,350원대 하락…수출주 주목", "달러 약세에 원화 강세 전환.", "매일경제", now, ""),
        ]


# ── 거시경제 지표 수집기 ──────────────────────────

class MacroCollector:
    """거시경제 지표 수집 (yfinance 기반)"""

    # Yahoo Finance 티커 매핑
    TICKERS = {
        "kospi": "^KS11",
        "kosdaq": "^KQ11",
        "usd_krw": "KRW=X",
        "vix": "^VIX",
        "wti_oil": "CL=F",
        "us_10y": "^TNX",
    }

    def __init__(self, cache_dir: str = "./data/ai_cache"):
        self._cache = _FileCache(cache_dir)

    def collect(self) -> MacroIndicators:
        """거시경제 지표 수집 (캐시 6시간)"""
        cached = self._cache.get("macro_indicators", ttl_seconds=21600)
        if cached:
            indicators = MacroIndicators()
            for k, v in cached.items():
                if hasattr(indicators, k):
                    setattr(indicators, k, v)
            indicators.timestamp = datetime.fromisoformat(cached.get("timestamp", datetime.now().isoformat()))
            return indicators

        try:
            import yfinance as yf
        except ImportError:
            print("[AI-매크로] yfinance 미설치. pip install yfinance")
            return self._demo_macro()

        indicators = MacroIndicators(timestamp=datetime.now())

        for name, ticker in self.TICKERS.items():
            try:
                data = yf.Ticker(ticker)
                hist = data.history(period="5d")
                if hist.empty or len(hist) < 2:
                    continue

                current = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                change_pct = ((current - prev) / prev) * 100 if prev != 0 else 0.0

                if name == "kospi":
                    indicators.kospi = current
                    indicators.kospi_change = change_pct
                elif name == "kosdaq":
                    indicators.kosdaq = current
                    indicators.kosdaq_change = change_pct
                elif name == "usd_krw":
                    # yfinance KRW=X returns USD per KRW, invert
                    indicators.usd_krw = 1 / current if current > 0 else 0
                    indicators.usd_krw_change = -change_pct
                elif name == "vix":
                    indicators.vix = current
                    indicators.vix_change = change_pct
                elif name == "wti_oil":
                    indicators.wti_oil = current
                    indicators.wti_oil_change = change_pct
                elif name == "us_10y":
                    indicators.us_10y_yield = current
                    indicators.us_10y_yield_change = current - prev

                time.sleep(0.2)  # rate limit

            except Exception as e:
                print(f"[AI-매크로] {name} 수집 실패: {e}")

        # 캐싱
        cache_data = {
            "kospi": indicators.kospi, "kospi_change": indicators.kospi_change,
            "kosdaq": indicators.kosdaq, "kosdaq_change": indicators.kosdaq_change,
            "usd_krw": indicators.usd_krw, "usd_krw_change": indicators.usd_krw_change,
            "vix": indicators.vix, "vix_change": indicators.vix_change,
            "wti_oil": indicators.wti_oil, "wti_oil_change": indicators.wti_oil_change,
            "us_10y_yield": indicators.us_10y_yield, "us_10y_yield_change": indicators.us_10y_yield_change,
            "timestamp": indicators.timestamp.isoformat(),
        }
        self._cache.set("macro_indicators", cache_data)

        # 유효한 데이터가 하나도 없으면 데모 데이터
        if indicators.kospi == 0 and indicators.vix == 0:
            return self._demo_macro()

        return indicators

    def _demo_macro(self) -> MacroIndicators:
        """데모용 거시경제 지표"""
        return MacroIndicators(
            kospi=2587.50, kospi_change=0.85,
            kosdaq=845.20, kosdaq_change=-0.32,
            usd_krw=1352.40, usd_krw_change=-0.15,
            vix=18.50, vix_change=-2.30,
            wti_oil=72.45, wti_oil_change=1.20,
            us_10y_yield=4.25, us_10y_yield_change=-0.03,
            timestamp=datetime.now(),
        )


# ── 시장 심리 수집기 ──────────────────────────────

class SentimentCollector:
    """시장 심리 데이터 수집"""

    def __init__(self, cache_dir: str = "./data/ai_cache"):
        self._cache = _FileCache(cache_dir)

    def collect(self) -> SentimentData:
        """시장 심리 데이터 수집 (캐시 1시간)"""
        cached = self._cache.get("sentiment_data", ttl_seconds=3600)
        if cached:
            data = SentimentData()
            for k, v in cached.items():
                if hasattr(data, k) and k != "timestamp":
                    setattr(data, k, v)
            return data

        sentiment = SentimentData(timestamp=datetime.now())

        # CNN Fear & Greed Index 가져오기 시도
        try:
            sentiment.fear_greed_index, sentiment.fear_greed_label = self._fetch_fear_greed()
        except Exception as e:
            print(f"[AI-심리] 공포탐욕지수 수집 실패: {e}")
            sentiment.fear_greed_index = 50.0
            sentiment.fear_greed_label = "중립"

        # VIX 기반 보조 판단
        try:
            import yfinance as yf
            vix_data = yf.Ticker("^VIX").history(period="5d")
            if not vix_data.empty:
                vix_val = float(vix_data["Close"].iloc[-1])
                if vix_val > 30:
                    sentiment.fear_greed_label = "극도의 공포"
                    sentiment.fear_greed_index = max(10, 50 - vix_val)
                elif vix_val > 20:
                    sentiment.fear_greed_label = "공포"
                    sentiment.fear_greed_index = max(20, 60 - vix_val)
                elif vix_val < 12:
                    sentiment.fear_greed_label = "탐욕"
                    sentiment.fear_greed_index = min(80, 90 - vix_val)
        except Exception:
            pass

        # 캐싱
        cache_data = {
            "fear_greed_index": sentiment.fear_greed_index,
            "fear_greed_label": sentiment.fear_greed_label,
            "foreign_net_buy": sentiment.foreign_net_buy,
            "institution_net_buy": sentiment.institution_net_buy,
            "individual_net_buy": sentiment.individual_net_buy,
            "volume_surge_count": sentiment.volume_surge_count,
        }
        self._cache.set("sentiment_data", cache_data)

        return sentiment

    def _fetch_fear_greed(self) -> tuple:
        """CNN Fear & Greed Index API"""
        try:
            resp = requests.get(
                "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                score = data.get("fear_and_greed", {}).get("score", 50)
                rating = data.get("fear_and_greed", {}).get("rating", "Neutral")
                label_map = {
                    "Extreme Fear": "극도의 공포", "Fear": "공포",
                    "Neutral": "중립", "Greed": "탐욕", "Extreme Greed": "극도의 탐욕",
                }
                return float(score), label_map.get(rating, "중립")
        except Exception:
            pass
        return 50.0, "중립"


# ── 통합 데이터 수집기 ────────────────────────────

class EconomicDataAggregator:
    """모든 경제 데이터를 통합 수집"""

    def __init__(self, cache_dir: str = "./data/ai_cache"):
        self.news_collector = NewsCollector(cache_dir)
        self.macro_collector = MacroCollector(cache_dir)
        self.sentiment_collector = SentimentCollector(cache_dir)

    def collect_all(self) -> Dict:
        """모든 경제 데이터 수집"""
        news = self.news_collector.collect()
        macro = self.macro_collector.collect()
        sentiment = self.sentiment_collector.collect()

        return {
            "news": news,
            "macro": macro,
            "sentiment": sentiment,
            "timestamp": datetime.now(),
        }

    def to_llm_prompt(self, data: Optional[Dict] = None) -> str:
        """수집된 데이터를 LLM 프롬프트용 텍스트로 변환"""
        if data is None:
            data = self.collect_all()

        parts = []

        # 뉴스 헤드라인
        news: List[NewsItem] = data["news"]
        parts.append("[오늘의 경제 뉴스 헤드라인]")
        for i, n in enumerate(news[:15], 1):
            parts.append(f"{i}. [{n.source}] {n.title}")

        parts.append("")

        # 거시경제 지표
        macro: MacroIndicators = data["macro"]
        parts.append(macro.to_prompt_text())

        parts.append("")

        # 시장 심리
        sentiment: SentimentData = data["sentiment"]
        parts.append(sentiment.to_prompt_text())

        return "\n".join(parts)
