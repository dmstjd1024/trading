"""
LLM 시장 분석 모듈

경제 뉴스, 거시경제 지표, 시장 심리 데이터를 종합하여
LLM(Gemini/GPT/Claude)을 통해 시장 전망을 분석합니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from ai.data_sources import EconomicDataAggregator, _FileCache


# ── 분석 결과 모델 ─────────────────────────────────

@dataclass
class MarketAnalysis:
    """LLM 시장 분석 결과"""
    outlook: str = "neutral"         # "bullish" | "bearish" | "neutral"
    confidence: float = 0.5          # 0.0 ~ 1.0
    reasoning: str = ""              # 분석 근거 (한글)
    sector_outlook: Dict[str, str] = field(default_factory=dict)  # {"반도체": "bullish", ...}
    risk_level: str = "medium"       # "low" | "medium" | "high"
    key_factors: list = field(default_factory=list)  # 주요 영향 요인
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "outlook": self.outlook,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "sector_outlook": self.sector_outlook,
            "risk_level": self.risk_level,
            "key_factors": self.key_factors,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MarketAnalysis":
        return cls(
            outlook=d.get("outlook", "neutral"),
            confidence=d.get("confidence", 0.5),
            reasoning=d.get("reasoning", ""),
            sector_outlook=d.get("sector_outlook", {}),
            risk_level=d.get("risk_level", "medium"),
            key_factors=d.get("key_factors", []),
            timestamp=datetime.fromisoformat(d["timestamp"]) if "timestamp" in d else datetime.now(),
        )

    @property
    def outlook_score(self) -> float:
        """전망을 수치 점수로 변환 (-1 ~ +1)"""
        mapping = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
        return mapping.get(self.outlook, 0.0) * self.confidence

    @property
    def risk_score(self) -> float:
        """리스크 레벨을 수치로 변환 (0~1, 높을수록 위험)"""
        return {"low": 0.2, "medium": 0.5, "high": 0.8}.get(self.risk_level, 0.5)

    @property
    def outlook_label_kr(self) -> str:
        """한국어 전망 라벨"""
        return {"bullish": "🟢 강세", "neutral": "🟡 보합", "bearish": "🔴 약세"}.get(self.outlook, "보합")


# ── 시스템 프롬프트 ────────────────────────────────

SYSTEM_PROMPT = """당신은 한국 주식시장 전문 애널리스트입니다.
주어진 경제 데이터(뉴스, 거시지표, 심리지표)를 종합 분석하여 시장 전망을 제공합니다.

분석 시 다음 사항을 고려하세요:
1. 글로벌 경제 흐름이 한국 시장에 미치는 영향
2. 환율·금리·유가 변동과 한국 수출기업에 대한 영향
3. 투자자 심리(공포/탐욕지수, 외국인·기관 매매 동향)
4. 뉴스 헤드라인의 전체적인 톤과 방향성

반드시 아래 JSON 형식으로만 응답하세요 (설명 텍스트 없이 JSON만):
{
    "outlook": "bullish" 또는 "bearish" 또는 "neutral",
    "confidence": 0.0~1.0 사이 숫자,
    "reasoning": "한국어로 2~3문장의 분석 근거",
    "sector_outlook": {"섹터명": "bullish/bearish/neutral", ...},
    "risk_level": "low" 또는 "medium" 또는 "high",
    "key_factors": ["핵심 요인 1", "핵심 요인 2", ...]
}"""


# ── LLM 분석기 클래스 ─────────────────────────────

class LLMAnalyzer:
    """LLM 기반 시장 분석기"""

    def __init__(
        self,
        provider: str = "gemini",
        api_key: str = "",
        model: str = "gemini-2.5-flash",
        cache_dir: str = "./data/ai_cache",
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self._cache = _FileCache(cache_dir)
        self._data_aggregator = EconomicDataAggregator(cache_dir)

    def analyze(self, force_refresh: bool = False) -> MarketAnalysis:
        """
        시장 전망 분석 실행

        Args:
            force_refresh: True면 캐시 무시하고 새로 분석

        Returns:
            MarketAnalysis 분석 결과
        """
        # 캐시 확인 (하루 2회 = 12시간 TTL)
        if not force_refresh:
            cached = self._cache.get("llm_analysis", ttl_seconds=43200)
            if cached:
                return MarketAnalysis.from_dict(cached)

        # 경제 데이터 수집
        econ_data = self._data_aggregator.collect_all()
        prompt_text = self._data_aggregator.to_llm_prompt(econ_data)

        # LLM 호출
        if not self.api_key:
            print("[AI-LLM] API 키 미설정 — 데모 분석 결과를 사용합니다.")
            return self._demo_analysis()

        try:
            if self.provider == "gemini":
                result = self._call_gemini(prompt_text)
            elif self.provider == "openai":
                result = self._call_openai(prompt_text)
            elif self.provider == "anthropic":
                result = self._call_anthropic(prompt_text)
            else:
                print(f"[AI-LLM] 미지원 프로바이더: {self.provider}")
                return self._demo_analysis()
        except Exception as e:
            print(f"[AI-LLM] API 호출 실패: {e}")
            return self._demo_analysis()

        # 결과 파싱
        analysis = self._parse_response(result)

        # 캐싱
        self._cache.set("llm_analysis", analysis.to_dict())

        return analysis

    def _call_gemini(self, user_prompt: str) -> str:
        """Google Gemini API 호출 (무료 티어 지원)"""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai 패키지가 필요합니다. pip install google-genai")

        client = genai.Client(api_key=self.api_key)
        model_name = self.model if "gemini" in self.model else "gemini-2.5-flash"

        # 2.5 모델은 thinking 비활성화 (토큰 절약, JSON 응답 안정성)
        thinking_config = None
        if "2.5" in model_name:
            thinking_config = types.ThinkingConfig(thinking_budget=0)

        response = client.models.generate_content(
            model=model_name,
            contents=f"오늘의 경제 데이터를 분석하여 한국 주식시장 전망을 제공해주세요.\n\n{user_prompt}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3 if "2.5" not in model_name else 1.0,
                max_output_tokens=2048,
                response_mime_type="application/json",
                thinking_config=thinking_config,
            ),
        )
        return response.text

    def _call_openai(self, user_prompt: str) -> str:
        """OpenAI API 호출"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 패키지가 필요합니다. pip install openai")

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"오늘의 경제 데이터를 분석하여 한국 주식시장 전망을 제공해주세요.\n\n{user_prompt}"},
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _call_anthropic(self, user_prompt: str) -> str:
        """Anthropic API 호출"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic 패키지가 필요합니다. pip install anthropic")

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model if "claude" in self.model else "claude-sonnet-4-20250514",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"오늘의 경제 데이터를 분석하여 한국 주식시장 전망을 제공해주세요.\n\n{user_prompt}"},
            ],
        )
        return response.content[0].text

    def _parse_response(self, response_text: str) -> MarketAnalysis:
        """LLM 응답을 MarketAnalysis로 파싱"""
        try:
            # JSON 블록 추출 (```json ... ``` 형태일 수 있음)
            text = response_text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)

            return MarketAnalysis(
                outlook=data.get("outlook", "neutral"),
                confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
                reasoning=data.get("reasoning", ""),
                sector_outlook=data.get("sector_outlook", {}),
                risk_level=data.get("risk_level", "medium"),
                key_factors=data.get("key_factors", []),
                timestamp=datetime.now(),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"[AI-LLM] 응답 파싱 실패: {e}")
            return self._demo_analysis()

    def _demo_analysis(self) -> MarketAnalysis:
        """데모용 분석 결과"""
        return MarketAnalysis(
            outlook="bullish",
            confidence=0.72,
            reasoning=(
                "반도체 수출 호조와 외국인 순매수 전환이 긍정적입니다. "
                "다만 환율 변동성과 미국 금리 불확실성이 상승 폭을 제한할 수 있습니다. "
                "전반적으로 코스피 2,600선 이상 안착을 시도할 것으로 판단됩니다."
            ),
            sector_outlook={
                "반도체": "bullish",
                "자동차": "bullish",
                "금융": "neutral",
                "바이오": "neutral",
                "건설": "bearish",
            },
            risk_level="medium",
            key_factors=[
                "반도체 수출 증가세 지속",
                "외국인 순매수 전환",
                "미 연준 금리인하 기대",
                "원/달러 환율 안정화",
            ],
            timestamp=datetime.now(),
        )

    def get_latest_analysis(self) -> Optional[MarketAnalysis]:
        """가장 최근 캐시된 분석 결과 반환 (없으면 None)"""
        cached = self._cache.get("llm_analysis", ttl_seconds=86400)  # 24시간 내 결과
        if cached:
            return MarketAnalysis.from_dict(cached)
        return None
