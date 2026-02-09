"""
설정 관리 모듈
한국투자증권 API 키 및 시스템 설정을 관리합니다.

사용법:
  1. 환경변수로 설정: KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO, KIS_HTS_ID
  2. 또는 secret.json 파일 사용 (python-kis 방식)
  3. 또는 아래 값을 직접 수정
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# .env 파일에서 환경변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv 미설치 시 수동으로 .env 파일 파싱
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        with open(_env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

# python-kis 라이브러리 임포트
# Python 3.10+ 필요 (union type 문법 사용)
PYKIS_AVAILABLE = False
PyKis = None
KisAuth = None

try:
    from pykis import PyKis, KisAuth
    PYKIS_AVAILABLE = True
except (ImportError, TypeError, SyntaxError):
    # Python 3.9 이하에서는 union type 문법 오류 발생
    pass


@dataclass
class KISConfig:
    """한국투자증권 API 설정"""

    # API 인증 정보 (환경변수 우선, 없으면 기본값)
    app_key: str = field(
        default_factory=lambda: os.getenv("KIS_APP_KEY", "YOUR_APP_KEY")
    )
    app_secret: str = field(
        default_factory=lambda: os.getenv("KIS_APP_SECRET", "YOUR_APP_SECRET")
    )
    account_no: str = field(
        default_factory=lambda: os.getenv("KIS_ACCOUNT_NO", "00000000-00")
    )
    hts_id: str = field(
        default_factory=lambda: os.getenv("KIS_HTS_ID", "")
    )

    # 모의투자 vs 실전투자 (기본: 모의투자 — 안전)
    is_paper: bool = True

    # python-kis 시크릿 파일 경로
    secret_file: str = "secret.json"

    @property
    def base_url(self) -> str:
        if self.is_paper:
            return "https://openapivts.koreainvestment.com:29443"
        return "https://openapi.koreainvestment.com:9443"

    def validate(self) -> bool:
        # secret.json 파일이 있으면 유효
        if Path(self.secret_file).exists():
            return True
        if self.app_key == "YOUR_APP_KEY":
            print("[경고] API 키가 설정되지 않았습니다.")
            print("  환경변수 KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO, KIS_HTS_ID를 설정하거나")
            print("  secret.json 파일을 생성하거나, config.py를 직접 수정해주세요.")
            return False
        return True

    def create_secret_file(self) -> bool:
        """환경변수로부터 secret.json 파일 생성"""
        if not PYKIS_AVAILABLE:
            print("[오류] python-kis가 설치되지 않았습니다. pip install python-kis")
            return False

        if self.app_key == "YOUR_APP_KEY":
            print("[오류] API 키가 설정되지 않았습니다.")
            return False

        auth = KisAuth(
            id=self.hts_id or "user",
            appkey=self.app_key,
            secretkey=self.app_secret,
            account=self.account_no,
            virtual=self.is_paper,
        )
        auth.save(self.secret_file)
        print(f"[설정] {self.secret_file} 파일 생성 완료")
        return True

    def get_pykis(self) -> Optional["PyKis"]:
        """PyKis 클라이언트 인스턴스 반환

        Note: python-kis v2.x는 실전투자 키가 필수입니다.
              모의투자 전용 키만 있는 경우 None을 반환하며,
              직접 REST API 모드로 자동 전환됩니다.
        """
        if not PYKIS_AVAILABLE:
            return None

        # secret.json 파일이 있으면 사용
        if Path(self.secret_file).exists():
            return PyKis(self.secret_file, keep_token=True)

        # 모의투자 전용 키인 경우 python-kis v2.x 사용 불가
        # (v2.x는 실전투자 키가 필수, 모의투자 키는 추가 옵션)
        if self.is_paper:
            return None

        # 실전투자: 일반 파라미터 사용
        if self.app_key != "YOUR_APP_KEY":
            return PyKis(
                id=self.hts_id or "user",
                account=self.account_no,
                appkey=self.app_key,
                secretkey=self.app_secret,
                keep_token=True,
            )

        return None


@dataclass
class BacktestConfig:
    """백테스팅 설정"""

    initial_capital: float = 10_000_000  # 초기 자본금 (1천만원)
    commission_rate: float = 0.00015     # 수수료율 (0.015%)
    slippage_rate: float = 0.001         # 슬리피지 (0.1%)
    tax_rate: float = 0.0018             # 거래세 (매도 시, 0.18% - 2025년 기준)
    stop_loss_rate: float = 0.05         # 손절선 (5% 손실 시 자동 청산)
    take_profit_rate: float = 0.10       # 익절선 (10% 수익 시 자동 청산)


@dataclass
class DataConfig:
    """데이터 저장 설정"""

    data_dir: str = "./data"
    db_path: str = "./data/stocks.db"


@dataclass
class DashboardConfig:
    """대시보드 설정"""

    page_title: str = "한국주식 트레이딩 시스템"
    page_icon: str = "📈"
    layout: str = "wide"
    refresh_interval: int = 60  # 초 단위 자동 새로고침


@dataclass
class AutoTradingConfig:
    """자동매매 설정"""

    enabled: bool = False
    strategy_name: str = "golden_cross"
    stock_codes: List[str] = field(default_factory=lambda: ["005930"])
    schedule_time: str = "09:05"  # 매일 실행 시간 (HH:MM)
    max_position_ratio: float = 0.3  # 종목당 최대 투자 비율
    log_dir: str = "./logs"


@dataclass
class ScreenerConfig:
    """멀티팩터 종목 스크리닝 설정"""

    # 종목 선정 수
    top_n: int = 10

    # 대상 시장 ("0000": 전체, "0001": 코스피, "1001": 코스닥)
    market: str = "0000"

    # 팩터 가중치 (기술적 vs 펀더멘탈, 합계 1.0)
    tech_weight: float = 0.5
    fund_weight: float = 0.5

    # 필터 조건
    min_market_cap: int = 1000         # 최소 시가총액 (억원)
    min_volume: int = 100_000          # 최소 일평균 거래량
    exclude_managed: bool = True       # 관리종목 제외

    # 기술적 팩터별 가중치
    rsi_weight: float = 0.3            # RSI (과매도 기회)
    volume_ratio_weight: float = 0.3   # 거래량 증가율
    momentum_weight: float = 0.4       # 20일 수익률 (모멘텀)

    # 펀더멘탈 팩터별 가중치
    per_weight: float = 0.35           # PER (낮을수록 저평가)
    pbr_weight: float = 0.35           # PBR (낮을수록 저평가)
    roe_weight: float = 0.30           # ROE (높을수록 우량)

    # 재무 데이터 캐시 (분기별 업데이트이므로 캐싱)
    cache_dir: str = "./data/screener_cache"
    cache_ttl_hours: int = 24          # 캐시 유효 시간


@dataclass
class AIConfig:
    """AI 분석 엔진 설정"""

    # LLM 설정
    ai_provider: str = field(
        default_factory=lambda: os.getenv("AI_PROVIDER", "gemini")
    )  # "gemini", "openai", "anthropic"
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("AI_LLM_MODEL", "gemini-2.5-flash")
    )

    # ML 설정
    ml_retrain_days: int = 7          # 재학습 주기 (일)
    ml_lookback_days: int = 365       # 학습 데이터 기간 (일)
    ml_prediction_days: int = 5       # 예측 기간 (거래일)
    ml_model_dir: str = "./data/models"

    # 데이터 수집 설정
    news_cache_ttl_minutes: int = 60  # 뉴스 캐시 유효 시간 (분)
    macro_cache_ttl_hours: int = 6    # 거시지표 캐시 유효 시간 (시간)
    ecos_api_key: str = field(
        default_factory=lambda: os.getenv("ECOS_API_KEY", "")
    )  # 한국은행 ECOS API 키

    # 신호 결합 가중치
    llm_weight: float = 0.4           # LLM 분석 가중치
    ml_weight: float = 0.4            # ML 예측 가중치
    technical_weight: float = 0.2     # 기술적 지표 가중치

    # 리스크 관리
    max_buy_ratio_high_risk: float = 0.3   # 고위험 시 최대 매수 비율
    min_confidence: float = 0.6            # 최소 신뢰도 (이하 시 매수 비율 축소)
    cooldown_after_losses: int = 3         # 연속 손실 후 쿨다운 횟수

    # 캐시 디렉토리
    cache_dir: str = "./data/ai_cache"

    def validate_llm(self) -> bool:
        """LLM API 키가 설정되어 있는지 확인"""
        if self.ai_provider == "gemini":
            return bool(self.gemini_api_key)
        elif self.ai_provider == "openai":
            return bool(self.openai_api_key)
        elif self.ai_provider == "anthropic":
            return bool(self.anthropic_api_key)
        return False


# 전역 설정 인스턴스
kis_config = KISConfig()
backtest_config = BacktestConfig()
data_config = DataConfig()
dashboard_config = DashboardConfig()
autotrading_config = AutoTradingConfig()
screener_config = ScreenerConfig()
ai_config = AIConfig()