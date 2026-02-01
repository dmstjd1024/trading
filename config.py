"""
설정 관리 모듈
한국투자증권 API 키 및 시스템 설정을 관리합니다.

사용법:
  1. 환경변수로 설정: KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO
  2. 또는 아래 값을 직접 수정
"""

import os
from dataclasses import dataclass, field


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

    # 모의투자 vs 실전투자
    is_paper: bool = True

    @property
    def base_url(self) -> str:
        if self.is_paper:
            return "https://openapivts.koreainvestment.com:29443"
        return "https://openapi.koreainvestment.com:9443"

    def validate(self) -> bool:
        if self.app_key == "YOUR_APP_KEY":
            print("[경고] API 키가 설정되지 않았습니다.")
            print("  환경변수 KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT_NO를 설정하거나")
            print("  config.py를 직접 수정해주세요.")
            return False
        return True


@dataclass
class BacktestConfig:
    """백테스팅 설정"""

    initial_capital: float = 10_000_000  # 초기 자본금 (1천만원)
    commission_rate: float = 0.00015     # 수수료율 (0.015%)
    slippage_rate: float = 0.001         # 슬리피지 (0.1%)
    tax_rate: float = 0.0018             # 거래세 (매도 시, 0.18% - 2025년 기준)


@dataclass
class DataConfig:
    """데이터 저장 설정"""

    data_dir: str = "./data"
    db_path: str = "./data/stocks.db"


# 전역 설정 인스턴스
kis_config = KISConfig()
backtest_config = BacktestConfig()
data_config = DataConfig()
