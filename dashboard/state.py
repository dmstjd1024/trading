"""
세션 상태 관리
Streamlit 세션 상태를 관리합니다.
"""

from __future__ import annotations

import streamlit as st
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class AppState:
    """애플리케이션 상태"""

    # 자동매매 상태
    autotrading_enabled: bool = False
    autotrading_strategy: str = "golden_cross"
    autotrading_stocks: List[str] = field(default_factory=lambda: ["005930"])

    # 마지막 업데이트 시간
    last_update: Optional[datetime] = None

    # 백테스트 결과 캐시
    last_backtest_result: Optional[dict] = None


def init_session_state() -> None:
    """세션 상태 초기화"""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()

    if "autotrading_enabled" not in st.session_state:
        st.session_state.autotrading_enabled = False

    if "selected_stocks" not in st.session_state:
        st.session_state.selected_stocks = ["005930"]

    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = {}

    if "trading_logs" not in st.session_state:
        st.session_state.trading_logs = []


def get_state() -> AppState:
    """현재 상태 반환"""
    init_session_state()
    return st.session_state.app_state


def set_autotrading_enabled(enabled: bool) -> None:
    """자동매매 활성화 설정"""
    st.session_state.autotrading_enabled = enabled
    st.session_state.app_state.autotrading_enabled = enabled


def is_autotrading_enabled() -> bool:
    """자동매매 활성화 여부"""
    init_session_state()
    return st.session_state.autotrading_enabled


def add_trading_log(log: dict) -> None:
    """거래 로그 추가"""
    init_session_state()
    st.session_state.trading_logs.insert(0, log)
    # 최대 100개만 유지
    if len(st.session_state.trading_logs) > 100:
        st.session_state.trading_logs = st.session_state.trading_logs[:100]


def get_trading_logs() -> List[dict]:
    """거래 로그 반환"""
    init_session_state()
    return st.session_state.trading_logs


def set_backtest_result(key: str, result: dict) -> None:
    """백테스트 결과 저장"""
    init_session_state()
    st.session_state.backtest_results[key] = result


def get_backtest_result(key: str) -> Optional[dict]:
    """백테스트 결과 반환"""
    init_session_state()
    return st.session_state.backtest_results.get(key)
