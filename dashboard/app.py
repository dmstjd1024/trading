"""
Streamlit 대시보드 메인 앱
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from config import dashboard_config

# 페이지 설정
st.set_page_config(
    page_title=dashboard_config.page_title,
    page_icon=dashboard_config.page_icon,
    layout=dashboard_config.layout,
    initial_sidebar_state="expanded",
)

# 세션 상태 초기화 & 스타일 주입
from dashboard.state import init_session_state
from dashboard.components.styles import inject_css

init_session_state()
inject_css()


# ── 홈 페이지 ──────────────────────────────────────

def home():
    from dashboard.components.styles import render_header, render_nav_card, render_section_header, render_trading_mode_toggle

    render_header(
        "한국주식 트레이딩 시스템",
        "데이터 수집 / 백테스팅 / 자동매매를 하나의 대시보드에서 관리하세요",
    )
    render_trading_mode_toggle()

    # 상태 카드
    render_section_header("📊", "시스템 현황")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="시스템 상태", value="✅ 정상")

    with col2:
        from dashboard.state import is_autotrading_enabled
        autotrading_on = is_autotrading_enabled()
        st.metric(
            label="자동매매",
            value="🟢 ON" if autotrading_on else "⏸️ OFF",
        )

    with col3:
        from datetime import datetime
        now = datetime.now()
        hour = now.hour
        market_open = 9 <= hour < 16
        market_status = "장 운영중" if market_open else "장 마감"
        st.metric(
            label=f"현재 시간 ({market_status})",
            value=now.strftime("%H:%M:%S"),
        )

    st.markdown("")

    # 메뉴 카드
    render_section_header("🧭", "바로가기")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_nav_card("💰", "포트폴리오", "계좌 잔고, 보유 종목, 수익률 확인", url="portfolio")

    with col2:
        render_nav_card("🔬", "백테스트", "과거 데이터로 전략 시뮬레이션", url="backtest")

    with col3:
        render_nav_card("🤖", "자동매매", "전략 자동 실행 설정 및 로그", url="autotrading")

    with col4:
        render_nav_card("🔍", "종목 스크리닝", "멀티팩터로 유망 종목 발굴", url="screening")

    # AI 분석 카드 (단독 행)
    col_ai, _, _, _ = st.columns(4)
    with col_ai:
        render_nav_card("🧠", "AI 분석", "LLM+ML 기반 시장 분석 & 매매 신호", url="ai-analysis")

    st.markdown("")

    # 시작 가이드
    render_section_header("🚀", "시작하기")

    st.markdown("""
    <div class="info-card">
        <strong>1단계:</strong> 왼쪽 사이드바에서 <b>백테스트</b> 페이지로 이동<br>
        <strong>2단계:</strong> '데모 데이터 사용'을 체크하고 전략을 선택<br>
        <strong>3단계:</strong> '백테스트 실행' 버튼을 눌러 결과 확인<br><br>
        <span style="color: #667eea; font-weight: 600;">
            API 키가 없어도 데모 모드로 모든 기능을 체험할 수 있습니다.
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔑 API 키 설정 방법"):
        st.markdown("**실제 데이터 사용 시 필요합니다.**")
        st.code("""
# 환경변수 설정
export KIS_APP_KEY="your_app_key"
export KIS_APP_SECRET="your_app_secret"
export KIS_ACCOUNT_NO="00000000-00"
        """, language="bash")
        st.caption("한국투자증권 OpenAPI 서비스에서 발급받을 수 있습니다.")


# ── 페이지 네비게이션 (한글 메뉴) ──────────────────

pg = st.navigation([
    st.Page(home, title="홈", icon="🏠", default=True, url_path="home"),
    st.Page("pages/1_portfolio.py", title="포트폴리오", icon="💰", url_path="portfolio"),
    st.Page("pages/2_backtest.py", title="백테스트", icon="🔬", url_path="backtest"),
    st.Page("pages/3_autotrading.py", title="자동매매", icon="🤖", url_path="autotrading"),
    st.Page("pages/4_screening.py", title="종목 스크리닝", icon="🔍", url_path="screening"),
    st.Page("pages/5_ai_analysis.py", title="AI 분석", icon="🧠", url_path="ai-analysis"),
])

pg.run()
