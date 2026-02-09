"""
공통 스타일 및 CSS
"""

import streamlit as st


CUSTOM_CSS = """
<style>
/* 전체 폰트 & 배경 */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Noto Sans KR', sans-serif;
}

/* Material Icons 폰트 보호 — 아이콘이 텍스트로 깨지는 것을 방지 */
[data-testid="stIconMaterial"],
[class*="material-symbols"],
[class*="material-icons"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

/* 메인 헤더 */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
}

.main-header h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 700;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    opacity: 0.85;
    font-size: 0.95rem;
}

/* 메트릭 카드 */
.metric-card {
    background: white;
    border: 1px solid #e8ecf1;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    transition: transform 0.2s, box-shadow 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
}

.metric-card .label {
    font-size: 0.8rem;
    color: #8892a4;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.2;
}

.metric-card .delta {
    font-size: 0.85rem;
    margin-top: 0.3rem;
    font-weight: 500;
}

.metric-card .delta.positive { color: #ef5350; }
.metric-card .delta.negative { color: #26a69a; }

/* 상태 뱃지 */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
}

.badge-success {
    background: #e8f5e9;
    color: #2e7d32;
}

.badge-danger {
    background: #ffebee;
    color: #c62828;
}

.badge-info {
    background: #e3f2fd;
    color: #1565c0;
}

.badge-warning {
    background: #fff3e0;
    color: #e65100;
}

/* 섹션 헤더 */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #667eea;
}

.section-header h3 {
    margin: 0;
    font-size: 1.15rem;
    font-weight: 600;
    color: #1a1a2e;
}

/* 정보 카드 */
.info-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* 메뉴 카드 */
.nav-card {
    background: white;
    border: 1px solid #e8ecf1;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.nav-card:hover {
    border-color: #667eea;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15);
    transform: translateY(-2px);
}

/* 네비게이션 카드 링크 */
a.nav-card-link {
    text-decoration: none !important;
    color: inherit !important;
    display: block;
}

a.nav-card-link .nav-card {
    cursor: pointer;
}

.nav-card .icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}

.nav-card .title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 0.3rem;
}

.nav-card .desc {
    font-size: 0.85rem;
    color: #8892a4;
}

/* Streamlit 기본 메트릭 커스텀 */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e8ecf1;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #8892a4 !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
}

/* 테이블 스타일 */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9fe 0%, #eef0f8 100%);
    overflow-x: visible !important;
    overflow-y: auto !important;
}

[data-testid="stSidebar"] > div {
    overflow-x: visible !important;
    word-wrap: break-word !important;
    word-break: break-word !important;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stDateInput label {
    font-weight: 500;
    color: #3d3d5c;
}

/* 사이드바 텍스트 오버플로우 방지 */
[data-testid="stSidebar"] * {
    max-width: 100% !important;
    box-sizing: border-box !important;
}

[data-testid="stSidebar"] [class*="element-container"] {
    overflow: visible !important;
    word-wrap: break-word !important;
}

/* 버튼 */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
    transform: translateY(-1px);
}

/* 탭 */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 500;
}

/* 디바이더 */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #d4d8e8, transparent);
    margin: 1.5rem 0;
}

/* 경고/정보 박스 */
.stAlert {
    border-radius: 12px !important;
}

/* Expander 스타일 수정 */
[data-testid="stExpander"] {
    margin-top: 1rem;
    margin-bottom: 1rem;
}

[data-testid="stExpander"] summary > span > div {
    font-size: 1rem !important;
    font-weight: 600 !important;
}

[data-testid="stExpander"] details {
    border-radius: 8px !important;
    border: 1px solid #e8ecf1 !important;
}

/* Expander 화살표: Streamlit 기본 아이콘 사용 */

/* 사이드바 스타일 수정 */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    overflow: visible !important;
    word-wrap: break-word !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 0 !important;
}

/* 사이드바 네비게이션 링크 */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a {
    display: block;
    padding: 0.5rem 0.75rem;
    margin: 0.25rem 0;
    border-radius: 6px;
    text-decoration: none;
    color: #3d3d5c;
    transition: all 0.2s;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a:hover {
    background: rgba(102, 126, 234, 0.1);
    color: #667eea;
}

/* 사이드바 헤더 텍스트 오버플로우 방지 */
[data-testid="stSidebar"] > div:first-child {
    overflow: visible !important;
    word-break: break-word !important;
}

/* Expander 내용 영역 */
[data-testid="stExpander"] > div {
    padding: 1rem !important;
    border-top: 1px solid #e8ecf1;
    margin-top: 0.5rem;
}

/* 코드 블록 스타일 */
[data-testid="stExpander"] code {
    background: #f8f9fa;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-size: 0.9em;
}

[data-testid="stExpander"] pre {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    overflow-x: auto;
    border: 1px solid #e8ecf1;
}
</style>
"""


def inject_css():
    """페이지에 커스텀 CSS를 주입합니다."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = ""):
    """그라데이션 헤더를 렌더링합니다."""
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, delta: str = "", delta_type: str = ""):
    """커스텀 메트릭 카드를 렌더링합니다."""
    delta_class = ""
    if delta_type == "positive":
        delta_class = "positive"
    elif delta_type == "negative":
        delta_class = "negative"

    delta_html = f'<div class="delta {delta_class}">{delta}</div>' if delta else ""

    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(icon: str, title: str):
    """섹션 헤더를 렌더링합니다."""
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size: 1.3rem;">{icon}</span>
        <h3>{title}</h3>
    </div>
    """, unsafe_allow_html=True)


def render_nav_card(icon: str, title: str, description: str, url: str = ""):
    """네비게이션 카드를 렌더링합니다. url이 주어지면 클릭 시 해당 페이지로 이동합니다."""
    if url:
        st.markdown(f"""
        <a href="/{url}" target="_self" class="nav-card-link">
            <div class="nav-card">
                <div class="icon">{icon}</div>
                <div class="title">{title}</div>
                <div class="desc">{description}</div>
            </div>
        </a>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="nav-card">
            <div class="icon">{icon}</div>
            <div class="title">{title}</div>
            <div class="desc">{description}</div>
        </div>
        """, unsafe_allow_html=True)


def render_badge(text: str, variant: str = "info"):
    """상태 뱃지를 렌더링합니다."""
    return f'<span class="badge badge-{variant}">{text}</span>'


def render_trading_mode_toggle():
    """모의투자/실전투자 전환 토글을 상단 우측에 렌더링합니다."""
    from config import kis_config

    # 세션 상태 초기화
    if "trading_mode_real" not in st.session_state:
        st.session_state.trading_mode_real = not kis_config.is_paper
    if "show_real_warning" not in st.session_state:
        st.session_state.show_real_warning = False

    # 상단 우측에 토글 배치
    cols = st.columns([4, 1, 1])
    with cols[1]:
        is_real = st.toggle(
            "실전" if st.session_state.trading_mode_real else "모의",
            value=st.session_state.trading_mode_real,
            key="_trading_mode_toggle",
            help="ON: 실전투자 (실제 주문 발생!), OFF: 모의투자 (테스트용)",
        )
    with cols[2]:
        if is_real:
            st.markdown(
                '<div style="padding-top:0.3rem;">'
                '<span class="badge badge-danger">🔴 실전투자</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="padding-top:0.3rem;">'
                '<span class="badge badge-info">🔵 모의투자</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    # 실전투자로 전환 시 경고 표시
    if is_real and not st.session_state.trading_mode_real:
        st.warning(
            "⚠️ **실전투자 모드로 전환합니다.** "
            "실제 계좌에서 주문이 실행되며, 손실이 발생할 수 있습니다. "
            "충분한 테스트 후 사용해주세요.",
            icon="⚠️",
        )
        st.session_state.trading_mode_real = True
        kis_config.is_paper = False

        # 토큰 캐시 삭제 (서버가 다르므로)
        from pathlib import Path
        cache_path = Path(__file__).parent.parent.parent / ".token_cache.json"
        if cache_path.exists():
            cache_path.unlink()

    # 모의투자로 전환 시
    elif not is_real and st.session_state.trading_mode_real:
        st.info("🔵 모의투자 모드로 전환되었습니다. 테스트용 환경에서 실행됩니다.")
        st.session_state.trading_mode_real = False
        kis_config.is_paper = True

        # 토큰 캐시 삭제 (서버가 다르므로)
        from pathlib import Path
        cache_path = Path(__file__).parent.parent.parent / ".token_cache.json"
        if cache_path.exists():
            cache_path.unlink()

    # 현재 config에 반영
    kis_config.is_paper = not is_real

    return is_real
