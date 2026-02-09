"""
종목 스크리닝 페이지
멀티팩터 스크리닝으로 종목을 발굴합니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import kis_config, screener_config
from dashboard.state import init_session_state
from dashboard.components.styles import inject_css, render_header, render_trading_mode_toggle

init_session_state()
inject_css()

render_header("🔍 종목 스크리닝", "멀티팩터 분석으로 유망 종목을 자동 선별합니다")
render_trading_mode_toggle()

# ── 사이드바: 설정 ──────────────────────────────

st.sidebar.markdown("### ⚙️ 스크리닝 설정")

# 시장 선택
market_options = {
    "전체": "0000",
    "코스피": "0001",
    "코스닥": "1001",
    "코스피200": "2001",
}
market_label = st.sidebar.selectbox("대상 시장", options=list(market_options.keys()))
market_code = market_options[market_label]

# 상위 종목 수
top_n = st.sidebar.slider("선정 종목 수", min_value=5, max_value=30, value=10, step=5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 팩터 가중치")

# 기술 vs 펀더멘탈 비율
tech_pct = st.sidebar.slider(
    "기술적 팩터 비중 (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=10,
    help="나머지는 펀더멘탈 팩터에 할당됩니다",
)
tech_weight = tech_pct / 100
fund_weight = 1.0 - tech_weight

col1, col2 = st.sidebar.columns(2)
col1.metric("기술적", f"{tech_pct}%")
col2.metric("펀더멘탈", f"{100 - tech_pct}%")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 필터 조건")

min_market_cap = st.sidebar.number_input(
    "최소 시가총액 (억원)",
    min_value=0,
    max_value=100000,
    value=screener_config.min_market_cap,
    step=500,
)

min_volume = st.sidebar.number_input(
    "최소 거래량",
    min_value=0,
    max_value=10_000_000,
    value=screener_config.min_volume,
    step=50000,
)

# ── 스크리닝 실행 ─────────────────────────────

if "screening_results" not in st.session_state:
    st.session_state.screening_results = None

run_clicked = st.button("🚀 스크리닝 실행", type="primary", width="stretch")

if run_clicked:
    if not kis_config.validate():
        st.error("API 키가 설정되지 않았습니다. `.env` 파일을 확인해주세요.")
    else:
        with st.spinner("스크리닝 중... (약 1~2분 소요)"):
            try:
                from screener import StockScreener
                from config import ScreenerConfig

                config = ScreenerConfig(
                    top_n=top_n,
                    market=market_code,
                    tech_weight=tech_weight,
                    fund_weight=fund_weight,
                    min_market_cap=min_market_cap,
                    min_volume=min_volume,
                )
                screener = StockScreener(config=config)
                results = screener.run(
                    top_n=top_n,
                    market=market_code,
                    tech_weight=tech_weight,
                    fund_weight=fund_weight,
                )
                st.session_state.screening_results = results
            except Exception as e:
                st.error(f"스크리닝 오류: {e}")
                st.session_state.screening_results = None

# ── 결과 표시 ─────────────────────────────────

results = st.session_state.screening_results

if results:
    st.success(f"스크리닝 완료! 상위 {len(results)}개 종목이 선정되었습니다.")

    # ── 요약 메트릭 ──
    cols = st.columns(4)
    avg_score = sum(r.get("total_score", 0) for r in results) / len(results)
    avg_per = sum(r.get("per", 0) for r in results if r.get("per", 0) > 0) / max(1, sum(1 for r in results if r.get("per", 0) > 0))
    avg_rsi = sum(r.get("rsi", 0) for r in results) / len(results)
    total_codes = len(results)

    cols[0].metric("선정 종목", f"{total_codes}개")
    cols[1].metric("평균 총점", f"{avg_score:.1f}")
    cols[2].metric("평균 PER", f"{avg_per:.1f}")
    cols[3].metric("평균 RSI", f"{avg_rsi:.1f}")

    st.markdown("---")

    # ── 종합 점수 바 차트 ──
    st.subheader("종합 점수")

    df = pd.DataFrame(results)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["name"].str[:8],
        y=df["total_score"],
        marker_color=px.colors.sequential.Viridis_r[:len(df)],
        text=df["total_score"].round(1),
        textposition="outside",
    ))
    fig.update_layout(
        xaxis_title="종목",
        yaxis_title="종합 점수",
        yaxis_range=[0, 110],
        height=400,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, width="stretch")

    # ── 기술 vs 펀더멘탈 비교 ──
    st.subheader("기술적 vs 펀더멘탈 점수")

    col1, col2 = st.columns(2)

    with col1:
        fig_tech = go.Figure()
        fig_tech.add_trace(go.Bar(
            y=df["name"].str[:8],
            x=df.get("tech_score", pd.Series([0] * len(df))),
            orientation="h",
            name="기술적",
            marker_color="#636EFA",
        ))
        fig_tech.update_layout(
            title="기술적 점수",
            xaxis_range=[0, 100],
            height=max(300, len(df) * 30),
            margin=dict(l=0, t=40),
        )
        st.plotly_chart(fig_tech, width="stretch")

    with col2:
        fig_fund = go.Figure()
        fig_fund.add_trace(go.Bar(
            y=df["name"].str[:8],
            x=df.get("fund_score", pd.Series([0] * len(df))),
            orientation="h",
            name="펀더멘탈",
            marker_color="#EF553B",
        ))
        fig_fund.update_layout(
            title="펀더멘탈 점수",
            xaxis_range=[0, 100],
            height=max(300, len(df) * 30),
            margin=dict(l=0, t=40),
        )
        st.plotly_chart(fig_fund, width="stretch")

    # ── 상세 테이블 ──
    st.subheader("상세 결과")

    display_cols = {
        "code": "종목코드",
        "name": "종목명",
        "price": "현재가",
        "change_rate": "등락률(%)",
        "total_score": "총점",
        "tech_score": "기술점수",
        "fund_score": "펀더멘탈점수",
        "per": "PER",
        "pbr": "PBR",
        "roe": "ROE(%)",
        "rsi": "RSI",
        "volume_ratio": "거래량비율",
        "momentum_20d": "20일모멘텀(%)",
    }

    # 존재하는 컬럼만 선택
    avail_cols = [c for c in display_cols.keys() if c in df.columns]
    display_df = df[avail_cols].copy()
    display_df.columns = [display_cols[c] for c in avail_cols]

    # 숫자 포맷
    if "현재가" in display_df.columns:
        display_df["현재가"] = display_df["현재가"].apply(lambda x: f"{x:,}")
    for col in ["총점", "기술점수", "펀더멘탈점수", "RSI"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(1)
    for col in ["PER", "PBR", "ROE(%)", "등락률(%)", "거래량비율", "20일모멘텀(%)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)

    st.dataframe(display_df, width="stretch", hide_index=True)

    # ── 자동매매 연동 ──
    st.markdown("---")
    st.subheader("자동매매 연동")

    selected_codes = [r["code"] for r in results]
    st.code(", ".join(selected_codes), language=None)

    if st.button("📋 자동매매 종목에 적용", type="secondary"):
        st.session_state.selected_stocks = selected_codes
        if hasattr(st.session_state, "autotrading_stocks"):
            st.session_state.autotrading_stocks = selected_codes
        st.success(f"{len(selected_codes)}개 종목이 자동매매 목록에 적용되었습니다!")
        st.info("자동매매 페이지에서 확인하세요.")

elif results is not None and len(results) == 0:
    st.warning("스크리닝 결과가 없습니다. 필터 조건을 완화해보세요.")
else:
    # 초기 상태: 안내 메시지
    st.info("좌측 설정을 조정한 후 **스크리닝 실행** 버튼을 눌러주세요.")

    with st.expander("멀티팩터 스크리닝이란?", expanded=True):
        st.markdown("""
**멀티팩터 스크리닝**은 여러 지표를 종합적으로 분석하여 투자 유망 종목을 자동으로 선별하는 방법입니다.

**기술적 팩터** (시장 심리 분석)
- **RSI**: 과매도(30 이하) 구간의 종목 = 반등 기회
- **거래량 비율**: 최근 거래량 증가 = 시장 관심 증가
- **20일 모멘텀**: 적당한 상승세 = 추세 확인

**펀더멘탈 팩터** (기업 가치 분석)
- **PER**: 낮을수록 이익 대비 저평가
- **PBR**: 낮을수록 자산 대비 저평가
- **ROE**: 높을수록 자기자본 수익성 우수

각 팩터를 0~100점으로 정규화한 뒤 가중 평균으로 종합 점수를 산출합니다.
        """)
