"""
AI 분석 대시보드 페이지
LLM 시장 분석, ML 가격 예측, 경제 뉴스 피드를 통합 표시합니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from config import ai_config, kis_config
from dashboard.state import init_session_state
from dashboard.components.styles import (
    inject_css,
    render_header,
    render_section_header,
    render_trading_mode_toggle,
)

init_session_state()
inject_css()

render_header("🧠 AI 분석", "LLM 시장 분석 + ML 가격 예측으로 투자 의사결정을 지원합니다")
render_trading_mode_toggle()


# ── 사이드바: AI 설정 ─────────────────────────────

st.sidebar.markdown("### ⚙️ AI 설정")

_provider_options = ["gemini", "openai", "anthropic"]
_provider_index = _provider_options.index(ai_config.ai_provider) if ai_config.ai_provider in _provider_options else 0

ai_provider = st.sidebar.selectbox(
    "LLM 프로바이더",
    options=_provider_options,
    index=_provider_index,
    format_func=lambda x: {"gemini": "Google Gemini (무료)", "openai": "OpenAI (유료)", "anthropic": "Anthropic (유료)"}.get(x, x),
)

_api_key_map = {"gemini": ai_config.gemini_api_key, "openai": ai_config.openai_api_key, "anthropic": ai_config.anthropic_api_key}
has_api_key = bool(_api_key_map.get(ai_provider, ""))

if has_api_key:
    st.sidebar.success(f"✅ {ai_provider.upper()} API 키 설정됨")
else:
    st.sidebar.warning(f"⚠️ {ai_provider.upper()} API 키 미설정 — 데모 모드")
    _env_name = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(ai_provider, "")
    st.sidebar.caption(
        f"환경변수 `{_env_name}`를 설정하세요."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 신호 가중치")

llm_w = st.sidebar.slider("LLM 분석 (%)", 0, 100, int(ai_config.llm_weight * 100), 5)
ml_w = st.sidebar.slider("ML 예측 (%)", 0, 100, int(ai_config.ml_weight * 100), 5)
tech_w = 100 - llm_w - ml_w
if tech_w < 0:
    st.sidebar.error("가중치 합이 100%를 초과합니다!")
    tech_w = 0
st.sidebar.info(f"기술적 분석: {tech_w}%")

demo_mode = st.sidebar.toggle("데모 모드", value=True, help="실제 API 호출 없이 샘플 데이터 사용")


# ── AI 분석 실행 ──────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def run_ai_analysis(_demo: bool = True):
    """AI 분석 실행 (1시간 캐시)"""
    from ai.data_sources import EconomicDataAggregator
    from ai.llm_analyzer import LLMAnalyzer
    from ai.ml_predictor import MLPredictor

    # 경제 데이터 수집
    aggregator = EconomicDataAggregator(ai_config.cache_dir)
    econ_data = aggregator.collect_all()

    # LLM 분석
    _key_map = {"gemini": ai_config.gemini_api_key, "openai": ai_config.openai_api_key, "anthropic": ai_config.anthropic_api_key}
    api_key = _key_map.get(ai_provider, "")
    llm = LLMAnalyzer(
        provider=ai_provider,
        api_key="" if _demo else api_key,
        model=ai_config.llm_model,
        cache_dir=ai_config.cache_dir,
    )
    analysis = llm.analyze()

    # ML 예측 (데모)
    ml = MLPredictor(model_dir=ai_config.ml_model_dir)
    prediction = ml._demo_prediction()

    return econ_data, analysis, prediction


# ── 메인 컨텐츠 ──────────────────────────────────

with st.spinner("🧠 AI 분석 중..."):
    econ_data, analysis, prediction = run_ai_analysis(_demo=demo_mode)


# ── 1. 시장 전망 카드 ────────────────────────────

render_section_header("🔮", "시장 전망")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="시장 전망",
        value=analysis.outlook_label_kr,
    )

with col2:
    st.metric(
        label="신뢰도",
        value=f"{analysis.confidence:.0%}",
    )

with col3:
    st.metric(
        label="리스크 레벨",
        value={"low": "🟢 낮음", "medium": "🟡 보통", "high": "🔴 높음"}.get(
            analysis.risk_level, "보통"
        ),
    )

with col4:
    st.metric(
        label="ML 예측",
        value=prediction.direction_label_kr,
    )

# 분석 근거
st.markdown(
    f"""
    <div class="info-card">
        <strong>📋 분석 근거</strong><br><br>
        {analysis.reasoning}
    </div>
    """,
    unsafe_allow_html=True,
)

# 핵심 요인
if analysis.key_factors:
    cols = st.columns(len(analysis.key_factors))
    for i, factor in enumerate(analysis.key_factors):
        with cols[i]:
            st.markdown(
                f"""
                <div class="metric-card" style="text-align:center; padding:0.8rem;">
                    <div class="label">핵심 요인 {i+1}</div>
                    <div style="font-size:0.9rem; font-weight:500; color:#1a1a2e;">{factor}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("")


# ── 2. AI 신호 현황 ─────────────────────────────

render_section_header("📊", "AI 복합 신호")

# 종합 점수 계산
llm_score = analysis.outlook_score
ml_score = prediction.direction_score

w_llm = llm_w / 100
w_ml = ml_w / 100
w_tech = tech_w / 100

# 기술적 점수 (데모용 중립)
tech_score = 0.1

combined = w_llm * llm_score + w_ml * ml_score + w_tech * tech_score

col1, col2 = st.columns([1, 1])

with col1:
    # 신호 게이지 차트
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=combined,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "종합 점수", "font": {"size": 16}},
        number={"font": {"size": 28}, "valueformat": "+.2f"},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1},
            "bar": {"color": "#667eea"},
            "steps": [
                {"range": [-1, -0.3], "color": "#ffcdd2"},
                {"range": [-0.3, 0.3], "color": "#fff9c4"},
                {"range": [0.3, 1], "color": "#c8e6c9"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 2},
                "thickness": 0.75,
                "value": combined,
            },
        },
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, key="gauge_combined", width="stretch")

with col2:
    # 각 신호 분해 바 차트
    signal_data = pd.DataFrame({
        "구성요소": ["LLM 분석", "ML 예측", "기술적 분석"],
        "점수": [llm_score, ml_score, tech_score],
        "가중치": [f"{llm_w}%", f"{ml_w}%", f"{tech_w}%"],
        "기여도": [w_llm * llm_score, w_ml * ml_score, w_tech * tech_score],
    })

    fig_bar = px.bar(
        signal_data,
        x="구성요소",
        y="기여도",
        color="점수",
        color_continuous_scale=["#ef5350", "#ffc107", "#4caf50"],
        range_color=[-1, 1],
        text="가중치",
    )
    fig_bar.update_layout(
        title="신호 분해",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, width="stretch")

# 최종 신호 표시
if combined > 0.3:
    signal_text = "🟢 **강한 매수** 신호"
    signal_desc = f"종합 점수 {combined:+.2f} — LLM과 ML 모두 긍정적 전망입니다."
elif combined > 0.15:
    signal_text = "🟡 **약한 매수** 신호"
    signal_desc = f"종합 점수 {combined:+.2f} — 일부 지표가 긍정적이나 확신도가 낮습니다."
elif combined < -0.3:
    signal_text = "🔴 **매도** 신호"
    signal_desc = f"종합 점수 {combined:+.2f} — 시장 전망이 부정적입니다. 리스크 관리 필요."
elif combined < -0.15:
    signal_text = "🟠 **약한 매도** 신호"
    signal_desc = f"종합 점수 {combined:+.2f} — 일부 부정적 신호가 감지되었습니다."
else:
    signal_text = "⚪ **관망** 신호"
    signal_desc = f"종합 점수 {combined:+.2f} — 뚜렷한 방향성이 없습니다. 대기 추천."

st.markdown(f"### {signal_text}")
st.caption(signal_desc)

st.markdown("")


# ── 3. 섹터별 전망 ──────────────────────────────

if analysis.sector_outlook:
    render_section_header("🏭", "섹터별 전망")

    sector_df = pd.DataFrame([
        {
            "섹터": sector,
            "전망": {"bullish": "🟢 강세", "neutral": "🟡 보합", "bearish": "🔴 약세"}.get(
                outlook, "보합"
            ),
            "점수": {"bullish": 1, "neutral": 0, "bearish": -1}.get(outlook, 0),
        }
        for sector, outlook in analysis.sector_outlook.items()
    ])

    # 섹터 히트맵
    fig_sector = px.bar(
        sector_df,
        x="섹터",
        y="점수",
        color="점수",
        color_continuous_scale=["#ef5350", "#ffc107", "#4caf50"],
        range_color=[-1, 1],
        text="전망",
    )
    fig_sector.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
        yaxis_title="전망 점수",
    )
    fig_sector.update_traces(textposition="outside")
    st.plotly_chart(fig_sector, width="stretch")

    st.markdown("")


# ── 4. ML 피처 중요도 ────────────────────────────

render_section_header("🔬", "ML 피처 중요도")

if prediction.features_importance:
    # 피처명 한글 매핑
    feature_names_kr = {
        "feat_rsi": "RSI(14)",
        "feat_macd": "MACD",
        "feat_macd_signal": "MACD 시그널",
        "feat_macd_hist": "MACD 히스토그램",
        "feat_bb_position": "볼린저밴드 위치",
        "feat_ma5_slope": "MA5 기울기",
        "feat_ma20_slope": "MA20 기울기",
        "feat_ma60_slope": "MA60 기울기",
        "feat_price_ma20_gap": "MA20 괴리율",
        "feat_volume_ratio": "거래량 비율",
        "feat_atr": "ATR(변동성)",
        "feat_return_1d": "1일 수익률",
        "feat_return_5d": "5일 수익률",
        "feat_return_20d": "20일 수익률",
        "feat_volatility": "변동성",
        "feat_sentiment_score": "LLM 센티멘트",
        "feat_sentiment_conf": "LLM 신뢰도",
        "feat_risk_score": "리스크 점수",
        "feat_kospi_change": "코스피 변화",
        "feat_vix_change": "VIX 변화",
        "feat_usd_krw_change": "환율 변화",
        "feat_oil_change": "유가 변화",
    }

    importance_df = pd.DataFrame([
        {
            "피처": feature_names_kr.get(feat, feat),
            "중요도": score,
        }
        for feat, score in prediction.features_importance.items()
    ]).sort_values("중요도", ascending=True)

    fig_imp = px.bar(
        importance_df,
        x="중요도",
        y="피처",
        orientation="h",
        color="중요도",
        color_continuous_scale="Viridis",
    )
    fig_imp.update_layout(
        height=max(250, len(importance_df) * 35),
        margin=dict(l=20, r=20, t=10, b=20),
        showlegend=False,
        yaxis_title="",
    )
    st.plotly_chart(fig_imp, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("예측 방향", prediction.direction_label_kr)
    with col2:
        st.metric("예측 확률", f"{prediction.probability:.0%}")

st.markdown("")


# ── 5. 거시경제 지표 ─────────────────────────────

render_section_header("🌍", "거시경제 현황")

macro = econ_data["macro"]

col1, col2, col3 = st.columns(3)

with col1:
    delta_color = "normal" if macro.kospi_change >= 0 else "inverse"
    st.metric(
        label="코스피",
        value=f"{macro.kospi:,.2f}",
        delta=f"{macro.kospi_change:+.2f}%",
        delta_color=delta_color,
    )
    st.metric(
        label="원/달러 환율",
        value=f"{macro.usd_krw:,.1f}",
        delta=f"{macro.usd_krw_change:+.2f}%",
        delta_color="inverse" if macro.usd_krw_change >= 0 else "normal",
    )

with col2:
    delta_color = "normal" if macro.kosdaq_change >= 0 else "inverse"
    st.metric(
        label="코스닥",
        value=f"{macro.kosdaq:,.2f}",
        delta=f"{macro.kosdaq_change:+.2f}%",
        delta_color=delta_color,
    )
    st.metric(
        label="WTI 유가",
        value=f"${macro.wti_oil:.2f}",
        delta=f"{macro.wti_oil_change:+.2f}%",
    )

with col3:
    st.metric(
        label="VIX (공포지수)",
        value=f"{macro.vix:.2f}",
        delta=f"{macro.vix_change:+.2f}%",
        delta_color="inverse" if macro.vix_change >= 0 else "normal",
    )
    st.metric(
        label="미국 10년물 금리",
        value=f"{macro.us_10y_yield:.3f}%",
        delta=f"{macro.us_10y_yield_change:+.3f}%p",
    )

st.markdown("")


# ── 6. 시장 심리 ─────────────────────────────────

render_section_header("💭", "시장 심리")

sentiment = econ_data["sentiment"]

col1, col2 = st.columns([1, 1])

with col1:
    # Fear & Greed 게이지
    fg_color = "#4caf50" if sentiment.fear_greed_index > 60 else (
        "#ef5350" if sentiment.fear_greed_index < 40 else "#ffc107"
    )
    fig_fg = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment.fear_greed_index,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"공포 & 탐욕 지수 ({sentiment.fear_greed_label})", "font": {"size": 14}},
        number={"font": {"size": 32}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": fg_color},
            "steps": [
                {"range": [0, 25], "color": "#ffcdd2"},
                {"range": [25, 45], "color": "#ffe0b2"},
                {"range": [45, 55], "color": "#fff9c4"},
                {"range": [55, 75], "color": "#dcedc8"},
                {"range": [75, 100], "color": "#c8e6c9"},
            ],
        },
    ))
    fig_fg.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_fg, width="stretch")

with col2:
    # 투자자별 매매 동향
    investor_data = pd.DataFrame({
        "투자자": ["외국인", "기관", "개인"],
        "순매수(억원)": [
            sentiment.foreign_net_buy,
            sentiment.institution_net_buy,
            sentiment.individual_net_buy,
        ],
    })

    fig_inv = px.bar(
        investor_data,
        x="투자자",
        y="순매수(억원)",
        color="순매수(억원)",
        color_continuous_scale=["#ef5350", "#ffc107", "#4caf50"],
        text_auto=True,
    )
    fig_inv.update_layout(
        title="투자자별 매매 동향",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_inv, width="stretch")

st.markdown("")


# ── 7. 경제 뉴스 피드 ────────────────────────────

render_section_header("📰", "경제 뉴스 피드")

news_items = econ_data["news"]

if news_items:
    for i, news in enumerate(news_items[:10]):
        with st.container():
            cols = st.columns([5, 1, 1])
            with cols[0]:
                title = news.title
                if news.url:
                    title = f"[{news.title}]({news.url})"
                st.markdown(f"**{title}**")
                if news.summary:
                    st.caption(news.summary[:150])
            with cols[1]:
                st.caption(news.source)
            with cols[2]:
                st.caption(news.published.strftime("%m/%d %H:%M"))

            if i < len(news_items) - 1:
                st.divider()
else:
    st.info("뉴스 데이터가 없습니다.")


st.markdown("")

# ── 8. AI 설정 안내 ──────────────────────────────

with st.expander("🔑 AI API 키 설정 방법"):
    st.markdown("""
**🌟 Google Gemini 사용 시 (무료 추천):**
    """)
    st.code("""
export GEMINI_API_KEY="AIza..."
export AI_PROVIDER="gemini"
export AI_LLM_MODEL="gemini-2.0-flash"
    """, language="bash")
    st.caption("발급: https://aistudio.google.com/apikey → API 키 만들기")

    st.markdown("""
**OpenAI (GPT) 사용 시:**
    """)
    st.code("""
export OPENAI_API_KEY="sk-..."
export AI_PROVIDER="openai"
export AI_LLM_MODEL="gpt-4o-mini"
    """, language="bash")

    st.markdown("""
**Anthropic (Claude) 사용 시:**
    """)
    st.code("""
export ANTHROPIC_API_KEY="sk-ant-..."
export AI_PROVIDER="anthropic"
export AI_LLM_MODEL="claude-sonnet-4-20250514"
    """, language="bash")

    st.caption(
        "API 키를 설정하면 실제 경제 데이터를 기반으로 LLM이 시장을 분석합니다. "
        "미설정 시 데모 데이터로 동작합니다."
    )

with st.expander("📖 AI 전략 사용 가이드"):
    st.markdown("""
### AI 복합 전략이란?

**3가지 분석을 결합**하여 최종 매매 신호를 생성합니다:

1. **LLM 시장 분석 (40%)**: 경제 뉴스, 거시지표, 심리 데이터를 GPT/Claude가 종합 분석
2. **ML 가격 예측 (40%)**: XGBoost 모델이 기술적 지표 기반으로 향후 5거래일 방향 예측
3. **기술적 확인 (20%)**: RSI, MACD, 이동평균 등 전통적 기술 지표로 확인

### 리스크 관리
- 시장 리스크 "높음" → 매수 비율 자동 축소 (최대 30%)
- ML 신뢰도 60% 미만 → 매수 비율 추가 축소
- 연속 3회 손실 → 자동 쿨다운 (1일 거래 중단)

### 백테스트에서 테스트
좌측 메뉴의 **백테스트** 페이지에서 `AI Composite(LLM+ML+Tech)` 전략을 선택하여 과거 성과를 확인할 수 있습니다.
    """)
