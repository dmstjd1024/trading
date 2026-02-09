"""
포트폴리오 페이지
계좌 잔고, 보유 종목, 수익률 차트 표시
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from datetime import datetime

from config import kis_config
from api_client import KISClient
from dashboard.state import init_session_state
from dashboard.components.charts import (
    create_portfolio_pie_chart,
    create_profit_loss_bar_chart,
)
from dashboard.components.styles import inject_css, render_header, render_section_header, render_trading_mode_toggle

init_session_state()
inject_css()

render_header("💰 포트폴리오", "내 계좌 현황과 보유 종목을 한눈에 확인하세요")
render_trading_mode_toggle()

# ── 데모 데이터 ────────────────────────────────────

DEMO_BALANCE = {
    "total_balance": 15_000_000,
    "cash_balance": 5_000_000,
    "stock_balance": 10_000_000,
    "profit_loss": 500_000,
    "profit_loss_rate": 3.45,
}

DEMO_HOLDINGS = [
    {
        "stock_code": "005930",
        "stock_name": "삼성전자",
        "quantity": 100,
        "avg_price": 72000,
        "current_price": 75000,
        "profit_loss": 300_000,
        "profit_loss_rate": 4.17,
    },
    {
        "stock_code": "000660",
        "stock_name": "SK하이닉스",
        "quantity": 30,
        "avg_price": 150000,
        "current_price": 156000,
        "profit_loss": 180_000,
        "profit_loss_rate": 4.0,
    },
    {
        "stock_code": "035720",
        "stock_name": "카카오",
        "quantity": 50,
        "avg_price": 45000,
        "current_price": 44600,
        "profit_loss": -20_000,
        "profit_loss_rate": -0.89,
    },
]

# ── 데이터 로드 ────────────────────────────────────

balance = None
holdings = None

if kis_config.validate():
    try:
        client = KISClient()
        account_balance = client.get_account_balance()
        balance = {
            "total_balance": account_balance.total_balance,
            "cash_balance": account_balance.cash_balance,
            "stock_balance": account_balance.stock_balance,
            "profit_loss": account_balance.profit_loss,
            "profit_loss_rate": account_balance.profit_loss_rate,
        }

        holdings_data = client.get_holdings()
        holdings = [
            {
                "stock_code": h.stock_code,
                "stock_name": h.stock_name,
                "quantity": h.quantity,
                "avg_price": h.avg_price,
                "current_price": h.current_price,
                "profit_loss": h.profit_loss,
                "profit_loss_rate": h.profit_loss_rate,
            }
            for h in holdings_data
        ]

        # API가 빈 데이터(총 자산 0원)를 반환하면 데모로 폴백
        if balance["total_balance"] == 0 and balance["cash_balance"] == 0:
            st.info("📊 계좌에 잔고가 없어 데모 데이터로 표시합니다.")
            balance = DEMO_BALANCE
            holdings = DEMO_HOLDINGS

    except Exception as e:
        st.warning(f"⚠️ API 연결 실패 — 데모 데이터로 표시합니다. ({e})")
        balance = DEMO_BALANCE
        holdings = DEMO_HOLDINGS
else:
    st.info("🔑 API 키 미설정 — 데모 데이터로 표시합니다.")
    balance = DEMO_BALANCE
    holdings = DEMO_HOLDINGS

# ── 계좌 요약 ──────────────────────────────────────

render_section_header("📋", "계좌 요약")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="총 평가금액",
        value=f"{balance['total_balance']:,.0f}원",
    )

with col2:
    st.metric(
        label="현금 잔고",
        value=f"{balance['cash_balance']:,.0f}원",
    )

with col3:
    st.metric(
        label="주식 평가금액",
        value=f"{balance['stock_balance']:,.0f}원",
    )

with col4:
    profit_color = "normal" if balance["profit_loss"] >= 0 else "inverse"
    st.metric(
        label="평가손익",
        value=f"{balance['profit_loss']:+,.0f}원",
        delta=f"{balance['profit_loss_rate']:+.2f}%",
        delta_color=profit_color,
    )

st.markdown("")

# ── 차트 ───────────────────────────────────────────

render_section_header("📈", "시각화")

tab1, tab2 = st.tabs(["포트폴리오 구성", "종목별 손익"])

with tab1:
    pie_chart = create_portfolio_pie_chart(holdings, balance["cash_balance"])
    st.plotly_chart(pie_chart, width="stretch")

with tab2:
    bar_chart = create_profit_loss_bar_chart(holdings)
    st.plotly_chart(bar_chart, width="stretch")

st.markdown("")

# ── 보유 종목 테이블 ───────────────────────────────

render_section_header("📄", "보유 종목 상세")

if holdings:
    df = pd.DataFrame(holdings)
    df = df.rename(columns={
        "stock_code": "종목코드",
        "stock_name": "종목명",
        "quantity": "보유수량",
        "avg_price": "평균단가",
        "current_price": "현재가",
        "profit_loss": "평가손익",
        "profit_loss_rate": "수익률(%)",
    })

    df["평균단가"] = df["평균단가"].apply(lambda x: f"{x:,.0f}")
    df["현재가"] = df["현재가"].apply(lambda x: f"{x:,.0f}")
    df["평가손익"] = df["평가손익"].apply(lambda x: f"{x:+,.0f}")
    df["수익률(%)"] = df["수익률(%)"].apply(lambda x: f"{x:+.2f}")

    st.dataframe(df, width="stretch", hide_index=True)
else:
    st.info("보유 종목이 없습니다.")

# ── 하단 ───────────────────────────────────────────

st.markdown("")
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🔄 새로고침", type="primary"):
        st.rerun()

with col3:
    st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
