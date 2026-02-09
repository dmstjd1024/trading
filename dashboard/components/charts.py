"""
Plotly 차트 컴포넌트
"""

from __future__ import annotations

from typing import List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "가격 차트",
    show_volume: bool = True,
) -> go.Figure:
    """
    캔들스틱 차트 생성

    Args:
        df: OHLCV 데이터프레임 (date, open, high, low, close, volume 컬럼 필요)
        title: 차트 제목
        show_volume: 거래량 표시 여부
    """
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
        )
    else:
        fig = go.Figure()

    # 캔들스틱
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="가격",
            increasing_line_color="#ef5350",  # 상승 빨강 (한국 주식 스타일)
            decreasing_line_color="#26a69a",  # 하락 파랑
        ),
        row=1 if show_volume else None,
        col=1 if show_volume else None,
    )

    # 거래량
    if show_volume and "volume" in df.columns:
        colors = [
            "#ef5350" if row["close"] >= row["open"] else "#26a69a"
            for _, row in df.iterrows()
        ]
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="거래량",
                marker_color=colors,
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_white",
        showlegend=False,
    )

    if show_volume:
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="거래량", row=2, col=1)

    return fig


def create_equity_curve(
    daily_equity: List[Tuple[datetime, float]],
    initial_capital: float,
    title: str = "자산 추이",
) -> go.Figure:
    """
    자산 추이 차트 생성

    Args:
        daily_equity: (날짜, 자산) 튜플 리스트
        initial_capital: 초기 자본금
        title: 차트 제목
    """
    if not daily_equity:
        fig = go.Figure()
        fig.add_annotation(text="데이터 없음", x=0.5, y=0.5, showarrow=False)
        return fig

    dates = [d[0] for d in daily_equity]
    values = [d[1] for d in daily_equity]

    fig = go.Figure()

    # 자산 추이 라인
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name="자산",
            line=dict(color="#1976d2", width=2),
            fill="tozeroy",
            fillcolor="rgba(25, 118, 210, 0.1)",
        )
    )

    # 초기 자본금 기준선
    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"초기자본: {initial_capital:,.0f}원",
    )

    fig.update_layout(
        title=title,
        xaxis_title="날짜",
        yaxis_title="자산 (원)",
        height=400,
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def create_portfolio_pie_chart(
    holdings: List[dict],
    cash_balance: float,
) -> go.Figure:
    """
    포트폴리오 구성 파이 차트

    Args:
        holdings: 보유 종목 리스트 (stock_name, current_price * quantity)
        cash_balance: 현금 잔고
    """
    labels = ["현금"]
    values = [cash_balance]
    colors = ["#90caf9"]  # 현금은 연한 파랑

    stock_colors = ["#ef5350", "#ff7043", "#ffca28", "#66bb6a", "#26a69a", "#42a5f5", "#ab47bc"]

    for i, holding in enumerate(holdings):
        labels.append(holding.get("stock_name", holding.get("stock_code", "Unknown")))
        values.append(holding.get("current_price", 0) * holding.get("quantity", 0))
        colors.append(stock_colors[i % len(stock_colors)])

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=colors,
                textinfo="label+percent",
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="포트폴리오 구성",
        height=400,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )

    return fig


def create_profit_loss_bar_chart(
    holdings: List[dict],
) -> go.Figure:
    """
    종목별 손익 바 차트

    Args:
        holdings: 보유 종목 리스트 (stock_name, profit_loss, profit_loss_rate)
    """
    if not holdings:
        fig = go.Figure()
        fig.add_annotation(text="보유 종목 없음", x=0.5, y=0.5, showarrow=False)
        return fig

    names = [h.get("stock_name", h.get("stock_code", "")) for h in holdings]
    profit_loss = [h.get("profit_loss", 0) for h in holdings]
    rates = [h.get("profit_loss_rate", 0) for h in holdings]

    colors = ["#ef5350" if pl >= 0 else "#26a69a" for pl in profit_loss]

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=profit_loss,
                marker_color=colors,
                text=[f"{r:+.2f}%" for r in rates],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title="종목별 평가손익",
        xaxis_title="종목",
        yaxis_title="손익 (원)",
        height=400,
        template="plotly_white",
    )

    fig.add_hline(y=0, line_color="gray", line_width=1)

    return fig


def create_backtest_result_chart(
    df: pd.DataFrame,
    trades: List[dict],
    title: str = "백테스트 결과",
) -> go.Figure:
    """
    백테스트 결과 차트 (가격 + 매수/매도 시점)

    Args:
        df: OHLCV 데이터프레임
        trades: 거래 기록 리스트 (date, signal, price)
        title: 차트 제목
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    # 가격 라인
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            mode="lines",
            name="종가",
            line=dict(color="#1976d2", width=1.5),
        ),
        row=1, col=1,
    )

    # 매수/매도 마커
    buy_trades = [t for t in trades if t.get("signal") == "BUY"]
    sell_trades = [t for t in trades if t.get("signal") == "SELL"]

    if buy_trades:
        fig.add_trace(
            go.Scatter(
                x=[t["date"] for t in buy_trades],
                y=[t["price"] for t in buy_trades],
                mode="markers",
                name="매수",
                marker=dict(symbol="triangle-up", size=12, color="#ef5350"),
            ),
            row=1, col=1,
        )

    if sell_trades:
        fig.add_trace(
            go.Scatter(
                x=[t["date"] for t in sell_trades],
                y=[t["price"] for t in sell_trades],
                mode="markers",
                name="매도",
                marker=dict(symbol="triangle-down", size=12, color="#26a69a"),
            ),
            row=1, col=1,
        )

    # 거래량
    if "volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="거래량",
                marker_color="rgba(100, 100, 100, 0.3)",
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title=title,
        height=600,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)

    return fig
