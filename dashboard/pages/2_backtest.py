"""
백테스트 페이지
전략 선택, 실행, 결과 시각화
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from config import kis_config, backtest_config
from api_client import KISClient, load_sample_data
from backtest_engine import BacktestEngine
from strategies import STRATEGIES
from dashboard.state import init_session_state, set_backtest_result, get_backtest_result
from dashboard.components.charts import create_equity_curve, create_backtest_result_chart
from dashboard.components.styles import inject_css, render_header, render_section_header, render_trading_mode_toggle

init_session_state()
inject_css()

render_header("🔬 백테스트", "과거 데이터로 전략을 시뮬레이션하고 성과를 분석하세요")
render_trading_mode_toggle()

# 사이드바: 설정
st.sidebar.markdown("### ⚙️ 백테스트 설정")

# 전략 선택
STRATEGY_LABELS = {
    "golden_cross": "📈 골든크로스",
    "rsi": "📊 RSI",
    "bollinger_band": "📉 볼린저밴드",
    "macd": "🔀 MACD",
    "composite": "🧩 복합전략",
    "ai_composite": "🧠 AI 복합전략 (LLM+ML)",
}

strategy_name = st.sidebar.selectbox(
    "전략 선택",
    options=list(STRATEGIES.keys()),
    format_func=lambda x: STRATEGY_LABELS.get(x, x),
)

# 종목 입력
stock_code = st.sidebar.text_input(
    "종목코드",
    value="005930",
    help="예: 005930 (삼성전자), 000660 (SK하이닉스)",
)

# 기간 설정
st.sidebar.markdown("**기간**")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "시작일",
        value=datetime.now() - timedelta(days=365),
    )
with col2:
    end_date = st.date_input(
        "종료일",
        value=datetime.now(),
    )

# 데이터 소스
api_valid = kis_config.app_key != "YOUR_APP_KEY"
use_demo = st.sidebar.checkbox(
    "데모 데이터 사용",
    value=not api_valid,
    help="체크 해제하면 실제 API 데이터를 사용합니다.",
)

# 백테스트 설정
st.sidebar.markdown("---")
st.sidebar.markdown("### 💵 비용 설정")

initial_capital = st.sidebar.number_input(
    "초기 자본금 (원)",
    value=int(backtest_config.initial_capital),
    min_value=1_000_000,
    step=1_000_000,
    format="%d",
)

commission_rate = st.sidebar.number_input(
    "수수료율 (%)",
    value=backtest_config.commission_rate * 100,
    min_value=0.0,
    max_value=1.0,
    step=0.001,
    format="%.3f",
)

st.sidebar.markdown("### 🛡️ 리스크 관리")

stop_loss_rate = st.sidebar.number_input(
    "손절선 (%)",
    value=backtest_config.stop_loss_rate * 100,
    min_value=0.0,
    max_value=20.0,
    step=0.5,
    format="%.1f",
    help="0이면 손절 비활성화",
)

take_profit_rate = st.sidebar.number_input(
    "익절선 (%)",
    value=backtest_config.take_profit_rate * 100,
    min_value=0.0,
    max_value=50.0,
    step=1.0,
    format="%.1f",
    help="0이면 익절 비활성화",
)

st.sidebar.markdown("---")

# 실행 버튼
run_backtest = st.sidebar.button(
    "🚀 백테스트 실행",
    type="primary",
    width="stretch",
)

# 메인 영역
if run_backtest:
    with st.spinner("백테스트 실행 중..."):
        try:
            # 데이터 로드
            if use_demo:
                candles = load_sample_data()
                df = pd.DataFrame([
                    {
                        "date": c.date,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                    }
                    for c in candles
                ])
                df.set_index("date", inplace=True)
                stock_code = "DEMO"
            else:
                client = KISClient()
                candles = client.get_daily_candles(
                    stock_code=stock_code,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                )
                df = pd.DataFrame([
                    {
                        "date": c.date,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                    }
                    for c in candles
                ])
                df.set_index("date", inplace=True)

            # 전략 생성
            strategy_class = STRATEGIES[strategy_name]
            strategy = strategy_class()

            # 백테스트 설정
            from config import BacktestConfig
            config = BacktestConfig(
                initial_capital=float(initial_capital),
                commission_rate=commission_rate / 100,
                stop_loss_rate=stop_loss_rate / 100,
                take_profit_rate=take_profit_rate / 100,
            )

            # 백테스트 실행
            engine = BacktestEngine(config)
            result = engine.run(strategy, df, stock_code)

            # 결과 저장
            result_dict = {
                "strategy_name": result.strategy_name,
                "stock_code": result.stock_code,
                "period": result.period,
                "initial_capital": result.initial_capital,
                "final_capital": result.final_capital,
                "total_return": result.total_return,
                "trade_count": result.trade_count,
                "win_rate": result.win_rate,
                "daily_equity": result.daily_equity,
                "trades": [
                    {
                        "date": t.date,
                        "signal": t.signal.value,
                        "price": t.price,
                        "quantity": t.quantity,
                    }
                    for t in result.trades
                ],
                "df": df.reset_index(),
            }

            set_backtest_result(f"{strategy_name}_{stock_code}", result_dict)
            st.success("백테스트 완료!")

        except Exception as e:
            st.error(f"백테스트 오류: {e}")
            st.stop()

# 결과 표시
result_key = f"{strategy_name}_{stock_code}"
result = get_backtest_result(result_key)

if result:
    render_section_header("📊", "백테스트 결과")

    # 요약 메트릭
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="초기 자본금",
            value=f"{result['initial_capital']:,.0f}원",
        )

    with col2:
        st.metric(
            label="최종 자본금",
            value=f"{result['final_capital']:,.0f}원",
        )

    with col3:
        profit_delta = result["total_return"]
        pnl = result['final_capital'] - result['initial_capital']
        st.metric(
            label="총 수익률",
            value=f"{profit_delta:+.2f}%",
            delta=f"{pnl:+,.0f}원",
            delta_color="normal" if pnl >= 0 else "inverse",
        )

    with col4:
        st.metric(
            label="거래 횟수 / 승률",
            value=f"{result['trade_count']}회",
            delta=f"승률 {result['win_rate']:.1f}%",
            delta_color="off",
        )

    st.markdown("")

    # 차트
    tab1, tab2 = st.tabs(["📈 자산 추이", "🎯 매매 시점"])

    with tab1:
        equity_chart = create_equity_curve(
            result["daily_equity"],
            result["initial_capital"],
            title=f"{result['strategy_name']} 자산 추이",
        )
        st.plotly_chart(equity_chart, width="stretch")

    with tab2:
        if "df" in result:
            df = result["df"]
            if "date" not in df.columns:
                df = df.reset_index()
            backtest_chart = create_backtest_result_chart(
                df,
                result["trades"],
                title=f"{result['strategy_name']} 매매 시점",
            )
            st.plotly_chart(backtest_chart, width="stretch")

    st.markdown("")

    # 거래 내역
    render_section_header("📝", "거래 내역")

    if result["trades"]:
        trades_df = pd.DataFrame(result["trades"])
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df["date"] = trades_df["date"].dt.strftime("%Y-%m-%d")
        trades_df = trades_df.rename(columns={
            "date": "날짜",
            "signal": "구분",
            "price": "체결가",
            "quantity": "수량",
        })
        trades_df["구분"] = trades_df["구분"].map({"BUY": "🔴 매수", "SELL": "🔵 매도"})
        trades_df["체결가"] = trades_df["체결가"].apply(lambda x: f"{x:,.0f}")

        st.dataframe(trades_df, width="stretch", hide_index=True)
    else:
        st.info("거래 내역이 없습니다.")

else:
    st.info("👈 왼쪽 사이드바에서 설정을 선택하고 **'백테스트 실행'** 버튼을 클릭하세요.")

    st.markdown("")

    # 전략 설명
    render_section_header("📖", "전략 설명")

    strategy_descriptions = {
        "golden_cross": {
            "title": "📈 골든크로스 전략",
            "buy": "단기 이동평균선(5일)이 장기 이동평균선(20일)을 **상향 돌파**할 때",
            "sell": "단기 이동평균선이 장기 이동평균선을 **하향 돌파**할 때",
            "desc": "전통적인 추세 추종 전략으로, 상승 추세의 시작과 끝을 포착합니다.",
        },
        "rsi": {
            "title": "📊 RSI 전략",
            "buy": "RSI가 과매도 구간(30 이하)에서 **상승**할 때",
            "sell": "RSI가 과매수 구간(70 이상)에서 **하락**할 때",
            "desc": "상대강도지수(RSI)를 사용한 역추세 전략으로, 반전을 노립니다.",
        },
        "bollinger_band": {
            "title": "📉 볼린저밴드 전략",
            "buy": "가격이 하단밴드 이탈 후 **복귀**할 때",
            "sell": "가격이 상단밴드 이탈 후 **복귀**할 때",
            "desc": "변동성 기반 전략으로, 횡보장에서 효과적입니다.",
        },
        "macd": {
            "title": "🔀 MACD 전략",
            "buy": "MACD선이 시그널선을 **상향 돌파**할 때",
            "sell": "MACD선이 시그널선을 **하향 돌파**할 때",
            "desc": "이동평균 수렴/발산 지표를 활용한 추세 전략입니다.",
        },
        "composite": {
            "title": "🧩 복합 전략 (MA+RSI+MACD)",
            "buy": "MACD 골든크로스 + 상승추세 + RSI 적정",
            "sell": "MACD 데드크로스 또는 RSI 과매수",
            "desc": "여러 지표를 조합하여 거짓 신호를 필터링합니다.",
        },
        "ai_composite": {
            "title": "🧠 AI 복합 전략 (LLM+ML+Tech)",
            "buy": "LLM 강세 전망(40%) + ML 상승 예측(40%) + 기술적 매수 신호(20%) 종합 점수 > 0.3",
            "sell": "종합 점수 < -0.3 또는 RSI 과매수 + 음의 종합점수",
            "desc": "경제 뉴스·거시지표를 LLM이 분석하고, XGBoost가 가격 방향을 예측하며, 기술적 지표로 최종 확인합니다. 고위험 시 매수 비율 자동 축소, 연속 3회 손실 시 쿨다운 등 리스크 관리가 내장되어 있습니다.",
        },
    }

    # 선택된 전략 강조 표시
    info = strategy_descriptions.get(strategy_name)
    if info:
        st.markdown(f"""
        ### {info['title']}  ← 현재 선택

        | 조건 | 설명 |
        |------|------|
        | **매수** | {info['buy']} |
        | **매도** | {info['sell']} |

        {info['desc']}
        """)

    # 나머지 전략 설명
    with st.expander("📚 전체 전략 설명 보기"):
        for key, desc in strategy_descriptions.items():
            if key == strategy_name:
                continue
            st.markdown(f"""
**{desc['title']}**

| 조건 | 설명 |
|------|------|
| **매수** | {desc['buy']} |
| **매도** | {desc['sell']} |

{desc['desc']}

---
            """)
