"""
자동매매 페이지
ON/OFF 토글, 전략 설정, 실행 로그 표시
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from datetime import datetime

from config import kis_config, autotrading_config
from strategies import STRATEGIES
from dashboard.state import (
    init_session_state,
    set_autotrading_enabled,
    is_autotrading_enabled,
    get_trading_logs,
    add_trading_log,
)
from dashboard.components.styles import inject_css, render_header, render_section_header, render_badge, render_trading_mode_toggle

init_session_state()
inject_css()

render_header("🤖 자동매매", "전략을 설정하고 자동으로 매매를 실행하세요")
render_trading_mode_toggle()

# API 키 확인
api_valid = kis_config.validate()

if not api_valid:
    st.warning("""
    **API 키가 설정되지 않았습니다.**
    자동매매를 사용하려면 환경변수를 설정하세요:
    `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_ACCOUNT_NO`
    """)

# 자동매매 상태
render_section_header("⚡", "자동매매 상태")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    current_status = is_autotrading_enabled()
    new_status = st.toggle(
        "자동매매 활성화",
        value=current_status,
        disabled=not api_valid,
    )

    if new_status != current_status:
        set_autotrading_enabled(new_status)
        if new_status:
            st.success("자동매매가 활성화되었습니다.")
            add_trading_log({
                "timestamp": datetime.now().isoformat(),
                "stock_code": "-",
                "strategy_name": "-",
                "signal": "-",
                "price": 0,
                "quantity": 0,
                "status": "system",
                "message": "자동매매 활성화됨",
            })
        else:
            st.info("자동매매가 비활성화되었습니다.")
            add_trading_log({
                "timestamp": datetime.now().isoformat(),
                "stock_code": "-",
                "strategy_name": "-",
                "signal": "-",
                "price": 0,
                "quantity": 0,
                "status": "system",
                "message": "자동매매 비활성화됨",
            })

with col2:
    if is_autotrading_enabled():
        badge = render_badge("실행 중", "success")
    else:
        badge = render_badge("중지됨", "warning")
    st.markdown(f"<div style='padding-top: 0.5rem;'>{badge}</div>", unsafe_allow_html=True)

with col3:
    st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("")

# 자동매매 설정
render_section_header("⚙️", "자동매매 설정")

STRATEGY_LABELS = {
    "golden_cross": "📈 골든크로스",
    "rsi": "📊 RSI",
    "bollinger_band": "📉 볼린저밴드",
    "macd": "🔀 MACD",
    "composite": "🧩 복합전략",
}

col1, col2 = st.columns(2)

with col1:
    selected_strategy = st.selectbox(
        "전략",
        options=list(STRATEGIES.keys()),
        index=list(STRATEGIES.keys()).index(autotrading_config.strategy_name)
            if autotrading_config.strategy_name in STRATEGIES else 0,
        format_func=lambda x: STRATEGY_LABELS.get(x, x),
    )

    schedule_time = st.time_input(
        "매일 실행 시간",
        value=datetime.strptime(autotrading_config.schedule_time, "%H:%M").time(),
        help="매일 이 시간에 전략을 실행합니다.",
    )

with col2:
    stock_codes_input = st.text_area(
        "대상 종목 (한 줄에 하나씩)",
        value="\n".join(autotrading_config.stock_codes),
        height=100,
        help="자동매매를 실행할 종목코드를 입력하세요.",
    )
    stock_codes = [code.strip() for code in stock_codes_input.split("\n") if code.strip()]

    max_position_ratio = st.slider(
        "종목당 최대 투자 비율 (%)",
        min_value=10,
        max_value=100,
        value=int(autotrading_config.max_position_ratio * 100),
        step=10,
    )

if st.button("💾 설정 저장", disabled=not api_valid):
    st.session_state.autotrading_strategy = selected_strategy
    st.session_state.autotrading_stocks = stock_codes
    st.session_state.autotrading_schedule = schedule_time.strftime("%H:%M")
    st.session_state.autotrading_max_ratio = max_position_ratio / 100

    st.success("설정이 저장되었습니다.")

    add_trading_log({
        "timestamp": datetime.now().isoformat(),
        "stock_code": "-",
        "strategy_name": selected_strategy,
        "signal": "-",
        "price": 0,
        "quantity": 0,
        "status": "system",
        "message": f"설정 변경: {STRATEGY_LABELS.get(selected_strategy, selected_strategy)}, 종목 {len(stock_codes)}개",
    })

st.markdown("")

# 수동 실행
render_section_header("▶️", "수동 실행")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("🚀 지금 실행", type="primary", disabled=not api_valid):
        with st.spinner("전략 실행 중..."):
            try:
                from autotrading.executor import StrategyExecutor

                executor = StrategyExecutor()
                results = executor.execute_strategy(
                    strategy_name=selected_strategy,
                    stock_codes=stock_codes,
                )

                for result in results:
                    add_trading_log(result)

                st.success(f"전략 실행 완료: {len(results)}개 종목 처리")

            except ImportError:
                demo_log = {
                    "timestamp": datetime.now().isoformat(),
                    "stock_code": stock_codes[0] if stock_codes else "005930",
                    "strategy_name": selected_strategy,
                    "signal": "HOLD",
                    "price": 75000,
                    "quantity": 0,
                    "status": "executed",
                    "message": "시그널 없음 (데모)",
                }
                add_trading_log(demo_log)
                st.info("전략 실행 완료 (데모 모드)")

            except Exception as e:
                st.error(f"실행 오류: {e}")

with col2:
    if st.button("🔄 새로고침"):
        st.rerun()

st.markdown("")

# 실행 로그
render_section_header("📜", "실행 로그")

logs = get_trading_logs()

if logs:
    log_df = pd.DataFrame(logs)

    log_df = log_df.rename(columns={
        "timestamp": "시간",
        "stock_code": "종목코드",
        "strategy_name": "전략",
        "signal": "시그널",
        "price": "가격",
        "quantity": "수량",
        "status": "상태",
        "message": "메시지",
    })

    log_df["시간"] = pd.to_datetime(log_df["시간"])
    log_df["시간"] = log_df["시간"].dt.strftime("%Y-%m-%d %H:%M:%S")
    log_df["가격"] = log_df["가격"].apply(lambda x: f"{x:,.0f}" if x > 0 else "-")

    st.dataframe(
        log_df,
        width="stretch",
        hide_index=True,
        height=400,
    )

    if st.button("🗑️ 로그 초기화"):
        st.session_state.trading_logs = []
        st.rerun()

else:
    st.info("실행 로그가 없습니다. '지금 실행' 버튼을 눌러보세요.")

# 하단 주의사항
st.markdown("")
st.markdown("""
<div class="info-card" style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);">
    <strong>⚠️ 주의사항</strong><br>
    <span style="font-size: 0.9rem;">
    자동매매는 반드시 모의투자 환경에서 충분히 테스트한 후 사용하세요.<br>
    실전투자 시 손실이 발생할 수 있으며, 스케줄러가 실행 중이어야 자동매매가 동작합니다.
    </span>
</div>
""", unsafe_allow_html=True)
