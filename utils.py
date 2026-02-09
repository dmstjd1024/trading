"""
유틸리티 함수

백테스팅 결과 시각화, 지표 계산 헬퍼 등
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd

from models import BacktestResult, Signal


def plot_backtest_result(result: BacktestResult, save_path: Optional[str] = None) -> None:
    """
    백테스팅 결과를 차트로 시각화

    - 상단: 주가 + 매수/매도 포인트
    - 하단: 자산 가치 변화
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[경고] matplotlib이 설치되지 않았습니다. pip install matplotlib")
        return

    # 한글 폰트 설정 시도
    try:
        plt.rcParams["font.family"] = "NanumGothic"
    except Exception:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
    fig.suptitle(f"Backtest: {result.strategy_name} / {result.stock_code}", fontsize=14)

    # 상단: 자산 가치 곡선
    dates = [d for d, _ in result.daily_equity]
    equity = [e for _, e in result.daily_equity]

    ax1.plot(dates, equity, "b-", linewidth=1.2, label="Portfolio Value")
    ax1.axhline(y=result.initial_capital, color="gray", linestyle="--", alpha=0.5, label="Initial Capital")

    # 매수/매도 포인트 표시
    buy_dates = [t.date for t in result.trades if t.signal == Signal.BUY]
    buy_equities = []
    for bd in buy_dates:
        closest = min(result.daily_equity, key=lambda x: abs((x[0] - bd).total_seconds()))
        buy_equities.append(closest[1])

    sell_dates = [t.date for t in result.trades if t.signal == Signal.SELL]
    sell_equities = []
    for sd in sell_dates:
        closest = min(result.daily_equity, key=lambda x: abs((x[0] - sd).total_seconds()))
        sell_equities.append(closest[1])

    ax1.scatter(buy_dates, buy_equities, marker="^", color="red", s=80, zorder=5, label="Buy")
    ax1.scatter(sell_dates, sell_equities, marker="v", color="blue", s=80, zorder=5, label="Sell")

    ax1.set_ylabel("Portfolio Value (KRW)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 하단: 일별 수익률
    equity_series = pd.Series(equity, index=dates)
    daily_returns = equity_series.pct_change().fillna(0) * 100

    colors = ["green" if r >= 0 else "red" for r in daily_returns]
    ax2.bar(dates, daily_returns, color=colors, alpha=0.6, width=1)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("Daily Return (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[차트] {save_path} 저장 완료")
    else:
        plt.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
        print("[차트] backtest_result.png 저장 완료")

    plt.close()


def calculate_max_drawdown(daily_equity: List[Tuple[datetime, float]]) -> float:
    """최대 낙폭(MDD) 계산 (%)"""
    if not daily_equity:
        return 0.0

    peak = daily_equity[0][1]
    max_dd = 0.0

    for _, equity in daily_equity:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return max_dd


def calculate_sharpe_ratio(
    daily_equity: List[Tuple[datetime, float]],
    risk_free_rate: float = 0.035,  # 한국 무위험 수익률 약 3.5%
) -> float:
    """샤프 비율 계산 (연간화)"""
    if len(daily_equity) < 2:
        return 0.0

    equity_values = [e for _, e in daily_equity]
    returns = pd.Series(equity_values).pct_change().dropna()

    if returns.std() == 0:
        return 0.0

    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf

    return (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)
