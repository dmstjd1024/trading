"""
메인 진입점

사용법:
  # 샘플 데이터로 백테스팅 (API 키 불필요)
  python main.py demo

  # API로 데이터 수집
  python main.py collect --code 005930 --start 20240101 --end 20241231

  # 백테스팅 실행
  python main.py backtest --strategy golden_cross --code 005930

  # 전략 비교
  python main.py compare --code 005930

  # 멀티팩터 종목 스크리닝
  python main.py screen
  python main.py screen --top 20 --market kospi
"""

import argparse
import sys
from datetime import datetime

from config import kis_config, backtest_config
from api_client import KISClient, load_sample_data
from data_collector import DataStore
from backtest_engine import BacktestEngine
from strategies import STRATEGIES, GoldenCrossStrategy, RSIStrategy
from models import Candle
from utils import plot_backtest_result, calculate_max_drawdown, calculate_sharpe_ratio

import pandas as pd


def cmd_demo(args):
    """샘플 데이터로 데모 백테스팅 실행"""
    print("=" * 60)
    print("  한국 주식 트레이딩 시스템 - 데모 모드")
    print("  (샘플 데이터 사용, API 키 불필요)")
    print("=" * 60)

    # 샘플 데이터 로드
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

    engine = BacktestEngine()

    # 골든크로스 전략
    strategy1 = GoldenCrossStrategy(short_window=5, long_window=20)
    result1 = engine.run(strategy1, df, stock_code="SAMPLE")
    print(result1.summary())
    mdd1 = calculate_max_drawdown(result1.daily_equity)
    sharpe1 = calculate_sharpe_ratio(result1.daily_equity)
    print(f"  MDD:      {mdd1:>13.2f}%")
    print(f"  Sharpe:   {sharpe1:>13.2f}")

    # 차트 저장
    plot_backtest_result(result1, "demo_golden_cross.png")

    print()

    # RSI 전략
    strategy2 = RSIStrategy(period=14, oversold=30, overbought=70)
    result2 = engine.run(strategy2, df, stock_code="SAMPLE")
    print(result2.summary())
    mdd2 = calculate_max_drawdown(result2.daily_equity)
    sharpe2 = calculate_sharpe_ratio(result2.daily_equity)
    print(f"  MDD:      {mdd2:>13.2f}%")
    print(f"  Sharpe:   {sharpe2:>13.2f}")

    plot_backtest_result(result2, "demo_rsi.png")


def cmd_collect(args):
    """API를 통해 데이터 수집"""
    if not kis_config.validate():
        sys.exit(1)

    client = KISClient()
    store = DataStore()

    print(f"\n[수집] {args.code} 데이터 수집 시작")
    print(f"  기간: {args.start} ~ {args.end}")

    candles = client.get_daily_candles(
        stock_code=args.code,
        start_date=args.start,
        end_date=args.end,
    )

    if candles:
        store.save_candles(args.code, candles)
        print(f"\n[완료] {len(candles)}개 봉 데이터 수집 및 저장 완료")
    else:
        print("[경고] 수집된 데이터가 없습니다.")


def cmd_backtest(args):
    """저장된 데이터로 백테스팅"""
    store = DataStore()
    df = store.load_dataframe(args.code)

    if df.empty:
        print(f"[오류] {args.code} 데이터가 없습니다. 먼저 collect 명령으로 데이터를 수집해주세요.")
        sys.exit(1)

    strategy_cls = STRATEGIES.get(args.strategy)
    if not strategy_cls:
        print(f"[오류] 알 수 없는 전략: {args.strategy}")
        print(f"  사용 가능한 전략: {', '.join(STRATEGIES.keys())}")
        sys.exit(1)

    strategy = strategy_cls()
    engine = BacktestEngine()
    result = engine.run(strategy, df, stock_code=args.code)

    print(result.summary())
    mdd = calculate_max_drawdown(result.daily_equity)
    sharpe = calculate_sharpe_ratio(result.daily_equity)
    print(f"  MDD:      {mdd:>13.2f}%")
    print(f"  Sharpe:   {sharpe:>13.2f}")

    chart_path = f"backtest_{args.code}_{args.strategy}.png"
    plot_backtest_result(result, chart_path)


def cmd_screen(args):
    """멀티팩터 종목 스크리닝"""
    if not kis_config.validate():
        sys.exit(1)

    from screener import StockScreener

    # 시장 코드 변환
    market_map = {
        "all": "0000",
        "kospi": "0001",
        "kosdaq": "1001",
        "kospi200": "2001",
    }
    market = market_map.get(args.market, args.market)

    screener = StockScreener()
    results = screener.run(
        top_n=args.top,
        market=market,
        tech_weight=args.tech_weight,
        fund_weight=args.fund_weight,
    )

    if not results:
        print("\n[결과] 스크리닝 결과가 없습니다.")
        sys.exit(0)

    # 종목 코드 목록 출력
    codes = [r["code"] for r in results]
    print(f"\n[종목 코드] {', '.join(codes)}")
    print(f"  → 자동매매 연동: autotrading_config.stock_codes에 설정 가능")


def cmd_compare(args):
    """모든 전략 비교"""
    store = DataStore()
    df = store.load_dataframe(args.code)

    if df.empty:
        print(f"[오류] {args.code} 데이터가 없습니다.")
        sys.exit(1)

    engine = BacktestEngine()

    print(f"\n{'='*70}")
    print(f"  전략 비교 - {args.code}")
    print(f"{'='*70}")

    results = []
    for name, strategy_cls in STRATEGIES.items():
        strategy = strategy_cls()
        result = engine.run(strategy, df.copy(), stock_code=args.code)
        mdd = calculate_max_drawdown(result.daily_equity)
        sharpe = calculate_sharpe_ratio(result.daily_equity)
        results.append((name, result, mdd, sharpe))

    print(f"\n{'전략':<25} {'수익률':>10} {'거래수':>8} {'승률':>8} {'MDD':>8} {'Sharpe':>8}")
    print("-" * 70)
    for name, result, mdd, sharpe in results:
        print(
            f"{result.strategy_name:<25} "
            f"{result.total_return:>9.2f}% "
            f"{result.trade_count:>7}회 "
            f"{result.win_rate:>7.1f}% "
            f"{mdd:>7.2f}% "
            f"{sharpe:>8.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="한국 주식 트레이딩 시스템")
    subparsers = parser.add_subparsers(dest="command", help="명령어")

    # demo
    sub_demo = subparsers.add_parser("demo", help="샘플 데이터로 데모 실행")

    # collect
    sub_collect = subparsers.add_parser("collect", help="데이터 수집")
    sub_collect.add_argument("--code", required=True, help="종목코드 (예: 005930)")
    sub_collect.add_argument("--start", default="20240101", help="시작일 (YYYYMMDD)")
    sub_collect.add_argument(
        "--end",
        default=datetime.now().strftime("%Y%m%d"),
        help="종료일 (YYYYMMDD)",
    )

    # backtest
    sub_bt = subparsers.add_parser("backtest", help="백테스팅 실행")
    sub_bt.add_argument("--code", required=True, help="종목코드")
    sub_bt.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGIES.keys()),
        help="전략 이름",
    )

    # compare
    sub_cmp = subparsers.add_parser("compare", help="전략 비교")
    sub_cmp.add_argument("--code", required=True, help="종목코드")

    # screen
    sub_screen = subparsers.add_parser("screen", help="멀티팩터 종목 스크리닝")
    sub_screen.add_argument("--top", type=int, default=10, help="상위 N개 종목 (기본: 10)")
    sub_screen.add_argument(
        "--market",
        default="all",
        choices=["all", "kospi", "kosdaq", "kospi200"],
        help="대상 시장 (기본: all)",
    )
    sub_screen.add_argument("--tech-weight", type=float, default=0.5, help="기술적 팩터 가중치 (기본: 0.5)")
    sub_screen.add_argument("--fund-weight", type=float, default=0.5, help="펀더멘탈 팩터 가중치 (기본: 0.5)")

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "screen":
        cmd_screen(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
