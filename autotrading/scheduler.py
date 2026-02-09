"""
스케줄러
APScheduler를 사용하여 자동매매를 예약 실행합니다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional, List

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import autotrading_config
from .executor import StrategyExecutor
from .log_manager import LogManager


class TradingScheduler:
    """자동매매 스케줄러"""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.executor = StrategyExecutor()
        self.log_manager = LogManager()
        self._is_running = False
        self._job_id = "autotrading_job"

    @property
    def is_running(self) -> bool:
        """스케줄러 실행 상태"""
        return self._is_running

    def start(self) -> None:
        """스케줄러 시작"""
        if self._is_running:
            print("[스케줄러] 이미 실행 중입니다.")
            return

        self.scheduler.start()
        self._is_running = True
        print(f"[스케줄러] 시작됨 - {datetime.now()}")

    def stop(self) -> None:
        """스케줄러 중지"""
        if not self._is_running:
            print("[스케줄러] 실행 중이 아닙니다.")
            return

        self.scheduler.shutdown(wait=False)
        self._is_running = False
        print(f"[스케줄러] 중지됨 - {datetime.now()}")

    def add_trading_job(
        self,
        strategy_name: str,
        stock_codes: List[str],
        schedule_time: Optional[str] = None,
        callback: Optional[Callable[[List[dict]], None]] = None,
    ) -> None:
        """
        자동매매 작업 추가

        Args:
            strategy_name: 전략 이름
            stock_codes: 대상 종목 코드 리스트
            schedule_time: 실행 시간 (HH:MM 형식)
            callback: 실행 완료 후 콜백 함수
        """
        schedule_time = schedule_time or autotrading_config.schedule_time

        # 기존 작업 제거
        if self.scheduler.get_job(self._job_id):
            self.scheduler.remove_job(self._job_id)

        # 시간 파싱
        hour, minute = map(int, schedule_time.split(":"))

        def job_func():
            print(f"\n[자동매매] 실행 시작 - {datetime.now()}")
            print(f"  전략: {strategy_name}")
            print(f"  종목: {stock_codes}")

            results = self.executor.execute_strategy(strategy_name, stock_codes)

            for result in results:
                status_emoji = "O" if result["status"] == "executed" else "X"
                print(f"  [{status_emoji}] {result['stock_code']}: {result['message']}")

            if callback:
                callback(results)

            print(f"[자동매매] 실행 완료 - {len(results)}개 종목 처리\n")

        # 작업 추가 (월~금 장 시작 후)
        self.scheduler.add_job(
            job_func,
            CronTrigger(
                hour=hour,
                minute=minute,
                day_of_week="mon-fri",  # 월~금
            ),
            id=self._job_id,
            replace_existing=True,
        )

        print(f"[스케줄러] 자동매매 작업 등록 - 매일 {schedule_time} (월~금)")

    def remove_trading_job(self) -> None:
        """자동매매 작업 제거"""
        if self.scheduler.get_job(self._job_id):
            self.scheduler.remove_job(self._job_id)
            print("[스케줄러] 자동매매 작업 제거됨")

    def get_next_run_time(self) -> Optional[datetime]:
        """다음 실행 시간 조회"""
        job = self.scheduler.get_job(self._job_id)
        if job:
            return job.next_run_time
        return None

    def run_now(
        self,
        strategy_name: Optional[str] = None,
        stock_codes: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        즉시 실행

        Args:
            strategy_name: 전략 이름 (없으면 설정값 사용)
            stock_codes: 종목 코드 (없으면 설정값 사용)

        Returns:
            실행 결과 리스트
        """
        strategy_name = strategy_name or autotrading_config.strategy_name
        stock_codes = stock_codes or autotrading_config.stock_codes

        print(f"\n[자동매매] 수동 실행 - {datetime.now()}")
        results = self.executor.execute_strategy(strategy_name, stock_codes)
        print(f"[자동매매] 완료 - {len(results)}개 종목 처리\n")

        return results


# 싱글톤 인스턴스
_scheduler_instance: Optional[TradingScheduler] = None


def get_scheduler() -> TradingScheduler:
    """스케줄러 싱글톤 인스턴스 반환"""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TradingScheduler()
    return _scheduler_instance
