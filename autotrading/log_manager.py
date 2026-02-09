"""
로그 관리자
자동매매 실행 로그를 파일로 저장하고 조회합니다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from config import autotrading_config
from models import TradingLog


class LogManager:
    """자동매매 로그 관리자"""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir or autotrading_config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_file(self, date: Optional[datetime] = None) -> Path:
        """날짜별 로그 파일 경로"""
        date = date or datetime.now()
        filename = f"trading_{date.strftime('%Y%m%d')}.jsonl"
        return self.log_dir / filename

    def write_log(self, log: TradingLog) -> None:
        """로그 기록"""
        log_file = self._get_log_file(log.timestamp)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log.to_dict(), ensure_ascii=False) + "\n")

    def write_log_dict(self, log_dict: dict) -> None:
        """딕셔너리 형태 로그 기록"""
        log_file = self._get_log_file()

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")

    def read_logs(
        self,
        date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[dict]:
        """로그 읽기"""
        log_file = self._get_log_file(date)

        if not log_file.exists():
            return []

        logs = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # 최신순 정렬
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return logs[:limit]

    def read_recent_logs(self, days: int = 7, limit: int = 100) -> List[dict]:
        """최근 N일간 로그 읽기"""
        from datetime import timedelta

        all_logs = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            logs = self.read_logs(date, limit=limit)
            all_logs.extend(logs)

        # 최신순 정렬 후 제한
        all_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return all_logs[:limit]

    def get_log_summary(self, date: Optional[datetime] = None) -> dict:
        """로그 요약"""
        logs = self.read_logs(date, limit=1000)

        if not logs:
            return {
                "total": 0,
                "executed": 0,
                "failed": 0,
                "skipped": 0,
            }

        executed = sum(1 for log in logs if log.get("status") == "executed")
        failed = sum(1 for log in logs if log.get("status") == "failed")
        skipped = sum(1 for log in logs if log.get("status") == "skipped")

        return {
            "total": len(logs),
            "executed": executed,
            "failed": failed,
            "skipped": skipped,
        }

    def cleanup_old_logs(self, keep_days: int = 30) -> int:
        """오래된 로그 삭제"""
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=keep_days)
        deleted = 0

        for log_file in self.log_dir.glob("trading_*.jsonl"):
            try:
                # 파일명에서 날짜 추출
                date_str = log_file.stem.replace("trading_", "")
                file_date = datetime.strptime(date_str, "%Y%m%d")

                if file_date < cutoff:
                    log_file.unlink()
                    deleted += 1

            except (ValueError, OSError):
                continue

        return deleted
