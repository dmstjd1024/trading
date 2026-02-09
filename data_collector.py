"""
데이터 수집 및 저장 모듈

일봉 데이터를 SQLite에 저장하고 pandas DataFrame으로 로드합니다.
API 호출을 최소화하기 위해 이미 저장된 데이터는 스킵합니다.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Optional, List

import pandas as pd

from config import data_config
from models import Candle


class DataStore:
    """SQLite 기반 주가 데이터 저장소"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or data_config.db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_candles (
                    stock_code TEXT NOT NULL,
                    date       TEXT NOT NULL,
                    open       REAL NOT NULL,
                    high       REAL NOT NULL,
                    low        REAL NOT NULL,
                    close      REAL NOT NULL,
                    volume     INTEGER NOT NULL,
                    PRIMARY KEY (stock_code, date)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_candles_code_date
                ON daily_candles (stock_code, date)
            """)

    def save_candles(self, stock_code: str, candles: List[Candle]) -> int:
        """
        봉 데이터 저장 (중복 시 업데이트)

        Returns:
            저장된 레코드 수
        """
        rows = [
            (
                stock_code,
                c.date.strftime("%Y-%m-%d"),
                c.open, c.high, c.low, c.close, c.volume,
            )
            for c in candles
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO daily_candles
                (stock_code, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, rows)

        print(f"[저장] {stock_code} {len(rows)}개 레코드 저장 완료")
        return len(rows)

    def load_dataframe(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        DataFrame으로 로드

        Args:
            stock_code: 종목코드
            start_date: 시작일 (YYYY-MM-DD), None이면 전체
            end_date: 종료일 (YYYY-MM-DD), None이면 전체

        Returns:
            columns: date, open, high, low, close, volume
        """
        query = "SELECT date, open, high, low, close, volume FROM daily_candles WHERE stock_code = ?"
        params: list = [stock_code]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        print(f"[로드] {stock_code} {len(df)}개 레코드 로드")
        return df

    def load_candles(self, stock_code: str) -> List[Candle]:
        """Candle 리스트로 로드"""
        df = self.load_dataframe(stock_code)
        return [
            Candle(
                date=idx.to_pydatetime(),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=int(row["volume"]),
            )
            for idx, row in df.iterrows()
        ]

    def get_latest_date(self, stock_code: str) -> Optional[str]:
        """해당 종목의 가장 최근 저장 날짜"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT MAX(date) FROM daily_candles WHERE stock_code = ?",
                (stock_code,),
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def list_stocks(self) -> List[str]:
        """저장된 종목코드 목록"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT DISTINCT stock_code FROM daily_candles ORDER BY stock_code"
            )
            return [row[0] for row in cursor.fetchall()]
