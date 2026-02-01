"""
한국투자증권 REST API 클라이언트

모의투자/실전투자 환경에서 인증, 시세 조회, 일봉 데이터 조회를 처리합니다.
공식 문서: https://apiportal.koreainvestment.com/apiservice
"""

import time
import requests
from datetime import datetime, timedelta

from config import KISConfig, kis_config
from models import Candle


class KISClient:
    """한국투자증권 API 클라이언트"""

    def __init__(self, config: KISConfig | None = None):
        self.config = config or kis_config
        self._access_token: str | None = None
        self._token_expires: datetime | None = None

    # ── 인증 ──────────────────────────────────────────

    def _get_access_token(self) -> str:
        """OAuth 토큰 발급 (캐싱 포함)"""
        now = datetime.now()

        # 토큰이 유효하면 재사용
        if self._access_token and self._token_expires and now < self._token_expires:
            return self._access_token

        url = f"{self.config.base_url}/oauth2/tokenP"
        body = {
            "grant_type": "client_credentials",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
        }

        resp = requests.post(url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        self._access_token = data["access_token"]
        # 토큰 만료 1시간 전에 갱신하도록 설정
        expires_in = int(data.get("expires_in", 86400))
        self._token_expires = now + timedelta(seconds=expires_in - 3600)

        print(f"[인증] 토큰 발급 완료 (만료: {self._token_expires})")
        return self._access_token

    def _headers(self, tr_id: str) -> dict:
        """API 요청 공통 헤더"""
        token = self._get_access_token()
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
            "tr_id": tr_id,
        }

    # ── 시세 조회 ────────────────────────────────────

    def get_current_price(self, stock_code: str) -> dict:
        """현재가 조회"""
        if self.config.is_paper:
            tr_id = "FHKST01010100"
        else:
            tr_id = "FHKST01010100"

        url = f"{self.config.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {
            "fid_cond_mrkt_div_code": "J",  # 주식
            "fid_input_iscd": stock_code,
        }

        resp = requests.get(
            url, headers=self._headers(tr_id), params=params, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"API 오류: {data.get('msg1', 'Unknown error')}")

        output = data["output"]
        return {
            "code": stock_code,
            "name": output.get("hts_kor_isnm", ""),
            "price": int(output.get("stck_prpr", 0)),
            "change": int(output.get("prdy_vrss", 0)),
            "change_rate": float(output.get("prdy_ctrt", 0)),
            "volume": int(output.get("acml_vol", 0)),
            "high": int(output.get("stck_hgpr", 0)),
            "low": int(output.get("stck_lwpr", 0)),
        }

    def get_daily_candles(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        adjust: bool = True,
    ) -> list[Candle]:
        """
        일봉 데이터 조회

        Args:
            stock_code: 종목코드 (예: "005930")
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            adjust: 수정주가 적용 여부

        Returns:
            Candle 리스트 (오래된 날짜순)
        """
        tr_id = "FHKST03010100"
        url = f"{self.config.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"

        all_candles: list[Candle] = []
        current_end = end_date

        while True:
            params = {
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": stock_code,
                "fid_input_date_1": start_date,
                "fid_input_date_2": current_end,
                "fid_period_div_code": "D",  # 일봉
                "fid_org_adj_prc": "0" if adjust else "1",
            }

            resp = requests.get(
                url, headers=self._headers(tr_id), params=params, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("rt_cd") != "0":
                raise RuntimeError(f"API 오류: {data.get('msg1', 'Unknown error')}")

            items = data.get("output2", [])
            if not items:
                break

            for item in items:
                date_str = item.get("stck_bsop_date", "")
                if not date_str:
                    continue

                candle = Candle(
                    date=datetime.strptime(date_str, "%Y%m%d"),
                    open=float(item.get("stck_oprc", 0)),
                    high=float(item.get("stck_hgpr", 0)),
                    low=float(item.get("stck_lwpr", 0)),
                    close=float(item.get("stck_clpr", 0)),
                    volume=int(item.get("acml_vol", 0)),
                )
                all_candles.append(candle)

            # 페이지네이션: API는 최신→과거 순으로 반환
            # 마지막 항목의 날짜 - 1일을 새 end_date로 설정
            last_date_str = items[-1].get("stck_bsop_date", "")
            if not last_date_str or last_date_str <= start_date:
                break

            last_date = datetime.strptime(last_date_str, "%Y%m%d")
            current_end = (last_date - timedelta(days=1)).strftime("%Y%m%d")

            # API 호출 제한 방지
            time.sleep(0.5)

        # 오래된 날짜순으로 정렬
        all_candles.sort(key=lambda c: c.date)

        # 중복 제거
        seen = set()
        unique_candles = []
        for c in all_candles:
            key = c.date.strftime("%Y%m%d")
            if key not in seen:
                seen.add(key)
                unique_candles.append(c)

        print(f"[데이터] {stock_code} 일봉 {len(unique_candles)}개 조회 완료")
        return unique_candles


# ── API 없이 사용할 수 있는 샘플 데이터 ──────────────

def load_sample_data() -> list[Candle]:
    """
    API 키 없이 테스트할 수 있는 샘플 데이터 (삼성전자 2024년 가상 데이터)
    실제로는 API나 CSV에서 데이터를 로드하세요.
    """
    import random

    random.seed(42)

    candles = []
    price = 72000.0
    date = datetime(2024, 1, 2)

    for _ in range(250):  # 약 1년치 거래일
        if date.weekday() >= 5:  # 주말 건너뛰기
            date += timedelta(days=1)
            continue

        change = random.gauss(0, 0.02)  # 일일 변동률
        open_price = price
        close_price = price * (1 + change)
        high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, 0.005)))
        volume = int(random.gauss(15_000_000, 5_000_000))

        candles.append(Candle(
            date=date,
            open=round(open_price, -2),   # 100원 단위
            high=round(high_price, -2),
            low=round(low_price, -2),
            close=round(close_price, -2),
            volume=max(volume, 1_000_000),
        ))

        price = close_price
        date += timedelta(days=1)

    return candles
