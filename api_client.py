"""
한국투자증권 REST API 클라이언트

python-kis 라이브러리를 사용하거나, 직접 REST API를 호출합니다.
Python 3.10+ 환경에서는 python-kis를 사용하고, 그 외에는 직접 호출합니다.

공식 문서: https://apiportal.koreainvestment.com/apiservice
python-kis: https://github.com/Soju06/python-kis
"""

from __future__ import annotations

import json
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, List

from models import Candle, AccountBalance, Holding, OrderType

# python-kis 임포트 시도
PYKIS_AVAILABLE = False
PyKis = None

try:
    from pykis import PyKis
    PYKIS_AVAILABLE = True
except (ImportError, TypeError, SyntaxError):
    # Python 3.9 이하에서는 union type 문법 오류 발생
    PYKIS_AVAILABLE = False

# 순환 참조 방지를 위해 함수 내에서 import
def _get_config():
    from config import kis_config
    return kis_config


class KISClient:
    """한국투자증권 API 클라이언트"""

    def __init__(self, config=None):
        self.config = config or _get_config()
        self._pykis = None
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._init_client()

    def _init_client(self) -> None:
        """클라이언트 초기화"""
        if PYKIS_AVAILABLE:
            try:
                self._pykis = self.config.get_pykis()
                if self._pykis:
                    print("[인증] python-kis 클라이언트 초기화 완료")
                    return
            except Exception as e:
                print(f"[경고] python-kis 초기화 실패: {e}")

        # python-kis 사용 불가 시 직접 API 모드
        if self.config.validate():
            print("[인증] 직접 REST API 모드로 초기화")
        else:
            print("[경고] API 키가 설정되지 않았습니다")

    @property
    def is_available(self) -> bool:
        """API 사용 가능 여부"""
        if self._pykis:
            return True
        return self.config.app_key != "YOUR_APP_KEY"

    # ── 인증 (직접 API 모드) ──────────────────────────

    _TOKEN_CACHE_FILE = ".token_cache.json"

    def _load_cached_token(self) -> bool:
        """파일에 캐시된 토큰 로드 (프로세스 간 토큰 재사용)"""
        try:
            from pathlib import Path
            cache_path = Path(__file__).parent / self._TOKEN_CACHE_FILE
            if not cache_path.exists():
                return False

            with open(cache_path) as f:
                cache = json.load(f)

            # 같은 API 키의 토큰인지 확인
            if cache.get("app_key") != self.config.app_key:
                return False

            expires = datetime.fromisoformat(cache["expires"])
            if datetime.now() >= expires:
                return False

            self._access_token = cache["token"]
            self._token_expires = expires
            print(f"[인증] 캐시된 토큰 사용 (만료: {self._token_expires})")
            return True
        except (json.JSONDecodeError, KeyError, ValueError):
            return False

    def _save_token_cache(self) -> None:
        """토큰을 파일에 캐시"""
        try:
            from pathlib import Path
            cache_path = Path(__file__).parent / self._TOKEN_CACHE_FILE
            cache = {
                "app_key": self.config.app_key,
                "token": self._access_token,
                "expires": self._token_expires.isoformat(),
            }
            with open(cache_path, "w") as f:
                json.dump(cache, f)
        except Exception:
            pass  # 캐시 저장 실패는 무시

    def _get_access_token(self) -> str:
        """OAuth 토큰 발급 (캐시 우선)"""
        now = datetime.now()

        # 메모리 캐시된 토큰 확인
        if self._access_token and self._token_expires and now < self._token_expires:
            return self._access_token

        # 파일 캐시된 토큰 확인
        if self._load_cached_token():
            return self._access_token

        # 새 토큰 발급 (1분당 1회 제한 주의)
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
        expires_in = int(data.get("expires_in", 86400))
        self._token_expires = now + timedelta(seconds=expires_in)

        # 파일에 캐시 저장
        self._save_token_cache()

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
        if not self.is_available:
            raise RuntimeError("API 클라이언트가 초기화되지 않았습니다")

        # python-kis 사용
        if self._pykis:
            stock = self._pykis.stock(stock_code)
            quote = stock.quote()
            return {
                "code": stock_code,
                "name": getattr(quote, "name", ""),
                "price": int(quote.close),
                "change": int(quote.change),
                "change_rate": float(quote.change_rate),
                "volume": int(quote.volume),
                "high": int(quote.high),
                "low": int(quote.low),
            }

        # 직접 API 호출
        tr_id = "FHKST01010100"
        url = f"{self.config.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": stock_code,
        }

        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
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
    ) -> List[Candle]:
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
        if not self.is_available:
            raise RuntimeError("API 클라이언트가 초기화되지 않았습니다")

        # python-kis 사용 시도
        if self._pykis:
            try:
                stock = self._pykis.stock(stock_code)
                start_dt = datetime.strptime(start_date, "%Y%m%d")
                end_dt = datetime.strptime(end_date, "%Y%m%d")

                all_candles: List[Candle] = []
                charts = stock.chart(start=start_dt, end=end_dt, period="D", adj=adjust)

                for chart in charts:
                    candle = Candle(
                        date=chart.date if hasattr(chart, 'date') else chart.time,
                        open=float(chart.open),
                        high=float(chart.high),
                        low=float(chart.low),
                        close=float(chart.close),
                        volume=int(chart.volume),
                    )
                    all_candles.append(candle)

                all_candles.sort(key=lambda c: c.date)
                print(f"[데이터] {stock_code} 일봉 {len(all_candles)}개 조회 완료 (python-kis)")
                return all_candles
            except Exception as e:
                print(f"[경고] python-kis 차트 조회 실패, 직접 API 사용: {e}")

        # 직접 API 호출
        return self._get_daily_candles_direct(stock_code, start_date, end_date, adjust)

    def _get_daily_candles_direct(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        adjust: bool = True,
    ) -> List[Candle]:
        """직접 REST API 호출로 일봉 조회"""
        tr_id = "FHKST03010100"
        url = f"{self.config.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"

        all_candles: List[Candle] = []
        current_end = end_date

        while True:
            params = {
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": stock_code,
                "fid_input_date_1": start_date,
                "fid_input_date_2": current_end,
                "fid_period_div_code": "D",
                "fid_org_adj_prc": "0" if adjust else "1",
            }

            resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
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

            last_date_str = items[-1].get("stck_bsop_date", "")
            if not last_date_str or last_date_str <= start_date:
                break

            last_date = datetime.strptime(last_date_str, "%Y%m%d")
            current_end = (last_date - timedelta(days=1)).strftime("%Y%m%d")

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

    # ── 계좌 조회 ────────────────────────────────────

    def get_account_balance(self) -> AccountBalance:
        """계좌 잔고 조회"""
        if not self.is_available:
            raise RuntimeError("API 클라이언트가 초기화되지 않았습니다")

        # python-kis 사용 (v2.x API)
        if self._pykis:
            account = self._pykis.account()
            balance = account.balance()
            total = float(getattr(balance, 'total', 0))
            withdrawable = float(getattr(balance, 'withdrawable_amount', 0) or getattr(balance, 'withdrawable', 0))
            stock_val = float(getattr(balance, 'current_amount', 0))
            profit = float(getattr(balance, 'profit', 0))
            profit_rate = float(getattr(balance, 'profit_rate', 0))
            return AccountBalance(
                total_balance=total,
                cash_balance=withdrawable,
                stock_balance=stock_val,
                profit_loss=profit,
                profit_loss_rate=profit_rate,
            )

        # 직접 API 호출
        tr_id = "VTTC8434R" if self.config.is_paper else "TTTC8434R"
        url = f"{self.config.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"

        account_parts = self.config.account_no.split("-")
        cano = account_parts[0]
        acnt_prdt_cd = account_parts[1] if len(account_parts) > 1 else "01"

        params = {
            "CANO": cano,
            "ACNT_PRDT_CD": acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"API 오류: {data.get('msg1', 'Unknown error')}")

        output2 = data.get("output2", [{}])[0] if data.get("output2") else {}

        total_balance = float(output2.get("tot_evlu_amt", 0))
        cash_balance = float(output2.get("dnca_tot_amt", 0))
        stock_balance = float(output2.get("scts_evlu_amt", 0))
        profit_loss = float(output2.get("evlu_pfls_smtl_amt", 0))

        profit_loss_rate = 0.0
        if total_balance - profit_loss > 0:
            profit_loss_rate = (profit_loss / (total_balance - profit_loss)) * 100

        return AccountBalance(
            total_balance=total_balance,
            cash_balance=cash_balance,
            stock_balance=stock_balance,
            profit_loss=profit_loss,
            profit_loss_rate=profit_loss_rate,
        )

    def get_holdings(self) -> List[Holding]:
        """보유 종목 조회"""
        if not self.is_available:
            raise RuntimeError("API 클라이언트가 초기화되지 않았습니다")

        # python-kis 사용 (v2.x API)
        if self._pykis:
            account = self._pykis.account()
            balance = account.balance()
            holdings = []
            for stock in balance.stocks:
                qty = int(getattr(stock, 'qty', 0) or getattr(stock, 'quantity', 0))
                if qty <= 0:
                    continue
                code = getattr(stock, 'symbol', '') or getattr(stock, 'code', '')
                name = getattr(stock, 'name', '') or getattr(stock, 'market', '')
                avg = float(getattr(stock, 'purchase_price', 0) or getattr(stock, 'avg_price', 0))
                price = float(getattr(stock, 'price', 0))
                profit = float(getattr(stock, 'profit', 0))
                profit_rate = float(getattr(stock, 'profit_rate', 0))
                holdings.append(Holding(
                    stock_code=code,
                    stock_name=name,
                    quantity=qty,
                    avg_price=avg,
                    current_price=price,
                    profit_loss=profit,
                    profit_loss_rate=profit_rate,
                ))
            return holdings

        # 직접 API 호출
        tr_id = "VTTC8434R" if self.config.is_paper else "TTTC8434R"
        url = f"{self.config.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"

        account_parts = self.config.account_no.split("-")
        cano = account_parts[0]
        acnt_prdt_cd = account_parts[1] if len(account_parts) > 1 else "01"

        params = {
            "CANO": cano,
            "ACNT_PRDT_CD": acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"API 오류: {data.get('msg1', 'Unknown error')}")

        holdings = []
        for item in data.get("output1", []):
            quantity = int(item.get("hldg_qty", 0))
            if quantity <= 0:
                continue

            holdings.append(Holding(
                stock_code=item.get("pdno", ""),
                stock_name=item.get("prdt_name", ""),
                quantity=quantity,
                avg_price=float(item.get("pchs_avg_pric", 0)),
                current_price=float(item.get("prpr", 0)),
                profit_loss=float(item.get("evlu_pfls_amt", 0)),
                profit_loss_rate=float(item.get("evlu_pfls_rt", 0)),
            ))

        return holdings

    # ── 주문 ────────────────────────────────────────

    def place_order(
        self,
        stock_code: str,
        order_type: OrderType,
        is_buy: bool,
        quantity: int,
        price: int = 0,
    ) -> dict:
        """
        주문 실행

        Args:
            stock_code: 종목코드
            order_type: 주문 유형 (MARKET/LIMIT)
            is_buy: 매수 여부 (True: 매수, False: 매도)
            quantity: 주문 수량
            price: 지정가 (시장가 주문 시 0)

        Returns:
            주문 결과 딕셔너리
        """
        if not self.is_available:
            raise RuntimeError("API 클라이언트가 초기화되지 않았습니다")

        # python-kis 사용
        if self._pykis:
            stock = self._pykis.stock(stock_code)
            if is_buy:
                if order_type == OrderType.MARKET:
                    order = stock.buy(qty=quantity)
                else:
                    order = stock.buy(price=price, qty=quantity)
            else:
                if order_type == OrderType.MARKET:
                    order = stock.sell(qty=quantity)
                else:
                    order = stock.sell(price=price, qty=quantity)

            order_no = getattr(order, 'order_no', '') or getattr(order, 'odno', '')
            order_time = getattr(order, 'order_time', '') or getattr(order, 'ord_tmd', '')

            print(f"[주문] {'매수' if is_buy else '매도'} {stock_code} {quantity}주 - 주문번호: {order_no}")
            return {
                "order_no": str(order_no),
                "order_time": str(order_time),
                "stock_code": stock_code,
                "is_buy": is_buy,
                "quantity": quantity,
                "price": price if order_type == OrderType.LIMIT else 0,
                "order_type": order_type.value,
            }

        # 직접 API 호출
        if self.config.is_paper:
            tr_id = "VTTC0802U" if is_buy else "VTTC0801U"
        else:
            tr_id = "TTTC0802U" if is_buy else "TTTC0801U"

        url = f"{self.config.base_url}/uapi/domestic-stock/v1/trading/order-cash"

        account_parts = self.config.account_no.split("-")
        cano = account_parts[0]
        acnt_prdt_cd = account_parts[1] if len(account_parts) > 1 else "01"

        if order_type == OrderType.MARKET:
            ord_dvsn = "01"
            ord_unpr = "0"
        else:
            ord_dvsn = "00"
            ord_unpr = str(price)

        body = {
            "CANO": cano,
            "ACNT_PRDT_CD": acnt_prdt_cd,
            "PDNO": stock_code,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": ord_unpr,
        }

        resp = requests.post(url, headers=self._headers(tr_id), json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"주문 오류: {data.get('msg1', 'Unknown error')}")

        output = data.get("output", {})
        order_no = output.get("ODNO", "")
        order_time = output.get("ORD_TMD", "")

        print(f"[주문] {'매수' if is_buy else '매도'} {stock_code} {quantity}주 - 주문번호: {order_no}")

        return {
            "order_no": order_no,
            "order_time": order_time,
            "stock_code": stock_code,
            "is_buy": is_buy,
            "quantity": quantity,
            "price": price if order_type == OrderType.LIMIT else 0,
            "order_type": order_type.value,
        }

    # ── 스크리닝 (종목 선정) ──────────────────────────

    def get_volume_ranking(
        self,
        market: str = "J",
        count: int = 50,
    ) -> List[dict]:
        """
        거래량 순위 조회

        Args:
            market: 시장 구분 ("J": 전체, "NX": NXT)
            count: 조회 개수 (최대 30개씩 페이징)

        Returns:
            거래량 순위 리스트 [{code, name, price, volume, change_rate, market_cap, ...}]
        """
        if not self.is_available:
            raise RuntimeError("API 클라이언트가 초기화되지 않았습니다")

        tr_id = "FHPST01710000"
        url = f"{self.config.base_url}/uapi/domestic-stock/v1/quotations/volume-rank"

        params = {
            "FID_COND_MRKT_DIV_CODE": market,
            "FID_COND_SCR_DIV_CODE": "20171",
            "FID_INPUT_ISCD": "0000",        # 전체 종목
            "FID_DIV_CLS_CODE": "0",         # 전체 (보통주+우선주)
            "FID_BLNG_CLS_CODE": "0",        # 평균거래량 기준
            "FID_TRGT_CLS_CODE": "111111111",
            "FID_TRGT_EXLS_CLS_CODE": "0000000000",
            "FID_INPUT_PRICE_1": "",
            "FID_INPUT_PRICE_2": "",
            "FID_VOL_CNT": "",
            "FID_INPUT_DATE_1": "",
        }

        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"API 오류: {data.get('msg1', 'Unknown error')}")

        results = []
        for item in data.get("output", [])[:count]:
            code = item.get("mksc_shrn_iscd", "")  # 종목코드
            if not code:
                continue
            results.append({
                "code": code,
                "name": item.get("hts_kor_isnm", ""),
                "price": int(item.get("stck_prpr", 0)),
                "volume": int(item.get("acml_vol", 0)),
                "change_rate": float(item.get("prdy_ctrt", 0)),
                "acml_tr_pbmn": int(item.get("acml_tr_pbmn", 0)),  # 거래대금
                "data_rank": int(item.get("data_rank", 0)),
            })

        print(f"[스크리닝] 거래량 순위 {len(results)}개 종목 조회 완료")
        return results

    def get_market_cap_ranking(
        self,
        market: str = "0000",
        count: int = 50,
    ) -> List[dict]:
        """
        시가총액 상위 종목 조회

        Args:
            market: 시장 구분 ("0000": 전체, "0001": 코스피, "1001": 코스닥)
            count: 조회 개수

        Returns:
            시가총액 순위 리스트 [{code, name, price, market_cap, per, pbr, ...}]
        """
        if not self.is_available:
            raise RuntimeError("API 클라이언트가 초기화되지 않았습니다")

        tr_id = "FHPST01740000"
        url = f"{self.config.base_url}/uapi/domestic-stock/v1/ranking/market-cap"

        params = {
            "fid_input_price_2": "",
            "fid_cond_mrkt_div_code": "J",
            "fid_cond_scr_div_code": "20174",
            "fid_div_cls_code": "0",         # 전체
            "fid_input_iscd": market,
            "fid_trgt_cls_code": "0",
            "fid_trgt_exls_cls_code": "0",
            "fid_input_price_1": "",
            "fid_vol_cnt": "",
        }

        resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("rt_cd") != "0":
            raise RuntimeError(f"API 오류: {data.get('msg1', 'Unknown error')}")

        results = []
        for item in data.get("output", [])[:count]:
            code = item.get("mksc_shrn_iscd", "") or item.get("stck_shrn_iscd", "")
            if not code:
                continue
            results.append({
                "code": code,
                "name": item.get("hts_kor_isnm", ""),
                "price": int(item.get("stck_prpr", 0)),
                "change_rate": float(item.get("prdy_ctrt", 0)),
                "market_cap": int(item.get("stck_avls", 0)),  # 시가총액 (억원)
                "per": float(item.get("per", 0) or 0),
                "pbr": float(item.get("pbr", 0) or 0),
                "volume": int(item.get("acml_vol", 0)),
                "data_rank": int(item.get("data_rank", 0)),
            })

        print(f"[스크리닝] 시가총액 상위 {len(results)}개 종목 조회 완료")
        return results

    def get_financial_data(self, stock_code: str) -> dict:
        """
        종목 재무 데이터 조회 (손익계산서 기반)

        Args:
            stock_code: 종목코드

        Returns:
            재무 데이터 딕셔너리 {per, pbr, eps, roe, revenue, operating_profit, ...}
        """
        if not self.is_available:
            raise RuntimeError("API 클라이언트가 초기화되지 않았습니다")

        result = {"code": stock_code}

        # 1) 현재가에서 PER, PBR 등 기본 지표 조회
        try:
            price_info = self.get_current_price(stock_code)
            result["name"] = price_info.get("name", "")
            result["price"] = price_info.get("price", 0)
        except Exception:
            result["name"] = ""
            result["price"] = 0

        # 2) 손익계산서 조회 (분기 기준)
        try:
            tr_id = "FHKST66430200"
            url = f"{self.config.base_url}/uapi/domestic-stock/v1/finance/income-statement"
            params = {
                "FID_DIV_CLS_CODE": "1",  # 분기
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": stock_code,
            }

            resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("rt_cd") == "0" and data.get("output"):
                items = data["output"]
                if isinstance(items, dict):
                    items = [items]
                if items:
                    latest = items[0]
                    result["revenue"] = float(latest.get("sale_account", 0) or 0)
                    result["operating_profit"] = float(latest.get("bsop_prti", 0) or 0)
                    result["net_income"] = float(latest.get("thtr_ntin", 0) or 0)
        except Exception:
            pass

        # 3) 대차대조표 조회 (분기 기준)
        try:
            tr_id = "FHKST66430100"
            url = f"{self.config.base_url}/uapi/domestic-stock/v1/finance/balance-sheet"
            params = {
                "FID_DIV_CLS_CODE": "1",  # 분기
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": stock_code,
            }

            resp = requests.get(url, headers=self._headers(tr_id), params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("rt_cd") == "0" and data.get("output"):
                items = data["output"]
                if isinstance(items, dict):
                    items = [items]
                if items:
                    latest = items[0]
                    result["total_assets"] = float(latest.get("total_aset", 0) or 0)
                    result["total_equity"] = float(latest.get("total_cptl", 0) or 0)
                    result["total_liabilities"] = float(latest.get("total_lblt", 0) or 0)

                    # ROE 계산: 순이익 / 자기자본 * 100
                    equity = result.get("total_equity", 0)
                    net_income = result.get("net_income", 0)
                    result["roe"] = (net_income / equity * 100) if equity > 0 else 0.0
        except Exception:
            pass

        # 기본값 설정
        result.setdefault("revenue", 0)
        result.setdefault("operating_profit", 0)
        result.setdefault("net_income", 0)
        result.setdefault("total_assets", 0)
        result.setdefault("total_equity", 0)
        result.setdefault("total_liabilities", 0)
        result.setdefault("roe", 0)

        return result

    # ── 실시간 시세 (python-kis 전용) ────────────────

    def subscribe_price(self, stock_code: str, callback) -> object:
        """실시간 체결가 구독 (python-kis 필요)"""
        if not self._pykis:
            raise RuntimeError("실시간 시세는 python-kis가 필요합니다 (Python 3.10+)")

        stock = self._pykis.stock(stock_code)
        ticket = stock.on("price", callback)
        print(f"[실시간] {stock_code} 체결가 구독 시작")
        return ticket

    def subscribe_orderbook(self, stock_code: str, callback) -> object:
        """실시간 호가 구독 (python-kis 필요)"""
        if not self._pykis:
            raise RuntimeError("실시간 시세는 python-kis가 필요합니다 (Python 3.10+)")

        stock = self._pykis.stock(stock_code)
        ticket = stock.on("orderbook", callback)
        print(f"[실시간] {stock_code} 호가 구독 시작")
        return ticket


# ── API 없이 사용할 수 있는 샘플 데이터 ──────────────

def load_sample_data() -> List[Candle]:
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
            open=round(open_price, -2),
            high=round(high_price, -2),
            low=round(low_price, -2),
            close=round(close_price, -2),
            volume=max(volume, 1_000_000),
        ))

        price = close_price
        date += timedelta(days=1)

    return candles
