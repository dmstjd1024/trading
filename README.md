# 한국 주식 트레이딩 시스템

한국투자증권 REST API 기반 데이터 수집 + 백테스팅 프레임워크

## 프로젝트 구조

```
korean-stock-trader/
├── config.py            # API 키, 설정값 관리
├── api_client.py        # 한국투자증권 API 클라이언트
├── data_collector.py    # 주가 데이터 수집 및 저장
├── backtest_engine.py   # 백테스팅 엔진
├── strategies/          # 전략 플러그인 디렉토리
│   ├── base.py          # 전략 추상 클래스
│   ├── golden_cross.py  # 예시: 골든크로스 전략
│   └── rsi_strategy.py  # 예시: RSI 전략
├── models.py            # 데이터 모델 (Trade, Position 등)
├── utils.py             # 유틸리티 함수
├── main.py              # 진입점
└── requirements.txt
```

## 설정 방법

1. 한국투자증권 계좌 개설 후 API 신청
2. 모의투자 앱키/시크릿 발급
3. `config.py`에 키 입력 (또는 환경변수 사용)
4. `pip install -r requirements.txt`
5. `python main.py collect --code 005930` (삼성전자 데이터 수집)
6. `python main.py backtest --strategy golden_cross --code 005930`

## 전략 추가 방법

`strategies/base.py`의 `Strategy` 추상 클래스를 상속하여 새 전략을 만들 수 있습니다.
`on_candle()` 메서드에서 매수/매도 시그널을 반환하면 됩니다.
