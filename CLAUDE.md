---
description: 
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Korean Stock Trading System (한국 주식 트레이딩 시스템) - a quantitative trading framework for Korean stocks using the Korea Investment & Securities (한국투자증권) REST API. Combines data collection, backtesting with realistic cost modeling (commission, slippage, tax), and a plugin-based strategy architecture.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Demo mode with sample data (no API key needed)
python main.py demo

# Collect historical data from API
python main.py collect --code 005930 --start 20240101 --end 20241231

# Run backtest on a strategy
python main.py backtest --strategy golden_cross --code 005930

# Compare all strategies
python main.py compare --code 005930
```

## Architecture

**Strategy Pattern**: All strategies inherit from `Strategy` (in `base.py`) and implement:
- `name` property - strategy identifier
- `on_init(data)` - calculate indicators on full DataFrame
- `on_candle(index, row, position, data)` - return `(Signal, ratio)` tuple
- `on_finish()` - optional cleanup

**Data Flow**: `KISClient` (api_client.py) → `DataStore` (data_collector.py, SQLite) → DataFrame → `BacktestEngine` (backtest_engine.py) → results

**Key Models** (models.py): `Signal` enum (BUY/SELL/HOLD), `Candle`, `Trade`, `Position`, `BacktestResult`

**Configuration** (config.py):
- `KISConfig` - API credentials via env vars: `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_ACCOUNT_NO`
- `BacktestConfig` - initial capital, commission (0.015%), slippage (0.1%), tax (0.18%)
- `is_paper=True` selects paper trading API endpoint

## Project Structure

The codebase uses a `strategies/` package structure as documented in README.md:
- `strategies/__init__.py` - exports `STRATEGIES` dict and strategy classes
- `strategies/base.py` - abstract Strategy class
- `strategies/golden_cross.py` - Golden Cross strategy
- `strategies/rsi_strategy.py` - RSI strategy

## Adding New Strategies

1. Inherit from `Strategy` class in `base.py`
2. Implement `name` property and `on_candle()` method
3. Register in `STRATEGIES` dict (needs to be created in package `__init__.py`)
4. Use `on_init()` to pre-calculate indicators on the DataFrame
