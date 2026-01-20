"""Market data utilities for Quant Screener scoring."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import yfinance as yf
from loguru import logger

from . import constants


@dataclass(frozen=True)
class PriceSnapshot:
    """Derived price metrics from historical data."""

    ticker: str
    symbol: str
    last_close: float
    last_volume: float
    avg_volume_20d: float
    avg_volume_60d: float
    return_20d: float
    return_60d: float
    volatility_20d: float
    max_drawdown_60d: float
    data_end: datetime


@dataclass(frozen=True)
class AssetSnapshot:
    """Metadata snapshot for assets used in screening filters."""

    ticker: str
    market_cap: float | None
    quote_type: str | None


@dataclass(frozen=True)
class FinancialSnapshot:
    """Key financial statement snapshots for deep scoring."""

    ticker: str
    capex_latest: float | None
    capex_prior: float | None
    revenue_latest: float | None
    revenue_prior: float | None
    net_income_latest: float | None
    net_income_prior: float | None
    period_latest: str | None
    period_prior: str | None


class AsyncRateLimiter:
    """Ensure a minimum interval between external requests."""

    def __init__(self, min_interval_s: float) -> None:
        self._min_interval_s = min_interval_s
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        if self._min_interval_s <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait_for = self._min_interval_s - (now - self._last_call)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
            self._last_call = time.monotonic()


def _split_symbol(ticker: str) -> str:
    return ticker.split(":", 1)[1] if ":" in ticker else ticker


def _download_prices(symbols: list[str], period_days: int) -> pd.DataFrame:
    period = f"{period_days}d"
    return yf.download(
        tickers=" ".join(symbols),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )


async def fetch_price_history(
    tickers: Iterable[str],
    period_days: int = 120,
    batch_size: int = 200,
) -> dict[str, pd.DataFrame]:
    tickers_list = list(tickers)
    if not tickers_list:
        return {}
    results: dict[str, pd.DataFrame] = {}
    for start in range(0, len(tickers_list), batch_size):
        batch = tickers_list[start : start + batch_size]
        symbols = [_split_symbol(ticker) for ticker in batch]
        try:
            data = await asyncio.to_thread(_download_prices, symbols, period_days)
        except Exception as exc:
            logger.warning(
                "Failed to fetch price history for batch {start}-{end}: {error}",
                start=start,
                end=min(start + batch_size, len(tickers_list)),
                error=exc,
            )
            continue
        if data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            for symbol in symbols:
                if symbol not in data.columns.get_level_values(0):
                    continue
                df = data[symbol].dropna()
                if not df.empty:
                    results[_symbol_to_ticker(batch, symbol)] = df
        else:
            df = data.dropna()
            if df.empty:
                continue
            symbol = symbols[0]
            results[_symbol_to_ticker(batch, symbol)] = df
        logger.info(
            "Fetched price history for batch {start}-{end} ({count} symbols)",
            start=start,
            end=min(start + batch_size, len(tickers_list)),
            count=len(symbols),
        )
    return results


def _symbol_to_ticker(batch: list[str], symbol: str) -> str:
    for ticker in batch:
        if _split_symbol(ticker) == symbol:
            return ticker
    return symbol


def build_price_snapshot(ticker: str, df: pd.DataFrame) -> PriceSnapshot | None:
    if df.empty or len(df) < 30:
        return None
    closes = df["Close"].dropna()
    volumes = df["Volume"].dropna()
    if closes.empty or volumes.empty:
        return None
    last_close = float(closes.iloc[-1])
    last_volume = float(volumes.iloc[-1])
    avg_volume_20d = float(volumes.tail(20).mean())
    avg_volume_60d = float(volumes.tail(60).mean())
    if len(closes) > 21:
        return_20d = float(closes.iloc[-1] / closes.iloc[-21] - 1.0)
    else:
        return_20d = 0.0
    if len(closes) > 61:
        return_60d = float(closes.iloc[-1] / closes.iloc[-61] - 1.0)
    else:
        return_60d = 0.0
    daily_returns = closes.pct_change().dropna().tail(20)
    volatility_20d = float(daily_returns.std()) if not daily_returns.empty else 0.0
    window = closes.tail(60)
    running_max = window.cummax()
    drawdowns = (window - running_max) / running_max
    max_drawdown_60d = float(drawdowns.min()) if not drawdowns.empty else 0.0
    data_end = closes.index[-1]
    if not isinstance(data_end, datetime):
        data_end = datetime.combine(
            data_end, datetime.min.time(), tzinfo=timezone.utc
        )
    return PriceSnapshot(
        ticker=ticker,
        symbol=_split_symbol(ticker),
        last_close=last_close,
        last_volume=last_volume,
        avg_volume_20d=avg_volume_20d,
        avg_volume_60d=avg_volume_60d,
        return_20d=return_20d,
        return_60d=return_60d,
        volatility_20d=volatility_20d,
        max_drawdown_60d=max_drawdown_60d,
        data_end=data_end,
    )


async def fetch_financial_snapshots(
    tickers: Iterable[str],
    max_concurrency: int = constants.FINANCIAL_MAX_CONCURRENCY,
    min_interval_s: float = constants.FINANCIAL_MIN_INTERVAL_S,
) -> dict[str, FinancialSnapshot]:
    tickers_list = list(tickers)
    if not tickers_list:
        return {}
    semaphore = asyncio.Semaphore(max_concurrency)
    limiter = AsyncRateLimiter(min_interval_s)
    tasks = [
        _fetch_financial_snapshot(ticker, semaphore, limiter)
        for ticker in tickers_list
    ]
    results = await asyncio.gather(*tasks)
    snapshots: dict[str, FinancialSnapshot] = {}
    for snapshot in results:
        if snapshot:
            snapshots[snapshot.ticker] = snapshot
    return snapshots


async def fetch_asset_metadata(
    tickers: Iterable[str],
    max_concurrency: int = constants.METADATA_MAX_CONCURRENCY,
    min_interval_s: float = constants.METADATA_MIN_INTERVAL_S,
) -> dict[str, AssetSnapshot]:
    tickers_list = list(tickers)
    if not tickers_list:
        return {}
    semaphore = asyncio.Semaphore(max_concurrency)
    limiter = AsyncRateLimiter(min_interval_s)
    tasks = [
        _fetch_asset_metadata(ticker, semaphore, limiter)
        for ticker in tickers_list
    ]
    results = await asyncio.gather(*tasks)
    metadata: dict[str, AssetSnapshot] = {}
    for snapshot in results:
        if snapshot:
            metadata[snapshot.ticker] = snapshot
    return metadata


async def _fetch_financial_snapshot(
    ticker: str,
    semaphore: asyncio.Semaphore,
    limiter: AsyncRateLimiter,
) -> FinancialSnapshot | None:
    async with semaphore:
        await limiter.wait()
        return await asyncio.to_thread(_fetch_financial_snapshot_sync, ticker)


def _fetch_financial_snapshot_sync(ticker: str) -> FinancialSnapshot | None:
    symbol = _split_symbol(ticker)
    yf_ticker = yf.Ticker(symbol)
    try:
        cashflow = yf_ticker.cashflow
        financials = yf_ticker.financials
    except Exception as exc:
        logger.warning(
            "Failed to load financials for {ticker}: {error}",
            ticker=ticker,
            error=exc,
        )
        return None

    capex_series = _extract_financial_series(
        cashflow,
        ["capital expenditures", "capital expenditure"],
    )
    revenue_series = _extract_financial_series(
        financials,
        ["total revenue", "totalrevenue", "revenue"],
    )
    net_income_series = _extract_financial_series(
        financials,
        ["net income", "netincome", "net income applicable to common shares"],
    )

    capex_latest, capex_prior, period_latest, period_prior = _latest_two_values(
        capex_series
    )
    revenue_latest, revenue_prior, _, _ = _latest_two_values(revenue_series)
    net_income_latest, net_income_prior, _, _ = _latest_two_values(net_income_series)

    return FinancialSnapshot(
        ticker=ticker,
        capex_latest=capex_latest,
        capex_prior=capex_prior,
        revenue_latest=revenue_latest,
        revenue_prior=revenue_prior,
        net_income_latest=net_income_latest,
        net_income_prior=net_income_prior,
        period_latest=period_latest,
        period_prior=period_prior,
    )


async def _fetch_asset_metadata(
    ticker: str,
    semaphore: asyncio.Semaphore,
    limiter: AsyncRateLimiter,
) -> AssetSnapshot | None:
    async with semaphore:
        await limiter.wait()
        return await asyncio.to_thread(_fetch_asset_metadata_sync, ticker)


def _fetch_asset_metadata_sync(ticker: str) -> AssetSnapshot | None:
    symbol = _split_symbol(ticker)
    yf_ticker = yf.Ticker(symbol)
    try:
        info = yf_ticker.get_info()
    except Exception as exc:
        logger.warning(
            "Failed to load asset metadata for {ticker} via info: {error}",
            ticker=ticker,
            error=exc,
        )
        return _fetch_asset_metadata_fast(ticker, yf_ticker)
    if not isinstance(info, dict) or not info:
        logger.warning("No asset metadata returned for {ticker}", ticker=ticker)
        return _fetch_asset_metadata_fast(ticker, yf_ticker)
    market_cap = _normalize_market_cap(info.get("marketCap"))
    quote_type = info.get("quoteType") or info.get("quote_type")
    return AssetSnapshot(
        ticker=ticker,
        market_cap=market_cap,
        quote_type=str(quote_type) if quote_type else None,
    )


def _fetch_asset_metadata_fast(
    ticker: str, yf_ticker: yf.Ticker
) -> AssetSnapshot | None:
    try:
        fast_info = yf_ticker.fast_info
    except Exception as exc:
        logger.warning(
            "Failed to load asset metadata for {ticker} via fast_info: {error}",
            ticker=ticker,
            error=exc,
        )
        return None
    market_cap = _normalize_market_cap(getattr(fast_info, "market_cap", None))
    return AssetSnapshot(ticker=ticker, market_cap=market_cap, quote_type=None)


def _normalize_market_cap(raw_value: float | int | None) -> float | None:
    if raw_value is None:
        return None
    return float(raw_value)


def _extract_financial_series(
    df: pd.DataFrame | None, labels: list[str]
) -> pd.Series | None:
    if df is None or df.empty:
        return None
    normalized_index = {str(idx).lower(): idx for idx in df.index}
    for label in labels:
        for candidate in normalized_index:
            if label in candidate:
                return df.loc[normalized_index[candidate]]
    return None


def _latest_two_values(
    series: pd.Series | None,
) -> tuple[float | None, float | None, str | None, str | None]:
    if series is None:
        return None, None, None, None
    cleaned = series.dropna()
    if cleaned.empty:
        return None, None, None, None
    values = cleaned.values.tolist()
    periods = [str(item) for item in cleaned.index.tolist()]
    latest_value = float(values[0]) if len(values) >= 1 else None
    prior_value = float(values[1]) if len(values) >= 2 else None
    latest_period = periods[0] if len(periods) >= 1 else None
    prior_period = periods[1] if len(periods) >= 2 else None
    return latest_value, prior_value, latest_period, prior_period
