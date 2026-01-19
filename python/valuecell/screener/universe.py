"""Universe loading utilities for the Quant Screener."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from loguru import logger

from valuecell.utils.env import ensure_system_env_dir

from .constants import DATA_DIR_NAME


SEC_TICKER_EXCHANGE_URL: str = (
    "https://www.sec.gov/files/company_tickers_exchange.json"
)
UNIVERSE_CACHE_TTL_DAYS: int = 7
SEC_USER_AGENT: str = "ValueCellQuantScreener/0.1 (research@valuecell.ai)"

_EXCHANGE_NORMALIZATION: dict[str, str] = {
    "nasdaq": "NASDAQ",
    "nyse": "NYSE",
    "nyse american": "AMEX",
    "nyse arca": "NYSE",
    "nyse mkt": "AMEX",
    "amex": "AMEX",
    "nyse american llc": "AMEX",
}
_EXCHANGE_PRIORITY: dict[str, int] = {"NASDAQ": 3, "NYSE": 2, "AMEX": 1}


@dataclass(frozen=True)
class UniverseTicker:
    """Normalized ticker entry for the screener universe."""

    ticker: str
    symbol: str
    name: str
    exchange: str


def get_universe_cache_path() -> Path:
    """Return the local cache path for the SEC ticker list."""
    base_dir = ensure_system_env_dir()
    universe_dir = base_dir / DATA_DIR_NAME / "screener" / "universe"
    universe_dir.mkdir(parents=True, exist_ok=True)
    return universe_dir / "company_tickers_exchange.json"


def _is_cache_fresh(path: Path, ttl_days: int) -> bool:
    if not path.exists():
        return False
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return False
    return datetime.now(timezone.utc) - mtime < timedelta(days=ttl_days)


async def _fetch_sec_universe() -> dict:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            SEC_TICKER_EXCHANGE_URL,
            headers={"User-Agent": SEC_USER_AGENT},
        )
        response.raise_for_status()
        return response.json()


def _normalize_exchange(exchange: str) -> str | None:
    if not exchange:
        return None
    exchange_key = exchange.strip().lower()
    return _EXCHANGE_NORMALIZATION.get(exchange_key, exchange.strip().upper())


def _dedupe_universe(entries: list[UniverseTicker]) -> list[UniverseTicker]:
    deduped: dict[str, UniverseTicker] = {}
    for entry in entries:
        existing = deduped.get(entry.symbol)
        if not existing:
            deduped[entry.symbol] = entry
            continue
        existing_priority = _EXCHANGE_PRIORITY.get(existing.exchange, 0)
        new_priority = _EXCHANGE_PRIORITY.get(entry.exchange, 0)
        if new_priority > existing_priority:
            deduped[entry.symbol] = entry
    return list(deduped.values())


def _parse_universe(payload: dict, allowlist: set[str]) -> list[UniverseTicker]:
    entries: list[UniverseTicker] = []
    items = payload.get("data", [])
    for item in items:
        ticker = str(item.get("ticker", "")).upper()
        name = str(item.get("name", "")).strip()
        exchange = _normalize_exchange(str(item.get("exchange", "")).strip())
        if not ticker or not exchange or exchange not in allowlist:
            continue
        symbol = ticker
        entries.append(
            UniverseTicker(
                ticker=f"{exchange}:{symbol}",
                symbol=symbol,
                name=name or symbol,
                exchange=exchange,
            )
        )
    return _dedupe_universe(entries)


async def load_us_universe(
    allowlist: list[str],
    ttl_days: int = UNIVERSE_CACHE_TTL_DAYS,
) -> list[UniverseTicker]:
    """Load the U.S. equity universe from SEC data with local caching."""
    cache_path = get_universe_cache_path()
    payload: dict | None = None
    if _is_cache_fresh(cache_path, ttl_days):
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.info("Loaded cached SEC universe from {path}", path=cache_path)
        except Exception as exc:
            logger.warning(
                "Failed to read cached universe at {path}: {error}",
                path=cache_path,
                error=exc,
            )
            payload = None
    if payload is None:
        try:
            payload = await _fetch_sec_universe()
            cache_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Fetched SEC universe and cached to {path}", path=cache_path)
        except Exception as exc:
            logger.warning(
                "Failed to fetch SEC universe: {error}",
                error=exc,
            )
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                logger.info(
                    "Loaded stale SEC universe from cache at {path}",
                    path=cache_path,
                )
            else:
                return []

    normalized_allowlist = {item.strip().upper() for item in allowlist}
    return _parse_universe(payload, normalized_allowlist)
