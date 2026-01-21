"""Universe loading utilities for the Quant Screener."""

from __future__ import annotations

import json
import time
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
SEC_FETCH_MAX_RETRIES: int = 3
SEC_FETCH_RETRY_BACKOFF_S: float = 1.5

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


def _fetch_sec_universe() -> dict:
    last_error: Exception | None = None
    for attempt in range(1, SEC_FETCH_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    SEC_TICKER_EXCHANGE_URL,
                    headers={"User-Agent": SEC_USER_AGENT},
                )
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Failed to fetch SEC universe on attempt {attempt}/{max_retries}: {error}",
                attempt=attempt,
                max_retries=SEC_FETCH_MAX_RETRIES,
                error=exc,
            )
            if attempt < SEC_FETCH_MAX_RETRIES:
                time.sleep(SEC_FETCH_RETRY_BACKOFF_S * attempt)
    logger.debug(
        "SEC universe fetch failed after {max_retries} attempts. "
        "URL={url} user_agent={user_agent} error={error}",
        max_retries=SEC_FETCH_MAX_RETRIES,
        url=SEC_TICKER_EXCHANGE_URL,
        user_agent=SEC_USER_AGENT,
        error=last_error,
    )
    print(
        "SEC universe fetch failed after {max_retries} attempts. "
        "URL={url} user_agent={user_agent} error={error}".format(
            max_retries=SEC_FETCH_MAX_RETRIES,
            url=SEC_TICKER_EXCHANGE_URL,
            user_agent=SEC_USER_AGENT,
            error=last_error,
        )
    )
    raise RuntimeError("SEC universe fetch failed") from last_error


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


def _get_item_value(
    item: object,
    field_indices: dict[str, int] | None,
    field_name: str,
) -> str:
    if isinstance(item, dict):
        return str(item.get(field_name, ""))
    if isinstance(item, (list, tuple)) and field_indices is not None:
        index = field_indices.get(field_name)
        if index is None or index >= len(item):
            return ""
        return str(item[index])
    return ""


def _parse_universe(payload: dict, allowlist: set[str]) -> list[UniverseTicker]:
    entries: list[UniverseTicker] = []
    field_indices: dict[str, int] | None = None
    fields = payload.get("fields")
    if isinstance(fields, list):
        field_indices = {
            field: index
            for index, field in enumerate(fields)
            if isinstance(field, str)
        }
    items = payload.get("data", [])
    if (
        isinstance(items, list)
        and items
        and isinstance(items[0], (list, tuple))
        and field_indices is None
    ):
        logger.warning(
            "SEC universe payload missing fields metadata; unable to parse list rows."
        )
        return []
    for item in items:
        ticker = _get_item_value(item, field_indices, "ticker").upper()
        name = _get_item_value(item, field_indices, "name").strip()
        exchange = _normalize_exchange(
            _get_item_value(item, field_indices, "exchange").strip()
        )
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


def load_us_universe(
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
            payload = _fetch_sec_universe()
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
