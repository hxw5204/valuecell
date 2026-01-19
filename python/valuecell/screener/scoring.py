"""Scoring utilities for the Quant Screener."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

from loguru import logger

from .market_data import FinancialSnapshot, PriceSnapshot


@dataclass(frozen=True)
class WideScoreResult:
    ticker: str
    score: float
    components: dict[str, float]


@dataclass(frozen=True)
class DeepScoreResult:
    ticker: str
    score: float
    components: dict[str, float]


@dataclass(frozen=True)
class RiskScoreResult:
    ticker: str
    score: float
    components: dict[str, float]


def _zscore(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    mean = sum(values.values()) / len(values)
    variance = sum((value - mean) ** 2 for value in values.values()) / len(values)
    stdev = sqrt(variance)
    if stdev == 0:
        return {key: 0.0 for key in values}
    return {key: (value - mean) / stdev for key, value in values.items()}


def _normalize_optional(values: dict[str, float | None]) -> dict[str, float]:
    filtered = {key: value for key, value in values.items() if value is not None}
    zscores = _zscore(filtered)
    return {key: zscores.get(key, 0.0) for key in values}


def _safe_growth(latest: float | None, prior: float | None) -> float | None:
    if latest is None or prior is None:
        return None
    if prior == 0:
        return None
    return (latest - prior) / abs(prior)


def score_wide(
    snapshots: Iterable[PriceSnapshot],
    weights: dict[str, float],
) -> dict[str, WideScoreResult]:
    snapshots_list = list(snapshots)
    momentum_20 = {s.ticker: s.return_20d for s in snapshots_list}
    momentum_60 = {s.ticker: s.return_60d for s in snapshots_list}
    liquidity = {
        s.ticker: s.last_close * s.avg_volume_20d for s in snapshots_list
    }
    volatility = {s.ticker: s.volatility_20d for s in snapshots_list}

    z_mom_20 = _zscore(momentum_20)
    z_mom_60 = _zscore(momentum_60)
    z_liquidity = _zscore(liquidity)
    z_volatility = _zscore(volatility)

    results: dict[str, WideScoreResult] = {}
    for snapshot in snapshots_list:
        components = {
            "fundamental_momentum": z_mom_60.get(snapshot.ticker, 0.0),
            "market_momentum": z_mom_20.get(snapshot.ticker, 0.0),
            "valuation": -z_volatility.get(snapshot.ticker, 0.0),
            "attention_underpriced": z_liquidity.get(snapshot.ticker, 0.0),
        }
        score = sum(weights.get(key, 0.0) * value for key, value in components.items())
        results[snapshot.ticker] = WideScoreResult(
            ticker=snapshot.ticker,
            score=score,
            components=components,
        )
    return results


def score_deep(
    snapshots: Iterable[PriceSnapshot],
    financials: dict[str, FinancialSnapshot],
    weights: dict[str, float],
) -> dict[str, DeepScoreResult]:
    snapshots_list = list(snapshots)
    price_accel = {
        s.ticker: (s.return_20d - 0.5 * s.return_60d) for s in snapshots_list
    }
    volume_accel = {
        s.ticker: (
            s.avg_volume_20d / s.avg_volume_60d - 1.0
            if s.avg_volume_60d != 0.0
            else None
        )
        for s in snapshots_list
    }
    revenue_growth = {
        s.ticker: _safe_growth(
            financials.get(s.ticker).revenue_latest
            if financials.get(s.ticker)
            else None,
            financials.get(s.ticker).revenue_prior
            if financials.get(s.ticker)
            else None,
        )
        for s in snapshots_list
    }
    capex_growth = {
        s.ticker: _safe_growth(
            financials.get(s.ticker).capex_latest
            if financials.get(s.ticker)
            else None,
            financials.get(s.ticker).capex_prior
            if financials.get(s.ticker)
            else None,
        )
        for s in snapshots_list
    }
    net_income_growth = {
        s.ticker: _safe_growth(
            financials.get(s.ticker).net_income_latest
            if financials.get(s.ticker)
            else None,
            financials.get(s.ticker).net_income_prior
            if financials.get(s.ticker)
            else None,
        )
        for s in snapshots_list
    }

    z_price_accel = _zscore(price_accel)
    z_volume_accel = _normalize_optional(volume_accel)
    z_revenue_growth = _normalize_optional(revenue_growth)
    z_capex_growth = _normalize_optional(capex_growth)
    z_net_income_growth = _normalize_optional(net_income_growth)

    results: dict[str, DeepScoreResult] = {}
    for snapshot in snapshots_list:
        components = {
            "supply_chain": z_volume_accel.get(snapshot.ticker, 0.0),
            "inflection": z_net_income_growth.get(snapshot.ticker, 0.0),
            "capex_cycle": z_capex_growth.get(snapshot.ticker, 0.0),
        }
        components["inflection"] += 0.5 * z_revenue_growth.get(snapshot.ticker, 0.0)
        components["inflection"] /= 1.5
        components["supply_chain"] += 0.4 * z_price_accel.get(snapshot.ticker, 0.0)
        components["supply_chain"] /= 1.4
        score = sum(weights.get(key, 0.0) * value for key, value in components.items())
        results[snapshot.ticker] = DeepScoreResult(
            ticker=snapshot.ticker,
            score=score,
            components=components,
        )
    return results


def score_risk(
    snapshots: Iterable[PriceSnapshot],
    weights: dict[str, float],
) -> dict[str, RiskScoreResult]:
    snapshots_list = list(snapshots)
    volatility = {s.ticker: s.volatility_20d for s in snapshots_list}
    drawdown = {s.ticker: abs(s.max_drawdown_60d) for s in snapshots_list}
    z_volatility = _zscore(volatility)
    z_drawdown = _zscore(drawdown)

    results: dict[str, RiskScoreResult] = {}
    for snapshot in snapshots_list:
        components = {
            "volatility": z_volatility.get(snapshot.ticker, 0.0),
            "balance_sheet": z_drawdown.get(snapshot.ticker, 0.0),
        }
        score = sum(weights.get(key, 0.0) * value for key, value in components.items())
        results[snapshot.ticker] = RiskScoreResult(
            ticker=snapshot.ticker,
            score=score,
            components=components,
        )
    if not snapshots_list:
        logger.warning("No snapshots available for risk scoring")
    return results
