"""Quant Screener pipeline scaffolding."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Iterable

from loguru import logger

from valuecell.utils.uuid import generate_uuid

from .config import load_screener_config
from . import market_data
from . import scoring
from .schemas import (
    LogicGraph,
    LogicGraphEdge,
    LogicGraphNode,
    ScreenerCandidate,
    ScreenerCandidateDetail,
    ScreenerEvidence,
    ScreenerRunConfig,
    ScreenerRunLog,
    ScreenerRunLogStep,
    ScreenerRunMeta,
    ScreenerRunResult,
    ScreenerScoreBreakdown,
)
from .storage import (
    write_candidate_detail,
    write_candidates,
    write_evaluation,
    write_report,
    write_run_meta,
)
from .universe import UniverseTicker, load_us_universe

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class CandidateContext:
    """Intermediate context for candidate assembly."""

    universe: UniverseTicker
    snapshot: market_data.PriceSnapshot
    asset: market_data.AssetSnapshot | None
    wide: scoring.WideScoreResult
    deep: scoring.DeepScoreResult
    risk: scoring.RiskScoreResult
    financial: market_data.FinancialSnapshot | None


def _safe_growth_rate(latest: float | None, prior: float | None) -> float | None:
    if latest is None or prior is None:
        return None
    if prior == 0:
        return None
    return (latest - prior) / abs(prior)


def _has_financial_periods(
    period_prior: str | None, period_latest: str | None, ticker: str
) -> bool:
    if period_prior is None or period_latest is None:
        logger.warning(
            "Missing financial periods for {ticker} (prior={prior}, latest={latest})",
            ticker=ticker,
            prior=period_prior,
            latest=period_latest,
        )
        return False
    return True


def _evidence_metric_name(evidence_id: str) -> str:
    if "momentum" in evidence_id:
        return "Price momentum"
    if "liquidity" in evidence_id:
        return "Liquidity (dollar volume)"
    if "profile" in evidence_id:
        return "Market cap"
    if "capex" in evidence_id:
        return "Capex growth"
    if "earnings" in evidence_id:
        return "Net income growth"
    return "Metric"


def _evidence_metric_unit(evidence_id: str) -> str:
    if "liquidity" in evidence_id:
        return "usd"
    if "profile" in evidence_id:
        return "usd"
    if "momentum" in evidence_id or "capex" in evidence_id or "earnings" in evidence_id:
        return "pct"
    return "unit"


def _evidence_metric_value(evidence: ScreenerEvidence) -> float | None:
    if "momentum" in evidence.evidence_id:
        return float(evidence.structured.get("return_20d", 0.0))
    if "liquidity" in evidence.evidence_id:
        return float(evidence.structured.get("dollar_volume_20d", 0.0))
    if "profile" in evidence.evidence_id:
        return float(evidence.structured.get("market_cap", 0.0))
    if "capex" in evidence.evidence_id:
        return float(evidence.structured.get("capex_growth", 0.0))
    if "earnings" in evidence.evidence_id:
        return float(evidence.structured.get("net_income_growth", 0.0))
    return None


def _is_fund_like_name(name: str) -> bool:
    if not name:
        return False
    keywords = (
        "ETF",
        "ETN",
        "TRUST",
        "FUND",
        "MUTUAL",
        "INDEX",
        "ISHARES",
        "PROSHARES",
        "SPDR",
    )
    upper = name.upper()
    return any(keyword in upper for keyword in keywords)


class ScreenerPipeline:
    """Wide -> Deep pipeline that writes run artifacts locally."""

    def run(self, config: ScreenerRunConfig) -> ScreenerRunResult:
        """Execute the pipeline and persist outputs."""
        started_at = datetime.now(timezone.utc)
        run_id = generate_uuid("run")
        config_payload = self._config_payload(config)
        config_hash = self._hash_payload(config_payload)
        universe_config = config_payload.get("universe", {})
        exchange_allowlist = universe_config.get("exchange_allowlist", [])
        universe = asyncio.run(load_us_universe(exchange_allowlist))
        universe_map = {item.ticker: item for item in universe}
        price_history = asyncio.run(
            market_data.fetch_price_history(universe_map.keys())
        )
        price_snapshots = self._build_price_snapshots(
            universe_map, price_history, universe_config, config
        )
        asset_metadata = asyncio.run(
            market_data.fetch_asset_metadata(
                universe_map.keys(),
                max_concurrency=2,
                delay_s=0.2,
            )
        )
        price_snapshots = self._filter_asset_snapshots(
            price_snapshots, universe_map, asset_metadata, universe_config
        )
        weights = config_payload.get("weights", {})
        wide_scores = scoring.score_wide(
            price_snapshots, weights.get("wide", {})
        )
        top_wide_tickers = self._select_top_wide(
            wide_scores, config.top_k
        )
        top_snapshots = [
            snapshot
            for snapshot in price_snapshots
            if snapshot.ticker in top_wide_tickers
        ]
        financials = asyncio.run(
            market_data.fetch_financial_snapshots(top_wide_tickers)
        )
        deep_scores = scoring.score_deep(
            top_snapshots, financials, weights.get("deep", {})
        )
        risk_scores = scoring.score_risk(
            top_snapshots, weights.get("risk", {})
        )
        contexts = self._build_candidate_contexts(
            top_wide_tickers,
            universe_map,
            top_snapshots,
            asset_metadata,
            wide_scores,
            deep_scores,
            risk_scores,
            financials,
        )
        candidates = self._build_candidates(contexts, config)
        candidate_details = self._build_candidate_details(
            candidates,
            contexts,
        )
        evaluation = self._build_evaluation()
        data_snapshot_hash = self._hash_payload(
            self._snapshot_payload(
                universe, price_snapshots, financials, asset_metadata
            )
        )
        ended_at = datetime.now(timezone.utc)
        run_log = self._build_run_log(
            run_id=run_id,
            started_at=started_at,
            ended_at=ended_at,
            candidates=candidates,
            config_payload=config_payload,
            universe_size=len(universe),
            data_points=len(price_snapshots),
        )
        meta = ScreenerRunMeta(
            run_id=run_id,
            as_of_date=started_at.date().isoformat(),
            run_timestamp_utc=started_at,
            data_cutoff=started_at.date().isoformat(),
            started_at=started_at,
            ended_at=ended_at,
            config_hash=config_hash,
            code_git_sha=self._get_git_sha(),
            data_snapshot_hash=data_snapshot_hash,
            universe_size=len(universe),
            status="completed",
            config=config,
            run_log=run_log,
        )

        write_run_meta(meta)
        write_candidates(run_id, candidates)
        for ticker, detail in candidate_details.items():
            write_candidate_detail(run_id, ticker, detail)
        write_report(run_id, self._build_summary(meta, candidates))
        write_evaluation(run_id, evaluation)

        logger.info("Screener run {run_id} completed", run_id=run_id)
        return ScreenerRunResult(
            meta=meta,
            candidates=candidates,
            candidate_details=candidate_details,
            evaluation=evaluation,
        )

    def _config_payload(self, config: ScreenerRunConfig) -> dict:
        weights_key = f"weights_{config.frequency}"
        return {
            "request": config.model_dump(),
            "universe": load_screener_config("universe"),
            "weights": load_screener_config(weights_key),
            "deep_checklist": load_screener_config("deep_checklist"),
        }

    @staticmethod
    def _hash_payload(payload: object) -> str:
        payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _get_git_sha() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    @staticmethod
    def _snapshot_payload(
        universe: Iterable[UniverseTicker],
        price_snapshots: Iterable[market_data.PriceSnapshot],
        financials: dict[str, market_data.FinancialSnapshot],
        asset_metadata: dict[str, market_data.AssetSnapshot],
    ) -> dict:
        return {
            "universe": [item.ticker for item in universe],
            "price_snapshots": [
                {
                    "ticker": snapshot.ticker,
                    "last_close": snapshot.last_close,
                    "avg_volume_20d": snapshot.avg_volume_20d,
                    "return_20d": snapshot.return_20d,
                    "return_60d": snapshot.return_60d,
                }
                for snapshot in price_snapshots
            ],
            "asset_metadata": [
                {
                    "ticker": snapshot.ticker,
                    "market_cap": (
                        asset_metadata.get(snapshot.ticker).market_cap
                        if asset_metadata.get(snapshot.ticker)
                        else None
                    ),
                    "quote_type": (
                        asset_metadata.get(snapshot.ticker).quote_type
                        if asset_metadata.get(snapshot.ticker)
                        else None
                    ),
                }
                for snapshot in price_snapshots
            ],
            "financials": {
                ticker: {
                    "capex_latest": data.capex_latest,
                    "capex_prior": data.capex_prior,
                    "revenue_latest": data.revenue_latest,
                    "revenue_prior": data.revenue_prior,
                    "net_income_latest": data.net_income_latest,
                    "net_income_prior": data.net_income_prior,
                }
                for ticker, data in financials.items()
            },
        }

    def _build_price_snapshots(
        self,
        universe_map: dict[str, UniverseTicker],
        price_history: dict[str, "pd.DataFrame"],
        universe_config: dict,
        config: ScreenerRunConfig,
    ) -> list[market_data.PriceSnapshot]:
        min_price = float(universe_config.get("min_price", 0.0))
        min_dollar_volume = float(universe_config.get("min_dollar_volume", 0.0))
        min_avg_volume = float(universe_config.get("min_avg_volume_20d", 0.0))
        if config.liquidity_min is not None:
            min_dollar_volume = float(config.liquidity_min)
        snapshots: list[market_data.PriceSnapshot] = []
        for ticker, df in price_history.items():
            if ticker not in universe_map:
                continue
            snapshot = market_data.build_price_snapshot(ticker, df)
            if snapshot is None:
                continue
            if snapshot.last_close < min_price:
                continue
            if snapshot.avg_volume_20d < min_avg_volume:
                continue
            dollar_volume = snapshot.last_close * snapshot.avg_volume_20d
            if dollar_volume < min_dollar_volume:
                continue
            snapshots.append(snapshot)
        logger.info(
            "Built {count} price snapshots (min_price={min_price}, "
            "min_dollar_volume={min_dollar_volume}, "
            "min_avg_volume_20d={min_avg_volume})",
            count=len(snapshots),
            min_price=min_price,
            min_dollar_volume=min_dollar_volume,
            min_avg_volume=min_avg_volume,
        )
        return snapshots

    @staticmethod
    def _filter_asset_snapshots(
        snapshots: Iterable[market_data.PriceSnapshot],
        universe_map: dict[str, UniverseTicker],
        asset_metadata: dict[str, market_data.AssetSnapshot],
        universe_config: dict,
    ) -> list[market_data.PriceSnapshot]:
        min_market_cap = float(universe_config.get("min_market_cap", 0.0))
        asset_type_allowlist = [
            str(item).upper()
            for item in universe_config.get("asset_type_allowlist", [])
        ]
        exclude_fund_like = bool(
            universe_config.get("exclude_fund_like_names", False)
        )
        filtered: list[market_data.PriceSnapshot] = []
        removed_market_cap = 0
        removed_type = 0
        removed_name = 0
        for snapshot in snapshots:
            metadata = asset_metadata.get(snapshot.ticker)
            if min_market_cap > 0:
                market_cap = metadata.market_cap if metadata else None
                if market_cap is None or market_cap < min_market_cap:
                    removed_market_cap += 1
                    continue
            if asset_type_allowlist:
                quote_type = (
                    metadata.quote_type.upper()
                    if metadata and metadata.quote_type
                    else None
                )
                if quote_type is None or quote_type not in asset_type_allowlist:
                    removed_type += 1
                    continue
            if exclude_fund_like:
                universe = universe_map.get(snapshot.ticker)
                if universe and _is_fund_like_name(universe.name):
                    removed_name += 1
                    continue
            filtered.append(snapshot)
        logger.info(
            "Filtered snapshots by metadata (kept={kept}, removed_market_cap={cap}, "
            "removed_type={type_removed}, removed_name={name_removed})",
            kept=len(filtered),
            cap=removed_market_cap,
            type_removed=removed_type,
            name_removed=removed_name,
        )
        return filtered

    @staticmethod
    def _select_top_wide(
        wide_scores: dict[str, scoring.WideScoreResult],
        top_k: int,
    ) -> list[str]:
        ranked = sorted(wide_scores.values(), key=lambda item: item.score, reverse=True)
        return [item.ticker for item in ranked[:top_k]]

    def _build_candidate_contexts(
        self,
        top_wide_tickers: list[str],
        universe_map: dict[str, UniverseTicker],
        snapshots: Iterable[market_data.PriceSnapshot],
        asset_metadata: dict[str, market_data.AssetSnapshot],
        wide_scores: dict[str, scoring.WideScoreResult],
        deep_scores: dict[str, scoring.DeepScoreResult],
        risk_scores: dict[str, scoring.RiskScoreResult],
        financials: dict[str, market_data.FinancialSnapshot],
    ) -> dict[str, CandidateContext]:
        snapshot_map = {snapshot.ticker: snapshot for snapshot in snapshots}
        contexts: dict[str, CandidateContext] = {}
        for ticker in top_wide_tickers:
            universe = universe_map.get(ticker)
            snapshot = snapshot_map.get(ticker)
            wide = wide_scores.get(ticker)
            deep = deep_scores.get(ticker)
            risk = risk_scores.get(ticker)
            if not universe or not snapshot or not wide:
                continue
            if deep is None:
                deep = scoring.DeepScoreResult(
                    ticker=ticker,
                    score=0.0,
                    components={
                        "supply_chain": 0.0,
                        "inflection": 0.0,
                        "capex_cycle": 0.0,
                    },
                )
            if risk is None:
                risk = scoring.RiskScoreResult(
                    ticker=ticker,
                    score=0.0,
                    components={"balance_sheet": 0.0, "volatility": 0.0},
                )
            contexts[ticker] = CandidateContext(
                universe=universe,
                snapshot=snapshot,
                asset=asset_metadata.get(ticker),
                wide=wide,
                deep=deep,
                risk=risk,
                financial=financials.get(ticker),
            )
        return contexts

    def _build_candidates(
        self,
        contexts: dict[str, CandidateContext],
        config: ScreenerRunConfig,
    ) -> list[ScreenerCandidate]:
        ranked = sorted(
            contexts.values(),
            key=lambda ctx: ctx.wide.score + ctx.deep.score - ctx.risk.score,
            reverse=True,
        )
        candidates: list[ScreenerCandidate] = []
        for index, context in enumerate(ranked[: config.top_n], 1):
            risk_penalty = max(context.risk.score, 0.0)
            total = context.wide.score + context.deep.score - risk_penalty
            risk_balance = max(
                context.risk.components.get("balance_sheet", 0.0), 0.0
            )
            risk_volatility = max(
                context.risk.components.get("volatility", 0.0), 0.0
            )
            components = {
                **context.wide.components,
                **context.deep.components,
                "risk_balance_sheet": -risk_balance,
                "risk_volatility": -risk_volatility,
            }
            breakdown = ScreenerScoreBreakdown(
                total_score=round(total, 4),
                components={key: round(value, 4) for key, value in components.items()},
                top_evidence_ids=self._candidate_evidence_ids(context),
            )
            candidates.append(
                ScreenerCandidate(
                    ticker=context.snapshot.ticker,
                    name=context.universe.name,
                    wide_score=round(context.wide.score, 4),
                    deep_score=round(context.deep.score, 4),
                    risk_score=round(risk_penalty, 4),
                    total_score=round(total, 4),
                    rank=index,
                    score_breakdown=breakdown,
                    rationale=self._build_rationale(context, config),
                )
            )
        return candidates

    @staticmethod
    def _candidate_evidence_ids(context: CandidateContext) -> list[str]:
        evidence_ids = []
        if context.asset and context.asset.market_cap is not None:
            evidence_ids.append(
                f"ev_{context.snapshot.ticker.lower()}_profile"
            )
        evidence_ids.extend(
            [
                f"ev_{context.snapshot.ticker.lower()}_momentum",
                f"ev_{context.snapshot.ticker.lower()}_liquidity",
            ]
        )
        if context.financial:
            capex_growth = _safe_growth_rate(
                context.financial.capex_latest, context.financial.capex_prior
            )
            if capex_growth is not None:
                evidence_ids.append(f"ev_{context.snapshot.ticker.lower()}_capex")
            net_income_growth = _safe_growth_rate(
                context.financial.net_income_latest,
                context.financial.net_income_prior,
            )
            if net_income_growth is not None:
                evidence_ids.append(
                    f"ev_{context.snapshot.ticker.lower()}_earnings"
                )
        return evidence_ids[:3]

    @staticmethod
    def _build_rationale(
        context: CandidateContext, config: ScreenerRunConfig
    ) -> str:
        snapshot = context.snapshot
        rationale = (
            f"{config.frequency.capitalize()} scan shows 20D return "
            f"{snapshot.return_20d:.1%} with liquidity "
            f"{snapshot.avg_volume_20d:,.0f} shares/day."
        )
        if context.financial:
            net_income_growth = _safe_growth_rate(
                context.financial.net_income_latest,
                context.financial.net_income_prior,
            )
            if net_income_growth is not None:
                rationale += f" Net income growth {net_income_growth:.1%}."
        return rationale

    def _build_candidate_details(
        self,
        candidates: Iterable[ScreenerCandidate],
        contexts: dict[str, CandidateContext],
    ) -> dict[str, ScreenerCandidateDetail]:
        details: dict[str, ScreenerCandidateDetail] = {}
        retrieved_at = datetime.now(timezone.utc)
        for candidate in candidates:
            context = contexts.get(candidate.ticker)
            if context is None:
                continue
            evidence_items = self._build_evidence(context, retrieved_at)
            logic_graph = self._build_logic_graph(context, evidence_items)
            details[candidate.ticker] = ScreenerCandidateDetail(
                candidate=candidate,
                evidence=evidence_items,
                logic_graph=logic_graph,
            )
        return details

    def _build_evidence(
        self, context: CandidateContext, retrieved_at: datetime
    ) -> list[ScreenerEvidence]:
        snapshot = context.snapshot
        symbol = snapshot.symbol
        source_url = f"https://finance.yahoo.com/quote/{symbol}"
        evidence: list[ScreenerEvidence] = []

        if context.asset and context.asset.market_cap is not None:
            profile_quote = (
                f"Market cap {context.asset.market_cap:,.0f} as of "
                f"{snapshot.data_end.date().isoformat()}."
            )
            profile_structured = {
                "market_cap": round(context.asset.market_cap, 2),
                "quote_type": context.asset.quote_type or "unknown",
            }
            evidence.append(
                ScreenerEvidence(
                    evidence_id=f"ev_{context.snapshot.ticker.lower()}_profile",
                    type="COMPANY_PROFILE",
                    ticker=context.snapshot.ticker,
                    published_at=snapshot.data_end,
                    retrieved_at=retrieved_at,
                    source_title="Company profile",
                    source_name="Yahoo Finance",
                    source_url=source_url,
                    publisher="Yahoo Finance",
                    reliability_level="secondary",
                    doc_ref={"dataset": "company_profile"},
                    quote=profile_quote,
                    structured=profile_structured,
                    sha256=self._hash_payload(
                        {"quote": profile_quote, "structured": profile_structured}
                    ),
                )
            )

        momentum_quote = (
            f"20D return {snapshot.return_20d:.2%}, "
            f"60D return {snapshot.return_60d:.2%} as of "
            f"{snapshot.data_end.date().isoformat()}."
        )
        momentum_structured = {
            "return_20d": round(snapshot.return_20d, 4),
            "return_60d": round(snapshot.return_60d, 4),
            "as_of": snapshot.data_end.date().isoformat(),
        }
        evidence.append(
            ScreenerEvidence(
                evidence_id=f"ev_{context.snapshot.ticker.lower()}_momentum",
                type="MARKET_DATA",
                ticker=context.snapshot.ticker,
                published_at=snapshot.data_end,
                retrieved_at=retrieved_at,
                source_title="Price history",
                source_name="Yahoo Finance",
                source_url=source_url,
                publisher="Yahoo Finance",
                reliability_level="secondary",
                doc_ref={
                    "dataset": "price_history",
                    "window_days": 60,
                },
                quote=momentum_quote,
                structured=momentum_structured,
                sha256=self._hash_payload(
                    {"quote": momentum_quote, "structured": momentum_structured}
                ),
            )
        )

        dollar_volume = snapshot.last_close * snapshot.avg_volume_20d
        liquidity_quote = (
            f"20D avg volume {snapshot.avg_volume_20d:,.0f} shares, "
            f"dollar volume {dollar_volume:,.0f}."
        )
        liquidity_structured = {
            "avg_volume_20d": round(snapshot.avg_volume_20d, 2),
            "dollar_volume_20d": round(dollar_volume, 2),
            "price": round(snapshot.last_close, 2),
        }
        evidence.append(
            ScreenerEvidence(
                evidence_id=f"ev_{context.snapshot.ticker.lower()}_liquidity",
                type="MARKET_DATA",
                ticker=context.snapshot.ticker,
                published_at=snapshot.data_end,
                retrieved_at=retrieved_at,
                source_title="Liquidity snapshot",
                source_name="Yahoo Finance",
                source_url=source_url,
                publisher="Yahoo Finance",
                reliability_level="secondary",
                doc_ref={
                    "dataset": "price_history",
                    "window_days": 20,
                },
                quote=liquidity_quote,
                structured=liquidity_structured,
                sha256=self._hash_payload(
                    {"quote": liquidity_quote, "structured": liquidity_structured}
                ),
            )
        )

        if context.financial:
            evidence.extend(self._build_financial_evidence(context, retrieved_at))

        return evidence

    def _build_financial_evidence(
        self, context: CandidateContext, retrieved_at: datetime
    ) -> list[ScreenerEvidence]:
        financial = context.financial
        if financial is None:
            return []
        symbol = context.snapshot.symbol
        source_url = f"https://finance.yahoo.com/quote/{symbol}/financials"
        evidence: list[ScreenerEvidence] = []

        capex_growth = _safe_growth_rate(financial.capex_latest, financial.capex_prior)
        net_income_growth = _safe_growth_rate(
            financial.net_income_latest, financial.net_income_prior
        )
        periods_ok = True
        if capex_growth is not None or net_income_growth is not None:
            periods_ok = _has_financial_periods(
                financial.period_prior, financial.period_latest, context.snapshot.ticker
            )
        if capex_growth is not None and periods_ok:
            capex_quote = (
                f"Capex change {capex_growth:.2%} from {financial.period_prior} "
                f"to {financial.period_latest}."
            )
            capex_structured = {
                "capex_latest": financial.capex_latest,
                "capex_prior": financial.capex_prior,
                "capex_growth": round(capex_growth, 4),
                "period_latest": financial.period_latest,
                "period_prior": financial.period_prior,
            }
            evidence.append(
                ScreenerEvidence(
                    evidence_id=f"ev_{context.snapshot.ticker.lower()}_capex",
                    type="FINANCIAL_STATEMENT",
                    ticker=context.snapshot.ticker,
                    published_at=retrieved_at,
                    retrieved_at=retrieved_at,
                    source_title="Cash flow statement",
                    source_name="Yahoo Finance",
                    source_url=source_url,
                    publisher="Yahoo Finance",
                    reliability_level="secondary",
                    doc_ref={
                        "statement": "cashflow",
                        "line_item": "Capital Expenditures",
                    },
                    quote=capex_quote,
                    structured=capex_structured,
                    sha256=self._hash_payload(
                        {"quote": capex_quote, "structured": capex_structured}
                    ),
                )
            )

        if net_income_growth is not None and periods_ok:
            earnings_quote = (
                f"Net income change {net_income_growth:.2%} from "
                f"{financial.period_prior} to {financial.period_latest}."
            )
            earnings_structured = {
                "net_income_latest": financial.net_income_latest,
                "net_income_prior": financial.net_income_prior,
                "net_income_growth": round(net_income_growth, 4),
                "period_latest": financial.period_latest,
                "period_prior": financial.period_prior,
            }
            evidence.append(
                ScreenerEvidence(
                    evidence_id=f"ev_{context.snapshot.ticker.lower()}_earnings",
                    type="FINANCIAL_STATEMENT",
                    ticker=context.snapshot.ticker,
                    published_at=retrieved_at,
                    retrieved_at=retrieved_at,
                    source_title="Income statement",
                    source_name="Yahoo Finance",
                    source_url=source_url,
                    publisher="Yahoo Finance",
                    reliability_level="secondary",
                    doc_ref={
                        "statement": "income_statement",
                        "line_item": "Net Income",
                    },
                    quote=earnings_quote,
                    structured=earnings_structured,
                    sha256=self._hash_payload(
                        {"quote": earnings_quote, "structured": earnings_structured}
                    ),
                )
            )

        return evidence

    def _build_logic_graph(
        self,
        context: CandidateContext,
        evidence_items: list[ScreenerEvidence],
    ) -> LogicGraph:
        nodes: list[LogicGraphNode] = []
        edges: list[LogicGraphEdge] = []
        ticker_key = context.snapshot.ticker.lower()
        node_index = 1

        for evidence in evidence_items:
            evidence_node_id = f"n{node_index}"
            node_index += 1
            nodes.append(
                LogicGraphNode(
                    id=evidence_node_id,
                    type="evidence_ref",
                    name=evidence.source_title or "Evidence",
                    evidence_id=evidence.evidence_id,
                )
            )
            metric_node_id = f"n{node_index}"
            node_index += 1
            metric_name = _evidence_metric_name(evidence.evidence_id)
            metric_value = _evidence_metric_value(evidence)
            nodes.append(
                LogicGraphNode(
                    id=metric_node_id,
                    type="metric",
                    name=metric_name,
                    value=metric_value,
                    unit=_evidence_metric_unit(evidence.evidence_id),
                    as_of=context.snapshot.data_end.date().isoformat(),
                )
            )
            edges.append(
                LogicGraphEdge(
                    source=evidence_node_id,
                    target=metric_node_id,
                    type="supports",
                    weight=0.8,
                )
            )

        claim_id = f"n{node_index}"
        node_index += 1
        claim_score = context.deep.score + context.wide.score - context.risk.score
        nodes.append(
            LogicGraphNode(
                id=claim_id,
                type="claim",
                name=f"Candidate thesis for {ticker_key.upper()}",
                value=round(claim_score, 4),
                unit="score",
                as_of=context.snapshot.data_end.date().isoformat(),
            )
        )

        for node in nodes:
            if node.type == "metric":
                edges.append(
                    LogicGraphEdge(
                        source=node.id,
                        target=claim_id,
                        type="leads_to",
                        weight=0.6,
                    )
                )

        return LogicGraph(nodes=nodes, edges=edges)

    @staticmethod
    def _build_evaluation() -> dict[str, dict[str, float]]:
        return {}

    @staticmethod
    def _build_run_log(
        run_id: str,
        started_at: datetime,
        ended_at: datetime,
        candidates: list[ScreenerCandidate],
        config_payload: dict,
        universe_size: int,
        data_points: int,
    ) -> ScreenerRunLog:
        step_delta = timedelta(seconds=1)
        init_end = started_at + step_delta
        regime_end = init_end + step_delta
        theme_end = regime_end + step_delta
        longlist_end = theme_end + step_delta
        deep_dive_end = longlist_end + step_delta
        shortlist_end = max(deep_dive_end + step_delta, ended_at)
        steps = [
            ScreenerRunLogStep(
                name="A Init",
                status="completed",
                started_at=started_at,
                ended_at=init_end,
                outputs=[
                    f"run_id={run_id}",
                    f"config_hash_input_keys={sorted(config_payload.keys())}",
                    f"universe_size={universe_size}",
                ],
            ),
            ScreenerRunLogStep(
                name="B Regime",
                status="unverified",
                started_at=init_end,
                ended_at=regime_end,
                outputs=[
                    "regime_tag=unverified",
                    "style_bias=unverified",
                    "macro_sources=missing",
                ],
                notes="Macro regime signals require web search or crawler inputs.",
            ),
            ScreenerRunLogStep(
                name="C Themes",
                status="unverified",
                started_at=regime_end,
                ended_at=theme_end,
                outputs=["themes=unverified", "theme_sources=missing"],
                notes="Theme discovery requires news/CapEx evidence sources.",
            ),
            ScreenerRunLogStep(
                name="D Longlist",
                status="completed" if data_points else "skipped",
                started_at=theme_end,
                ended_at=longlist_end,
                outputs=[
                    f"price_snapshots={data_points}",
                    "scoring=wide+deep-risk",
                    f"filters={config_payload.get('universe', {})}",
                ],
                notes="Wide scan uses price + volume + financials for scoring.",
            ),
            ScreenerRunLogStep(
                name="E Deep dive",
                status="completed" if candidates else "skipped",
                started_at=longlist_end,
                ended_at=deep_dive_end,
                outputs=[
                    f"details={len(candidates)}",
                    "evidence_sources=yahoo_finance",
                ],
            ),
            ScreenerRunLogStep(
                name="F Shortlist",
                status="completed",
                started_at=deep_dive_end,
                ended_at=shortlist_end,
                outputs=[f"top_n={len(candidates)}"],
            ),
        ]
        return ScreenerRunLog(
            run_id=run_id,
            run_timestamp_utc=started_at,
            data_cutoff=started_at.date().isoformat(),
            steps=steps,
        )

    @staticmethod
    def _build_summary(
        meta: ScreenerRunMeta, candidates: list[ScreenerCandidate]
    ) -> str:
        lines = [
            f"# Screener Run {meta.run_id}",
            "",
            f"- As of: {meta.as_of_date}",
            f"- Data cutoff: {meta.data_cutoff or 'unknown'}",
            f"- Universe size: {meta.universe_size}",
            f"- Candidates: {len(candidates)}",
            f"- Config hash: {meta.config_hash}",
            "",
            "## Top Candidates",
        ]
        for candidate in candidates[:5]:
            lines.append(
                f"- {candidate.rank}. {candidate.ticker} "
                f"({candidate.total_score})"
            )
        if meta.run_log:
            lines.extend(["", "## Run Log"])
            for step in meta.run_log.steps:
                outputs = "; ".join(step.outputs) if step.outputs else "none"
                notes = f" ({step.notes})" if step.notes else ""
                lines.append(
                    f"- {step.name}: {step.status} | outputs: {outputs}{notes}"
                )
        return "\n".join(lines)
