"""Quant Screener pipeline scaffolding."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from typing import Iterable

from loguru import logger

from valuecell.utils.uuid import generate_uuid

from .config import load_screener_config
from .constants import DEFAULT_UNIVERSE_SIZE
from .schemas import (
    LogicGraph,
    LogicGraphEdge,
    LogicGraphNode,
    ScreenerCandidate,
    ScreenerCandidateDetail,
    ScreenerEvidence,
    ScreenerRunConfig,
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


class ScreenerPipeline:
    """Wide -> Deep pipeline that writes run artifacts locally."""

    def run(self, config: ScreenerRunConfig) -> ScreenerRunResult:
        """Execute the pipeline and persist outputs."""
        started_at = datetime.now(timezone.utc)
        run_id = generate_uuid("run")
        config_payload = self._config_payload(config)
        config_hash = self._hash_payload(config_payload)
        candidates = self._build_candidates(config)
        data_snapshot_hash = self._hash_payload(
            [candidate.model_dump() for candidate in candidates]
        )
        meta = ScreenerRunMeta(
            run_id=run_id,
            as_of_date=started_at.date().isoformat(),
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
            config_hash=config_hash,
            code_git_sha=self._get_git_sha(),
            data_snapshot_hash=data_snapshot_hash,
            universe_size=DEFAULT_UNIVERSE_SIZE,
            status="completed",
            config=config,
        )
        candidate_details = self._build_candidate_details(candidates)
        evaluation = self._build_evaluation()

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

    def _build_candidates(self, config: ScreenerRunConfig) -> list[ScreenerCandidate]:
        base_candidates = [
            ("NVDA", "NVIDIA Corporation", 28.4, 31.2, 4.5),
            ("ASML", "ASML Holding N.V.", 26.1, 29.4, 3.8),
            ("PLTR", "Palantir Technologies", 23.7, 27.1, 6.0),
            ("SMCI", "Super Micro Computer", 25.3, 24.8, 5.2),
            ("ELF", "e.l.f. Beauty", 22.9, 21.5, 3.6),
        ]
        candidates: list[ScreenerCandidate] = []
        for index, (ticker, name, wide, deep, risk) in enumerate(base_candidates, 1):
            total = wide + deep - risk
            breakdown = ScreenerScoreBreakdown(
                total_score=round(total, 2),
                components={
                    "fundamental_momentum": round(wide * 0.45, 2),
                    "valuation": round(wide * 0.25, 2),
                    "market_momentum": round(wide * 0.30, 2),
                    "supply_chain": round(deep * 0.35, 2),
                    "inflection": round(deep * 0.40, 2),
                    "capex_cycle": round(deep * 0.25, 2),
                    "risk_penalty": round(-risk, 2),
                },
                top_evidence_ids=[f"ev_{ticker.lower()}_1", f"ev_{ticker.lower()}_2"],
            )
            candidates.append(
                ScreenerCandidate(
                    ticker=ticker,
                    name=name,
                    wide_score=round(wide, 2),
                    deep_score=round(deep, 2),
                    risk_score=round(risk, 2),
                    total_score=round(total, 2),
                    rank=index,
                    score_breakdown=breakdown,
                    rationale=(
                        "Wide scan momentum aligned with evidence-backed inflection "
                        f"signals for {config.frequency} cadence."
                    ),
                )
            )
        return candidates[: config.top_n]

    def _build_candidate_details(
        self, candidates: Iterable[ScreenerCandidate]
    ) -> dict[str, ScreenerCandidateDetail]:
        details: dict[str, ScreenerCandidateDetail] = {}
        retrieved_at = datetime.now(timezone.utc)
        for candidate in candidates:
            evidence_items = self._build_evidence(candidate, retrieved_at)
            logic_graph = self._build_logic_graph(candidate)
            details[candidate.ticker] = ScreenerCandidateDetail(
                candidate=candidate,
                evidence=evidence_items,
                logic_graph=logic_graph,
            )
        return details

    def _build_evidence(
        self, candidate: ScreenerCandidate, retrieved_at: datetime
    ) -> list[ScreenerEvidence]:
        published_at = retrieved_at
        return [
            ScreenerEvidence(
                evidence_id=f"ev_{candidate.ticker.lower()}_1",
                type="SEC_FILING",
                ticker=candidate.ticker,
                published_at=published_at,
                retrieved_at=retrieved_at,
                source_name="SEC EDGAR",
                source_url="https://www.sec.gov/",
                doc_ref={
                    "accession": "0000000000-00-000000",
                    "form": "10-Q",
                    "section": "MD&A",
                    "chunk_id": "chunk_0001",
                    "span": [120, 310],
                },
                quote=(
                    "Management highlighted backlog growth and margin stabilization "
                    "as demand normalized."
                ),
                structured={
                    "metric": "Backlog",
                    "value": 123456789,
                    "unit": "USD",
                },
                sha256="placeholder_sha_1",
            ),
            ScreenerEvidence(
                evidence_id=f"ev_{candidate.ticker.lower()}_2",
                type="NEWS",
                ticker=candidate.ticker,
                published_at=published_at,
                retrieved_at=retrieved_at,
                source_name="Company Newswire",
                source_url="https://example.com/news",
                doc_ref={
                    "accession": "newswire",
                    "form": "NEWS",
                    "section": "Press Release",
                    "chunk_id": "chunk_0002",
                    "span": [12, 88],
                },
                quote=(
                    "The company reiterated its capacity expansion timeline and "
                    "raised capex guidance."
                ),
                structured={
                    "metric": "Capex Guidance",
                    "value": 4.2,
                    "unit": "B USD",
                },
                sha256="placeholder_sha_2",
            ),
        ]

    @staticmethod
    def _build_logic_graph(candidate: ScreenerCandidate) -> LogicGraph:
        nodes = [
            LogicGraphNode(
                id="n1",
                type="metric",
                name="Backlog_YoY_Accel",
                value=2.1,
                unit="pp",
                as_of="2025Q3",
            ),
            LogicGraphNode(
                id="n2",
                type="evidence_ref",
                name="Evidence",
                evidence_id=f"ev_{candidate.ticker.lower()}_1",
            ),
            LogicGraphNode(
                id="n3",
                type="claim",
                name="Demand inflection likely",
                value=0.74,
                unit="confidence",
            ),
        ]
        edges = [
            LogicGraphEdge(source="n2", target="n1", type="supports", weight=0.9),
            LogicGraphEdge(source="n1", target="n3", type="leads_to", weight=0.7),
        ]
        return LogicGraph(nodes=nodes, edges=edges)

    @staticmethod
    def _build_evaluation() -> dict[str, dict[str, float]]:
        return {
            "1w": {"hit_rate": 0.55, "avg_excess_return": 0.012},
            "1m": {"hit_rate": 0.62, "avg_excess_return": 0.043},
            "3m": {"hit_rate": 0.68, "avg_excess_return": 0.091},
        }

    @staticmethod
    def _build_summary(
        meta: ScreenerRunMeta, candidates: list[ScreenerCandidate]
    ) -> str:
        lines = [
            f"# Screener Run {meta.run_id}",
            "",
            f"- As of: {meta.as_of_date}",
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
        return "\n".join(lines)
