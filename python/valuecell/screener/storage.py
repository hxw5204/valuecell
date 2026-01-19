"""Local storage helpers for Quant Screener runs."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger

from valuecell.utils.env import ensure_system_env_dir

from .constants import (
    DATA_DIR_NAME,
    EVIDENCE_DIR_NAME,
    LOGIC_GRAPH_DIR_NAME,
    REPORTS_DIR_NAME,
    RUNS_DIR_NAME,
)
from .schemas import (
    LogicGraph,
    ScreenerCandidate,
    ScreenerCandidateDetail,
    ScreenerEvidence,
    ScreenerRunMeta,
    ScreenerRunSummary,
)


def get_runs_root() -> Path:
    """Return the root directory for screener runs."""
    base_dir = ensure_system_env_dir()
    runs_root = base_dir / DATA_DIR_NAME / RUNS_DIR_NAME
    runs_root.mkdir(parents=True, exist_ok=True)
    return runs_root


def get_run_dir(run_id: str) -> Path:
    """Return the directory for a specific run."""
    run_dir = get_runs_root() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_meta(run_meta: ScreenerRunMeta) -> None:
    """Write run metadata to meta.json."""
    run_dir = get_run_dir(run_meta.run_id)
    meta_path = run_dir / "meta.json"
    meta_path.write_text(run_meta.model_dump_json(indent=2), encoding="utf-8")


def write_candidates(run_id: str, candidates: Iterable[ScreenerCandidate]) -> None:
    """Write candidates to CSV and JSONL."""
    run_dir = get_run_dir(run_id)
    csv_path = run_dir / "candidates.csv"
    jsonl_path = run_dir / "candidates.jsonl"

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "rank",
                "ticker",
                "name",
                "total_score",
                "wide_score",
                "deep_score",
                "risk_score",
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.rank,
                    candidate.ticker,
                    candidate.name,
                    candidate.total_score,
                    candidate.wide_score,
                    candidate.deep_score,
                    candidate.risk_score,
                ]
            )

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for candidate in candidates:
            jsonl_file.write(candidate.model_dump_json())
            jsonl_file.write("\n")


def write_candidate_detail(
    run_id: str, ticker: str, detail: ScreenerCandidateDetail
) -> None:
    """Write evidence and logic graph files for a candidate."""
    run_dir = get_run_dir(run_id)
    evidence_dir = run_dir / EVIDENCE_DIR_NAME
    logic_dir = run_dir / LOGIC_GRAPH_DIR_NAME
    evidence_dir.mkdir(parents=True, exist_ok=True)
    logic_dir.mkdir(parents=True, exist_ok=True)

    evidence_path = evidence_dir / f"{ticker}.json"
    logic_path = logic_dir / f"{ticker}.json"

    evidence_payload = [item.model_dump(mode="json") for item in detail.evidence]
    evidence_path.write_text(
        json.dumps(evidence_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logic_path.write_text(detail.logic_graph.model_dump_json(indent=2), encoding="utf-8")


def write_report(run_id: str, summary: str) -> None:
    """Write summary report markdown."""
    run_dir = get_run_dir(run_id)
    reports_dir = run_dir / REPORTS_DIR_NAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")


def write_evaluation(run_id: str, evaluation: dict[str, dict[str, float]]) -> None:
    """Write evaluation JSON files."""
    run_dir = get_run_dir(run_id)
    reports_dir = run_dir / REPORTS_DIR_NAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    for horizon, metrics in evaluation.items():
        path = reports_dir / f"evaluation_{horizon}.json"
        path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read JSON file {path}: {error}", path=path, error=exc)
        return None


def load_run_meta(run_id: str) -> Optional[ScreenerRunMeta]:
    """Load run metadata from disk."""
    meta_path = get_run_dir(run_id) / "meta.json"
    if not meta_path.exists():
        return None
    data = _read_json(meta_path)
    if not data:
        return None
    return ScreenerRunMeta.model_validate(data)


def list_runs() -> list[ScreenerRunSummary]:
    """List all available runs."""
    runs_root = get_runs_root()
    summaries: list[ScreenerRunSummary] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        data = _read_json(meta_path)
        if not data:
            continue
        try:
            meta = ScreenerRunMeta.model_validate(data)
            candidate_count = _load_candidate_count(run_dir)
            summaries.append(
                ScreenerRunSummary(
                    run_id=meta.run_id,
                    as_of_date=meta.as_of_date,
                    status=meta.status,
                    started_at=meta.started_at,
                    ended_at=meta.ended_at,
                    candidate_count=candidate_count,
                    frequency=meta.config.frequency,
                )
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse run metadata for {path}: {error}",
                path=meta_path,
                error=exc,
            )
    summaries.sort(key=lambda item: item.started_at, reverse=True)
    return summaries


def _load_candidate_count(run_dir: Path) -> int:
    jsonl_path = run_dir / "candidates.jsonl"
    if not jsonl_path.exists():
        return 0
    try:
        return sum(1 for _ in jsonl_path.open("r", encoding="utf-8"))
    except Exception:
        return 0


def load_candidates(run_id: str) -> list[ScreenerCandidate]:
    """Load candidates for a run."""
    jsonl_path = get_run_dir(run_id) / "candidates.jsonl"
    if not jsonl_path.exists():
        return []
    candidates: list[ScreenerCandidate] = []
    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                candidates.append(ScreenerCandidate.model_validate(data))
            except Exception as exc:
                logger.warning(
                    "Failed to parse candidate line for run {run_id}: {error}",
                    run_id=run_id,
                    error=exc,
                )
    return candidates


def load_candidate_detail(
    run_id: str, ticker: str
) -> Optional[ScreenerCandidateDetail]:
    """Load candidate detail for a ticker."""
    run_dir = get_run_dir(run_id)
    candidate = _find_candidate(run_id, ticker)
    if not candidate:
        return None
    evidence_path = run_dir / EVIDENCE_DIR_NAME / f"{ticker}.json"
    logic_path = run_dir / LOGIC_GRAPH_DIR_NAME / f"{ticker}.json"
    evidence_items: list[ScreenerEvidence] = []
    logic_graph = LogicGraph(nodes=[], edges=[])

    if evidence_path.exists():
        evidence_data = _read_json(evidence_path)
        if evidence_data:
            evidence_items = [
                ScreenerEvidence.model_validate(item) for item in evidence_data
            ]
    if logic_path.exists():
        logic_data = _read_json(logic_path)
        if logic_data:
            logic_graph = LogicGraph.model_validate(logic_data)

    return ScreenerCandidateDetail(
        candidate=candidate,
        evidence=evidence_items,
        logic_graph=logic_graph,
    )


def _find_candidate(run_id: str, ticker: str) -> Optional[ScreenerCandidate]:
    for candidate in load_candidates(run_id):
        if candidate.ticker == ticker:
            return candidate
    return None


def load_export_csv(run_id: str) -> Optional[str]:
    """Load candidate CSV for export."""
    csv_path = get_run_dir(run_id) / "candidates.csv"
    if not csv_path.exists():
        return None
    return csv_path.read_text(encoding="utf-8")


def delete_run(run_id: str) -> bool:
    """Delete all artifacts for a screener run."""
    run_dir = get_runs_root() / run_id
    if not run_dir.exists():
        return False
    if not run_dir.is_dir():
        logger.warning("Run path {path} is not a directory", path=run_dir)
        return False
    shutil.rmtree(run_dir)
    logger.info("Deleted screener run {run_id}", run_id=run_id)
    return True
