"""Pydantic schemas for the Quant Screener pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .constants import DEFAULT_TOP_K, DEFAULT_TOP_N


class ScreenerRunConfig(BaseModel):
    """Configuration for a screener run."""

    frequency: Literal["weekly", "monthly", "quarterly"] = Field(
        default="monthly", description="Run frequency"
    )
    style: Optional[str] = Field(default=None, description="Style preference")
    risk: Optional[str] = Field(default=None, description="Risk preference")
    industry: Optional[str] = Field(default=None, description="Industry preference")
    size: Optional[str] = Field(default=None, description="Market cap preference")
    liquidity_min: Optional[float] = Field(
        default=None, description="Minimum daily dollar volume"
    )
    top_k: int = Field(default=DEFAULT_TOP_K, description="Wide scan candidate size")
    top_n: int = Field(default=DEFAULT_TOP_N, description="Final candidate size")


class ScreenerScoreBreakdown(BaseModel):
    """Score breakdown for a candidate."""

    total_score: float = Field(..., description="Total score")
    components: dict[str, float] = Field(..., description="Component scores")
    top_evidence_ids: list[str] = Field(
        default_factory=list, description="Top evidence identifiers"
    )


class ScreenerRunLogStep(BaseModel):
    """Structured log entry for a screener pipeline step."""

    name: str = Field(..., description="Step name")
    status: Literal["completed", "skipped", "unverified"] = Field(
        ..., description="Step execution status"
    )
    started_at: datetime = Field(..., description="Step start time")
    ended_at: datetime = Field(..., description="Step end time")
    outputs: list[str] = Field(default_factory=list, description="Key outputs")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class ScreenerRunLog(BaseModel):
    """Structured log for a screener run."""

    run_id: str = Field(..., description="Run identifier")
    run_timestamp_utc: datetime = Field(..., description="Run timestamp (UTC)")
    data_cutoff: str = Field(..., description="Latest data cutoff date")
    steps: list[ScreenerRunLogStep] = Field(
        default_factory=list, description="Pipeline steps"
    )


class ScreenerCandidate(BaseModel):
    """Candidate entry from the screener."""

    ticker: str = Field(..., description="Ticker symbol")
    name: str = Field(..., description="Company name")
    wide_score: float = Field(..., description="Wide scan score")
    deep_score: float = Field(..., description="Deep dive score")
    risk_score: float = Field(..., description="Risk penalty score")
    total_score: float = Field(..., description="Total score")
    rank: int = Field(..., description="Ranking number")
    score_breakdown: ScreenerScoreBreakdown = Field(
        ..., description="Score breakdown"
    )
    rationale: str = Field(..., description="Short rationale")


class ScreenerRunMeta(BaseModel):
    """Metadata for a screener run."""

    run_id: str = Field(..., description="Run identifier")
    as_of_date: str = Field(..., description="As-of date")
    run_timestamp_utc: Optional[datetime] = Field(
        default=None, description="Run timestamp (UTC)"
    )
    data_cutoff: Optional[str] = Field(
        default=None, description="Latest data cutoff date"
    )
    started_at: datetime = Field(..., description="Run start time")
    ended_at: datetime = Field(..., description="Run end time")
    config_hash: str = Field(..., description="Config hash")
    code_git_sha: str = Field(..., description="Git commit SHA")
    data_snapshot_hash: str = Field(..., description="Input data hash")
    universe_size: int = Field(..., description="Universe size")
    status: str = Field(..., description="Run status")
    config: ScreenerRunConfig = Field(..., description="Run configuration")
    run_log: Optional[ScreenerRunLog] = Field(
        default=None, description="Run log with pipeline steps"
    )


class ScreenerRunSummary(BaseModel):
    """Summary entry for a screener run list."""

    run_id: str = Field(..., description="Run identifier")
    as_of_date: str = Field(..., description="As-of date")
    status: str = Field(..., description="Run status")
    started_at: datetime = Field(..., description="Run start time")
    ended_at: datetime = Field(..., description="Run end time")
    candidate_count: int = Field(..., description="Number of candidates")
    frequency: str = Field(..., description="Run frequency")


class ScreenerEvidence(BaseModel):
    """Evidence item for a candidate."""

    evidence_id: str = Field(..., description="Evidence identifier")
    type: str = Field(..., description="Evidence type")
    ticker: str = Field(..., description="Ticker symbol")
    published_at: datetime = Field(..., description="Published timestamp")
    retrieved_at: datetime = Field(..., description="Retrieved timestamp")
    source_title: Optional[str] = Field(default=None, description="Source title")
    source_name: str = Field(..., description="Source name")
    source_url: str = Field(..., description="Source URL")
    publisher: Optional[str] = Field(default=None, description="Publisher name")
    reliability_level: Optional[str] = Field(
        default=None, description="Reliability level"
    )
    doc_ref: dict[str, str | int | list[int]] = Field(
        ..., description="Document reference"
    )
    quote: str = Field(..., description="Quoted excerpt")
    structured: dict[str, str | int | float] = Field(
        ..., description="Structured evidence"
    )
    sha256: str = Field(..., description="SHA256 hash")


class LogicGraphNode(BaseModel):
    """Logic graph node."""

    id: str = Field(..., description="Node identifier")
    type: str = Field(..., description="Node type")
    name: str = Field(..., description="Node name")
    value: Optional[float] = Field(default=None, description="Node value")
    unit: Optional[str] = Field(default=None, description="Unit")
    as_of: Optional[str] = Field(default=None, description="As-of date")
    evidence_id: Optional[str] = Field(default=None, description="Evidence ID")


class LogicGraphEdge(BaseModel):
    """Logic graph edge."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    type: str = Field(..., description="Edge type")
    weight: float = Field(..., description="Edge weight")


class LogicGraph(BaseModel):
    """Logic graph data structure."""

    nodes: list[LogicGraphNode] = Field(..., description="Graph nodes")
    edges: list[LogicGraphEdge] = Field(..., description="Graph edges")


class ScreenerCandidateDetail(BaseModel):
    """Detailed candidate data with evidence and logic graph."""

    candidate: ScreenerCandidate = Field(..., description="Candidate overview")
    evidence: list[ScreenerEvidence] = Field(
        default_factory=list, description="Evidence items"
    )
    logic_graph: LogicGraph = Field(..., description="Logic graph")


class ScreenerRunResult(BaseModel):
    """Full result for a screener run."""

    meta: ScreenerRunMeta = Field(..., description="Run metadata")
    candidates: list[ScreenerCandidate] = Field(
        default_factory=list, description="Candidates"
    )
    candidate_details: dict[str, ScreenerCandidateDetail] = Field(
        default_factory=dict, description="Candidate details"
    )
    evaluation: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Evaluation metrics"
    )
