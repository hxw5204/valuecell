"""API schemas for Quant Screener endpoints."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

import valuecell.screener.schemas as screener_schemas


class ScreenerRunRequest(BaseModel):
    """Request payload for starting a screener run."""

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
    top_k: Optional[int] = Field(default=None, description="Wide scan size")
    top_n: Optional[int] = Field(default=None, description="Final candidate size")


class ScreenerRunData(BaseModel):
    """Response payload for a screener run."""

    run: screener_schemas.ScreenerRunMeta = Field(..., description="Run metadata")
    candidate_count: int = Field(..., description="Candidate count")
    top_candidates: list[screener_schemas.ScreenerCandidate] = Field(
        default_factory=list, description="Top candidates"
    )


class ScreenerRunListData(BaseModel):
    """Response payload for screener run list."""

    runs: list[screener_schemas.ScreenerRunSummary] = Field(
        default_factory=list, description="Run summaries"
    )


class ScreenerRunDetailData(BaseModel):
    """Response payload for a specific run detail."""

    run: screener_schemas.ScreenerRunMeta = Field(..., description="Run metadata")
    evaluation: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Evaluation metrics"
    )


class ScreenerCandidateListData(BaseModel):
    """Response payload for candidate list."""

    run_id: str = Field(..., description="Run identifier")
    candidates: list[screener_schemas.ScreenerCandidate] = Field(
        default_factory=list, description="Candidates"
    )


class ScreenerCandidateDetailData(BaseModel):
    """Response payload for candidate detail."""

    run_id: str = Field(..., description="Run identifier")
    detail: screener_schemas.ScreenerCandidateDetail = Field(
        ..., description="Candidate detail"
    )


class ScreenerExportData(BaseModel):
    """Response payload for export endpoint."""

    run_id: str = Field(..., description="Run identifier")
    filename: str = Field(..., description="Export filename")
    content_type: str = Field(..., description="Content type")
    content: str = Field(..., description="CSV content")


class ScreenerRunDeleteData(BaseModel):
    """Response payload for deleting a run."""

    run_id: str = Field(..., description="Run identifier")
