"""Service layer for the Quant Screener API."""

from __future__ import annotations

import asyncio
from typing import Optional

from loguru import logger

from valuecell.screener.pipeline import ScreenerPipeline
from valuecell.screener.schemas import (
    ScreenerCandidate,
    ScreenerCandidateDetail,
    ScreenerRunConfig,
    ScreenerRunMeta,
    ScreenerRunResult,
    ScreenerRunSummary,
)
from valuecell.screener.storage import (
    list_runs,
    load_candidate_detail,
    load_candidates,
    load_export_csv,
    load_run_meta,
)


class ScreenerService:
    """Screener orchestration and data access."""

    @staticmethod
    async def run(config: ScreenerRunConfig) -> ScreenerRunResult:
        pipeline = ScreenerPipeline()
        logger.info(
            "Starting screener run with frequency {frequency}",
            frequency=config.frequency,
        )
        return await asyncio.to_thread(pipeline.run, config)

    @staticmethod
    def list_runs() -> list[ScreenerRunSummary]:
        return list_runs()

    @staticmethod
    def get_run_meta(run_id: str) -> Optional[ScreenerRunMeta]:
        return load_run_meta(run_id)

    @staticmethod
    def get_candidates(run_id: str) -> list[ScreenerCandidate]:
        return load_candidates(run_id)

    @staticmethod
    def get_candidate_detail(
        run_id: str, ticker: str
    ) -> Optional[ScreenerCandidateDetail]:
        return load_candidate_detail(run_id, ticker)

    @staticmethod
    def get_export_csv(run_id: str) -> Optional[str]:
        return load_export_csv(run_id)
