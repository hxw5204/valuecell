"""Quant Screener API router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path

from valuecell.server.api.schemas.base import SuccessResponse
from valuecell.server.api.schemas.screener import (
    ScreenerCandidateDetailData,
    ScreenerCandidateListData,
    ScreenerExportData,
    ScreenerRunData,
    ScreenerRunDeleteData,
    ScreenerRunDetailData,
    ScreenerRunListData,
    ScreenerRunRequest,
)
from valuecell.server.services.screener_service import ScreenerService
from valuecell.screener.schemas import ScreenerRunConfig


def create_screener_router() -> APIRouter:
    """Create screener router."""
    router = APIRouter(
        prefix="/screener",
        tags=["screener"],
        responses={404: {"description": "Not found"}},
    )

    @router.post(
        "/run",
        response_model=SuccessResponse[ScreenerRunData],
        summary="Run the quant screener",
        description="Start a screener run and persist local artifacts.",
    )
    async def run_screener(request: ScreenerRunRequest) -> SuccessResponse[ScreenerRunData]:
        config_payload = request.model_dump(exclude_none=True)
        config = ScreenerRunConfig(**config_payload)
        result = await ScreenerService.run(config)
        response_data = ScreenerRunData(
            run=result.meta,
            candidate_count=len(result.candidates),
            top_candidates=result.candidates[:10],
        )
        return SuccessResponse.create(data=response_data, msg="Screener run completed")

    @router.get(
        "/runs",
        response_model=SuccessResponse[ScreenerRunListData],
        summary="List screener runs",
        description="Get all screener run summaries.",
    )
    async def list_screener_runs() -> SuccessResponse[ScreenerRunListData]:
        runs = ScreenerService.list_runs()
        return SuccessResponse.create(data=ScreenerRunListData(runs=runs))

    @router.get(
        "/runs/{run_id}",
        response_model=SuccessResponse[ScreenerRunDetailData],
        summary="Get screener run detail",
        description="Get metadata and evaluation summary for a run.",
    )
    async def get_screener_run(
        run_id: str = Path(..., description="Run identifier"),
    ) -> SuccessResponse[ScreenerRunDetailData]:
        meta = ScreenerService.get_run_meta(run_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Run not found")
        return SuccessResponse.create(
            data=ScreenerRunDetailData(run=meta, evaluation={}),
        )

    @router.get(
        "/runs/{run_id}/candidates",
        response_model=SuccessResponse[ScreenerCandidateListData],
        summary="Get run candidates",
        description="Get candidates for a run.",
    )
    async def get_run_candidates(
        run_id: str = Path(..., description="Run identifier"),
    ) -> SuccessResponse[ScreenerCandidateListData]:
        candidates = ScreenerService.get_candidates(run_id)
        return SuccessResponse.create(
            data=ScreenerCandidateListData(run_id=run_id, candidates=candidates)
        )

    @router.get(
        "/runs/{run_id}/candidates/{ticker}",
        response_model=SuccessResponse[ScreenerCandidateDetailData],
        summary="Get candidate detail",
        description="Get evidence and logic graph for a ticker.",
    )
    async def get_candidate_detail(
        run_id: str = Path(..., description="Run identifier"),
        ticker: str = Path(..., description="Ticker symbol"),
    ) -> SuccessResponse[ScreenerCandidateDetailData]:
        detail = ScreenerService.get_candidate_detail(run_id, ticker)
        if not detail:
            raise HTTPException(status_code=404, detail="Candidate not found")
        return SuccessResponse.create(
            data=ScreenerCandidateDetailData(run_id=run_id, detail=detail)
        )

    @router.get(
        "/runs/{run_id}/export",
        response_model=SuccessResponse[ScreenerExportData],
        summary="Export candidates",
        description="Export candidate list as CSV content.",
    )
    async def export_candidates(
        run_id: str = Path(..., description="Run identifier"),
    ) -> SuccessResponse[ScreenerExportData]:
        csv_content = ScreenerService.get_export_csv(run_id)
        if csv_content is None:
            raise HTTPException(status_code=404, detail="Export not found")
        return SuccessResponse.create(
            data=ScreenerExportData(
                run_id=run_id,
                filename=f"{run_id}_candidates.csv",
                content_type="text/csv",
                content=csv_content,
            )
        )

    @router.delete(
        "/runs/{run_id}",
        response_model=SuccessResponse[ScreenerRunDeleteData],
        summary="Delete screener run",
        description="Delete a screener run and its stored artifacts.",
    )
    async def delete_screener_run(
        run_id: str = Path(..., description="Run identifier"),
    ) -> SuccessResponse[ScreenerRunDeleteData]:
        deleted = ScreenerService.delete_run(run_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Run not found")
        return SuccessResponse.create(data=ScreenerRunDeleteData(run_id=run_id))

    return router
