from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ...config import Settings
from ...corpus_catalog import load_search_current_catalog
from ...project_store import ProjectStore
from ..schemas import CreateSearchJobRequest, SearchJobResultResponse, SearchJobStatusResponse

router = APIRouter()


@router.post("/search/jobs", response_model=SearchJobStatusResponse)
def create_search_job(request: Request, payload: CreateSearchJobRequest) -> SearchJobStatusResponse:
    service_manager = request.app.state.service_manager
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")
    root_dir = service_manager.root_dir
    settings = Settings.from_env(root_dir=root_dir)
    store = ProjectStore(settings)
    try:
        project = store.require_project(payload.project_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Workspace not found") from exc

    workspace_scope = project.selected_corpora or []
    if not workspace_scope:
        raise HTTPException(status_code=409, detail="The current workspace has no corpora selected.")

    available_corpora = set(load_search_current_catalog(settings).corpus_keys)
    unavailable = [corpus for corpus in workspace_scope if corpus not in available_corpora]
    if unavailable:
        raise HTTPException(
            status_code=409,
            detail="The current workspace contains unavailable corpora. Update the workspace scope before searching.",
        )

    with service_manager.acquire_services() as services:
        return services.jobs.submit(
            query=query,
            project_id=payload.project_id,
            workspace_scope=workspace_scope,
            top_k=payload.top_k,
            display_k=payload.display_k,
        )


@router.get("/search/jobs/{job_id}", response_model=SearchJobStatusResponse)
def get_search_job(request: Request, job_id: str) -> SearchJobStatusResponse:
    service_manager = request.app.state.service_manager
    with service_manager.acquire_job_services(job_id) as services:
        status = services.jobs.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Search job not found")
    return status


@router.get("/search/jobs/{job_id}/result", response_model=SearchJobResultResponse)
def get_search_job_result(request: Request, job_id: str) -> SearchJobResultResponse:
    service_manager = request.app.state.service_manager
    with service_manager.acquire_job_services(job_id) as services:
        status = services.jobs.get_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Search job not found")
        if status.status == "failed":
            raise HTTPException(status_code=409, detail=status.error or "Search job failed")
        if status.status != "completed":
            raise HTTPException(status_code=409, detail="Search job is not completed yet")
        result = services.jobs.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=500, detail="Search result is missing")
    return result
