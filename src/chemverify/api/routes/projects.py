from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ...config import Settings
from ...models import ProjectPaperSessionRecord, ProjectSearchThreadRecord
from ...project_store import ProjectStore
from ...utils import now_iso
from ..schemas import (
    CreateProjectRequest,
    ProjectDetailResponse,
    ProjectListResponse,
    ProjectMutationResponse,
    ProjectSummaryResponse,
    UpdateProjectRequest,
    UpsertProjectPaperSessionRequest,
    UpsertProjectThreadRequest,
)

router = APIRouter()


def _project_store(request: Request) -> ProjectStore:
    root_dir = request.app.state.service_manager.root_dir
    settings = Settings.from_env(root_dir=root_dir)
    return ProjectStore(settings)


def _build_project_summary(store: ProjectStore, project) -> ProjectSummaryResponse:
    return ProjectSummaryResponse(
        project_id=project.project_id,
        title=project.title,
        created_at=project.created_at,
        updated_at=project.updated_at,
        selected_corpora=project.selected_corpora or [],
        search_thread_count=store.thread_count(project.project_id),
        paper_session_count=store.paper_session_count(project.project_id),
    )


@router.get("/projects", response_model=ProjectListResponse)
def list_projects(request: Request) -> ProjectListResponse:
    store = _project_store(request)
    projects = [_build_project_summary(store, project) for project in store.list_projects()]
    return ProjectListResponse(projects=projects)


@router.post("/projects", response_model=ProjectSummaryResponse)
def create_project(request: Request, payload: CreateProjectRequest) -> ProjectSummaryResponse:
    store = _project_store(request)
    try:
        project = store.create_project(payload.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _build_project_summary(store, project)


@router.get("/projects/{project_id}", response_model=ProjectDetailResponse)
def get_project(request: Request, project_id: str) -> ProjectDetailResponse:
    store = _project_store(request)
    try:
        project = store.require_project(project_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    return ProjectDetailResponse(
        project=_build_project_summary(store, project),
        threads=store.list_threads(project_id),
        paper_sessions=store.list_paper_sessions(project_id),
    )


@router.patch("/projects/{project_id}", response_model=ProjectSummaryResponse)
def update_project(request: Request, project_id: str, payload: UpdateProjectRequest) -> ProjectSummaryResponse:
    store = _project_store(request)
    try:
        updates: dict[str, object] = {}
        if "title" in payload.model_fields_set:
            updates["title"] = payload.title
        if "selected_corpora" in payload.model_fields_set:
            updates["selected_corpora"] = payload.selected_corpora or []
        if not updates:
            project = store.require_project(project_id)
        else:
            project = store.update_project(project_id, **updates)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404 if isinstance(exc, FileNotFoundError) else 400, detail=str(exc)) from exc
    return _build_project_summary(store, project)


@router.delete("/projects/{project_id}", response_model=ProjectMutationResponse)
def delete_project(request: Request, project_id: str) -> ProjectMutationResponse:
    store = _project_store(request)
    try:
        store.delete_project(project_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    return ProjectMutationResponse(project_id=project_id, status="deleted")


@router.post("/projects/{project_id}/clear", response_model=ProjectMutationResponse)
def clear_project(request: Request, project_id: str) -> ProjectMutationResponse:
    store = _project_store(request)
    try:
        store.clear_project(project_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    return ProjectMutationResponse(project_id=project_id, status="cleared")


@router.get("/projects/{project_id}/threads/{thread_id}", response_model=ProjectSearchThreadRecord)
def get_project_thread(request: Request, project_id: str, thread_id: str) -> ProjectSearchThreadRecord:
    store = _project_store(request)
    try:
        thread = store.get_thread(project_id, thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    if thread is None:
        raise HTTPException(status_code=404, detail="Project thread not found")
    return thread


@router.put("/projects/{project_id}/threads/{thread_id}", response_model=ProjectSearchThreadRecord)
def upsert_project_thread(
    request: Request,
    project_id: str,
    thread_id: str,
    payload: UpsertProjectThreadRequest,
) -> ProjectSearchThreadRecord:
    store = _project_store(request)
    existing = None
    try:
        store.require_project(project_id)
        existing = store.get_thread(project_id, thread_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    record = ProjectSearchThreadRecord(
        project_id=project_id,
        thread_id=thread_id,
        query=payload.query,
        trace_id=payload.trace_id,
        created_at=existing.created_at if existing is not None else now_iso(),
        updated_at=now_iso(),
        result_counts=payload.result_counts,
        paper_ids=payload.paper_ids,
        workspace_scope=payload.workspace_scope,
        query_scope=payload.query_scope,
        effective_scope=payload.effective_scope,
    )
    return store.save_thread(record)


@router.get("/projects/{project_id}/papers/{paper_id}/session", response_model=ProjectPaperSessionRecord)
def get_project_paper_session(request: Request, project_id: str, paper_id: str) -> ProjectPaperSessionRecord:
    store = _project_store(request)
    try:
        session = store.get_paper_session(project_id, paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    if session is None:
        raise HTTPException(status_code=404, detail="Project paper session not found")
    return session


@router.put("/projects/{project_id}/papers/{paper_id}/session", response_model=ProjectPaperSessionRecord)
def upsert_project_paper_session(
    request: Request,
    project_id: str,
    paper_id: str,
    payload: UpsertProjectPaperSessionRequest,
) -> ProjectPaperSessionRecord:
    store = _project_store(request)
    existing = None
    try:
        store.require_project(project_id)
        existing = store.get_paper_session(project_id, paper_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    record = ProjectPaperSessionRecord(
        project_id=project_id,
        paper_id=paper_id,
        paper_title=payload.paper_title,
        source_thread_id=payload.source_thread_id,
        created_at=existing.created_at if existing is not None else now_iso(),
        updated_at=now_iso(),
        chat_history=payload.chat_history,
        last_active_evidence_id=payload.last_active_evidence_id,
    )
    return store.save_paper_session(record)
