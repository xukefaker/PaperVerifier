from __future__ import annotations

from typing import Literal
from typing import Any

from pydantic import BaseModel, Field

from ..models import (
    PaperRecord,
    PaperResult,
    ProjectChatMessage,
    ProjectPaperSessionRecord,
    ProjectRecord,
    ProjectSearchThreadRecord,
    SearchTrace,
)


class HealthJobSummary(BaseModel):
    total_jobs: int = 0
    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    data_dir: str
    index_state: dict[str, Any] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    jobs: HealthJobSummary = Field(default_factory=HealthJobSummary)


class CreateSearchJobRequest(BaseModel):
    query: str
    top_k: int = 10
    display_k: int = 10


class SearchJobProgressResponse(BaseModel):
    stage_index: int = 0
    stage_total: int = 0
    stage_progress: float = 0.0
    overall_progress: float = 0.0
    completed_items: int | None = None
    total_items: int | None = None


class SearchJobStatusResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    message: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_ms: float = 0.0
    trace_id: str | None = None
    error: str | None = None
    progress: SearchJobProgressResponse | None = None


class SearchResultCounts(BaseModel):
    satisfied: int = 0
    partial: int = 0
    rejected: int = 0


class SearchJobResultResponse(BaseModel):
    job_id: str
    query: str
    trace_id: str
    mode: str
    counts: SearchResultCounts
    display_results: list[PaperResult] = Field(default_factory=list)
    satisfied: list[PaperResult] = Field(default_factory=list)
    partial: list[PaperResult] = Field(default_factory=list)
    rejected: list[PaperResult] = Field(default_factory=list)


class PaperChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PaperChatRequest(BaseModel):
    paper_id: str
    query: str
    history: list[PaperChatMessage] = Field(default_factory=list)


class PaperChatCitation(BaseModel):
    evidence_id: str
    page_start: int
    page_end: int
    section_path: list[str] = Field(default_factory=list)
    snippet: str
    html: str | None = None


class PaperChatEvidence(BaseModel):
    evidence_id: str
    evidence_type: str
    heading: str
    section_path: list[str] = Field(default_factory=list)
    page_start: int
    page_end: int
    snippet: str
    html: str | None = None


class PaperChatVerifier(BaseModel):
    support_verdict: str
    alignment_verdict: str
    interpretation_verdict: str = "resolved"
    competition_verdict: str = "distinct_winner"
    strongest_competitor_id: str | None = None
    confidence: float = 0.0
    verified_evidence_ids: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


class PaperChatResponse(BaseModel):
    paper_id: str
    decision: Literal["ask_clarification", "answer", "unsupported"] = "answer"
    answer: str
    rewritten_query: str | None = None
    uncertainty_note: str | None = None
    citations: list[PaperChatCitation] = Field(default_factory=list)
    evidence: list[PaperChatEvidence] = Field(default_factory=list)
    verifier: PaperChatVerifier | None = None


class CreateProjectRequest(BaseModel):
    title: str


class UpdateProjectRequest(BaseModel):
    title: str


class ProjectSummaryResponse(BaseModel):
    project_id: str
    title: str
    created_at: str
    updated_at: str
    search_thread_count: int = 0
    paper_session_count: int = 0


class ProjectListResponse(BaseModel):
    projects: list[ProjectSummaryResponse] = Field(default_factory=list)


class ProjectDetailResponse(BaseModel):
    project: ProjectSummaryResponse
    threads: list[ProjectSearchThreadRecord] = Field(default_factory=list)
    paper_sessions: list[ProjectPaperSessionRecord] = Field(default_factory=list)


class ProjectMutationResponse(BaseModel):
    project_id: str
    status: Literal["cleared", "deleted"]


class UpsertProjectThreadRequest(BaseModel):
    query: str
    trace_id: str | None = None
    result_counts: dict[str, int] = Field(default_factory=dict)
    paper_ids: list[str] = Field(default_factory=list)


class UpsertProjectPaperSessionRequest(BaseModel):
    paper_title: str | None = None
    source_thread_id: str | None = None
    chat_history: list[ProjectChatMessage] = Field(default_factory=list)
    last_active_evidence_id: str | None = None


__all__ = [
    "CreateProjectRequest",
    "CreateSearchJobRequest",
    "HealthJobSummary",
    "HealthResponse",
    "PaperChatCitation",
    "PaperChatEvidence",
    "PaperChatMessage",
    "PaperChatRequest",
    "PaperChatResponse",
    "PaperChatVerifier",
    "ProjectDetailResponse",
    "ProjectListResponse",
    "ProjectMutationResponse",
    "ProjectSummaryResponse",
    "SearchJobProgressResponse",
    "PaperRecord",
    "ProjectPaperSessionRecord",
    "ProjectRecord",
    "ProjectSearchThreadRecord",
    "SearchJobResultResponse",
    "SearchJobStatusResponse",
    "SearchResultCounts",
    "SearchTrace",
    "UpdateProjectRequest",
    "UpsertProjectPaperSessionRequest",
    "UpsertProjectThreadRequest",
]
