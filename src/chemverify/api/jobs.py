from __future__ import annotations

from dataclasses import dataclass, field
from queue import Queue
from threading import Lock, Thread
from time import perf_counter
from uuid import uuid4

from ..models import SearchResponse
from ..search import SearchEngine, SearchProgressUpdate
from ..utils import now_iso
from .schemas import (
    HealthJobSummary,
    SearchJobProgressResponse,
    SearchJobResultResponse,
    SearchJobStatusResponse,
    SearchResultCounts,
)

_SENTINEL = object()


@dataclass(slots=True)
class _JobRecord:
    job_id: str
    project_id: str
    query: str
    workspace_scope: list[str]
    top_k: int
    display_k: int
    status: str = "queued"
    stage: str = "queued"
    message: str = "Queued for execution."
    created_at: str = field(default_factory=now_iso)
    created_perf: float = field(default_factory=perf_counter)
    started_at: str | None = None
    started_perf: float | None = None
    finished_at: str | None = None
    finished_perf: float | None = None
    trace_id: str | None = None
    error: str | None = None
    result: SearchResponse | None = None
    progress: SearchProgressUpdate | None = None


class SearchJobManager:
    def __init__(self, engine: SearchEngine) -> None:
        self.engine = engine
        self._jobs: dict[str, _JobRecord] = {}
        self._queue: Queue[str | object] = Queue()
        self._lock = Lock()
        self._worker = Thread(target=self._worker_loop, name="paper-search-job-worker", daemon=True)
        self._worker.start()

    def close(self) -> None:
        self._queue.put(_SENTINEL)
        self._worker.join(timeout=5.0)

    def submit(
        self,
        *,
        query: str,
        project_id: str,
        workspace_scope: list[str],
        top_k: int,
        display_k: int,
    ) -> SearchJobStatusResponse:
        job = _JobRecord(
            job_id=uuid4().hex[:16],
            project_id=project_id,
            query=query,
            workspace_scope=list(workspace_scope),
            top_k=top_k,
            display_k=display_k,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._queue.put(job.job_id)
        return self._status_response(job)

    def get_status(self, job_id: str) -> SearchJobStatusResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return self._status_response(job)

    def get_result(self, job_id: str) -> SearchJobResultResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.result is None:
                return None
            result = job.result
            return SearchJobResultResponse(
                job_id=job.job_id,
                query=job.query,
                trace_id=result.trace_id,
                mode=result.mode,
                workspace_scope=result.workspace_scope,
                query_scope=result.query_scope.model_dump(),
                effective_scope=result.effective_scope,
                counts=SearchResultCounts(
                    satisfied=len(result.satisfied),
                    partial=len(result.partial),
                    rejected=len(result.rejected),
                ),
                display_results=self._select_display_results(result, job.display_k),
                satisfied=result.satisfied,
                partial=result.partial,
                rejected=result.rejected,
            )

    def summary(self) -> HealthJobSummary:
        with self._lock:
            counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
            return HealthJobSummary(total_jobs=len(self._jobs), **counts)

    def should_retain(self, *, now: float | None = None, completed_job_retention_seconds: float) -> bool:
        current = perf_counter() if now is None else now
        with self._lock:
            if not self._jobs:
                return False
            for job in self._jobs.values():
                if job.status in {"queued", "running"}:
                    return True
                finished_perf = job.finished_perf if job.finished_perf is not None else job.created_perf
                if current - finished_perf <= completed_job_retention_seconds:
                    return True
            return False

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is _SENTINEL:
                self._queue.task_done()
                return
            assert isinstance(job_id, str)
            self._run_job(job_id)
            self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"
            job.stage = "starting"
            job.message = "Starting search job."
            job.started_at = now_iso()
            job.started_perf = perf_counter()

        try:
            result = self.engine.search(
                job.query,
                top_k=job.top_k,
                workspace_scope=job.workspace_scope,
                progress_callback=lambda update: self._update_progress(job_id, update),
            )
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.stage = "failed"
                job.message = "Search job failed."
                job.error = str(exc)
                job.finished_at = now_iso()
                job.finished_perf = perf_counter()
            return

        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.stage = "completed"
            job.message = "Search job completed."
            job.finished_at = now_iso()
            job.finished_perf = perf_counter()
            job.trace_id = result.trace_id
            job.result = result

    def _update_progress(self, job_id: str, update: SearchProgressUpdate) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status == "failed":
                return
            job.stage = update.stage
            job.message = update.message
            job.progress = update

    def _status_response(self, job: _JobRecord) -> SearchJobStatusResponse:
        if job.status == "completed" and job.finished_perf is not None and job.started_perf is not None:
            elapsed_ms = (job.finished_perf - job.started_perf) * 1000
        elif job.started_perf is not None:
            elapsed_ms = (perf_counter() - job.started_perf) * 1000
        else:
            elapsed_ms = (perf_counter() - job.created_perf) * 1000
        return SearchJobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            stage=job.stage,
            message=job.message,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            elapsed_ms=elapsed_ms,
            trace_id=job.trace_id,
            error=job.error,
            progress=self._progress_response(job.progress),
        )

    @staticmethod
    def _progress_response(update: SearchProgressUpdate | None) -> SearchJobProgressResponse | None:
        if update is None:
            return None
        return SearchJobProgressResponse(
            stage_index=update.stage_index,
            stage_total=update.stage_total,
            stage_progress=update.stage_progress,
            overall_progress=update.overall_progress,
            completed_items=update.completed_items,
            total_items=update.total_items,
        )

    @staticmethod
    def _select_display_results(result: SearchResponse, display_k: int):
        if display_k <= 0:
            return []
        display_results = list(result.satisfied[:display_k])
        if len(display_results) < display_k:
            remaining = display_k - len(display_results)
            display_results.extend(result.partial[:remaining])
        return display_results[:display_k]
