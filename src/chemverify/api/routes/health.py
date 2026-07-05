from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    with request.app.state.service_manager.acquire_services() as services:
        index_state = services.store.load_index_state()
        counts = {
            "papers": int(index_state.get("papers", 0)),
            "chunks": int(index_state.get("chunks", 0)),
            "traces": _count_json_files(services.store.trace_dir),
        }
        return HealthResponse(
            data_dir=str(services.settings.data_dir),
            index_state=index_state,
            counts=counts,
            jobs=services.jobs.summary(),
        )


def _count_json_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for candidate in path.iterdir() if candidate.is_file() and candidate.suffix == ".json")
