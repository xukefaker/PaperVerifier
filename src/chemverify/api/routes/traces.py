from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ...models import SearchTrace

router = APIRouter()


@router.get("/traces/{trace_id}", response_model=SearchTrace)
def get_trace(request: Request, trace_id: str) -> SearchTrace:
    store = request.app.state.service_manager.find_trace_store(trace_id)
    if store is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    trace = store.load_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
