from __future__ import annotations

from fastapi import APIRouter, Request

from ...config import Settings
from ...corpus_catalog import load_search_current_catalog
from ..schemas import CorpusCatalogEntryResponse, CorpusCatalogResponse

router = APIRouter()


@router.get("/corpora/catalog", response_model=CorpusCatalogResponse)
def get_corpus_catalog(request: Request) -> CorpusCatalogResponse:
    root_dir = request.app.state.service_manager.root_dir
    settings = Settings.from_env(root_dir=root_dir)
    catalog = load_search_current_catalog(settings)
    return CorpusCatalogResponse(
        build_id=catalog.build_id,
        built_at=catalog.built_at,
        corpora=[
            CorpusCatalogEntryResponse(
                corpus_key=entry.corpus_key,
                venue=entry.venue,
                year=entry.year,
                track=entry.track,
                papers=entry.papers,
                chunks=entry.chunks,
                deep_chat_evidence_units=entry.deep_chat_evidence_units,
            )
            for entry in catalog.corpora
        ],
    )
