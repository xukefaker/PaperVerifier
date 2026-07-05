from __future__ import annotations

import json
from dataclasses import dataclass

from .config import Settings

DEFAULT_DEMO_CORPUS = "chemqa40/2026/all"


@dataclass(slots=True)
class CorpusCatalogEntry:
    corpus_key: str
    venue: str
    year: int
    track: str
    papers: int
    chunks: int
    deep_chat_evidence_units: int


@dataclass(slots=True)
class CorpusCatalog:
    build_id: str | None
    built_at: str | None
    corpora: list[CorpusCatalogEntry]

    @property
    def corpus_keys(self) -> list[str]:
        return [entry.corpus_key for entry in self.corpora]


def load_search_current_catalog(settings: Settings) -> CorpusCatalog:
    manifest_path = settings.search_current_manifest_path
    if not manifest_path.exists():
        return CorpusCatalog(build_id=None, built_at=None, corpora=[])

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries: list[CorpusCatalogEntry] = []
    for item in payload.get("corpora", []):
        corpus_key = str(item.get("corpus") or "").strip()
        if not corpus_key:
            continue
        parts = corpus_key.split("/")
        if len(parts) != 3:
            continue
        venue, year_text, track = parts
        try:
            year = int(year_text)
        except ValueError:
            continue
        entries.append(
            CorpusCatalogEntry(
                corpus_key=corpus_key,
                venue=venue,
                year=year,
                track=track,
                papers=int(item.get("papers") or 0),
                chunks=int(item.get("chunks") or 0),
                deep_chat_evidence_units=int(item.get("deep_chat_evidence_units") or 0),
            )
        )

    entries.sort(key=lambda entry: (entry.venue, entry.year, entry.track))
    return CorpusCatalog(
        build_id=str(payload.get("build_id") or "") or None,
        built_at=str(payload.get("built_at") or "") or None,
        corpora=entries,
    )


def default_selected_corpora(settings: Settings) -> list[str]:
    catalog = load_search_current_catalog(settings)
    if DEFAULT_DEMO_CORPUS in catalog.corpus_keys:
        return [DEFAULT_DEMO_CORPUS]
    return catalog.corpus_keys
