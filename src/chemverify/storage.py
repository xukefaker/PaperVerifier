from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np
from pydantic import BaseModel

from .config import Settings
from .models import (
    ChunkRecord,
    ObjectRecord,
    PaperRecord,
    ParseFailureRecord,
    SearchTrace,
    SectionRecord,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class LocalStore:
    def __init__(self, settings: Settings, *, root_dir: Path | None = None) -> None:
        self.settings = settings
        self.root_dir = root_dir
        if root_dir is None:
            normalized_dir = settings.normalized_dir
            index_dir = settings.index_dir
            trace_dir = settings.trace_dir
            raw_paper_path = settings.raw_dir / "papers.jsonl"
        else:
            normalized_dir = root_dir / "normalized"
            index_dir = root_dir / "indexes" / "layout"
            trace_dir = root_dir / "traces"
            raw_paper_path = root_dir / "raw" / "papers.jsonl"
        self.normalized_dir = normalized_dir
        self.index_dir = index_dir
        self.trace_dir = trace_dir
        self.raw_paper_path = raw_paper_path
        self.paper_path = normalized_dir / "papers.jsonl"
        self.section_path = normalized_dir / "sections.jsonl"
        self.object_path = normalized_dir / "objects.jsonl"
        self.chunk_path = normalized_dir / "chunks.jsonl"
        self.parse_failure_path = normalized_dir / "parse_failures.jsonl"
        self.index_state_path = index_dir / "index_state.json"

    def save_raw_papers(self, papers: Iterable[PaperRecord]) -> None:
        self._write_jsonl(self.raw_paper_path, papers)

    def load_raw_papers(self) -> list[PaperRecord]:
        return self._read_jsonl(self.raw_paper_path, PaperRecord)

    def load_source_papers(self) -> list[PaperRecord]:
        raw_papers = self.load_raw_papers()
        if raw_papers:
            return raw_papers
        return self.load_papers()

    def save_papers(self, papers: Iterable[PaperRecord]) -> None:
        self._write_jsonl(self.paper_path, papers)

    def load_papers(self) -> list[PaperRecord]:
        return self._read_jsonl(self.paper_path, PaperRecord)

    def save_sections(self, sections: Iterable[SectionRecord]) -> None:
        self._write_jsonl(self.section_path, sections)

    def load_sections(self) -> list[SectionRecord]:
        return self._read_jsonl(self.section_path, SectionRecord)

    def save_objects(self, objects: Iterable[ObjectRecord]) -> None:
        self._write_jsonl(self.object_path, objects)

    def load_objects(self) -> list[ObjectRecord]:
        return self._read_jsonl(self.object_path, ObjectRecord)

    def save_chunks(self, chunks: Iterable[ChunkRecord]) -> None:
        self._write_jsonl(self.chunk_path, chunks)

    def load_chunks(self) -> list[ChunkRecord]:
        return self._read_jsonl(self.chunk_path, ChunkRecord)

    def save_parse_failures(self, failures: Iterable[ParseFailureRecord]) -> None:
        self._write_jsonl(self.parse_failure_path, failures)

    def load_parse_failures(self) -> list[ParseFailureRecord]:
        return self._read_jsonl(self.parse_failure_path, ParseFailureRecord)

    def save_index_meta(self, name: str, payload: dict) -> None:
        path = self.index_dir / f"{name}_index_meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_index_meta(self, name: str) -> dict:
        path = self.index_dir / f"{name}_index_meta.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_vectors(self, name: str, ids: list[str], matrix: np.ndarray) -> None:
        path = self.index_dir / f"{name}_vectors.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=np.array(ids, dtype=object), matrix=matrix)

    def load_vectors(self, name: str) -> tuple[list[str], np.ndarray]:
        path = self.index_dir / f"{name}_vectors.npz"
        if not path.exists():
            return [], np.empty((0, 0), dtype=float)
        data = np.load(path, allow_pickle=True)
        return [str(item) for item in data["ids"].tolist()], np.asarray(data["matrix"], dtype=np.float32)

    def save_index_state(self, payload: dict) -> None:
        self.index_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_index_state(self) -> dict:
        if not self.index_state_path.exists():
            return {}
        return json.loads(self.index_state_path.read_text(encoding="utf-8"))

    def save_trace(self, trace: SearchTrace) -> None:
        path = self.trace_dir / f"{trace.trace_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")

    def load_trace(self, trace_id: str) -> SearchTrace | None:
        path = self.trace_dir / f"{trace_id}.json"
        if not path.exists():
            return None
        return SearchTrace.model_validate_json(path.read_text(encoding="utf-8"))

    def get_paper(self, paper_id: str) -> PaperRecord | None:
        for paper in self.load_papers():
            if paper.paper_id == paper_id:
                return paper
        for paper in self.load_raw_papers():
            if paper.paper_id == paper_id:
                return paper
        return None

    def _write_jsonl(self, path: Path, items: Iterable[BaseModel]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(item.model_dump_json())
                handle.write("\n")

    def _read_jsonl(self, path: Path, model: type[ModelT]) -> list[ModelT]:
        if not path.exists():
            return []
        output: list[ModelT] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    output.append(model.model_validate_json(line))
        return output
