from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np
from pydantic import BaseModel

from ..config import Settings
from .models import EvidenceUnit

ModelT = TypeVar("ModelT", bound=BaseModel)


class DeepChatStore:
    def __init__(self, settings: Settings, *, root_dir: Path | None = None) -> None:
        self.settings = settings
        if root_dir is None:
            normalized_dir = settings.deep_chat_normalized_dir
            index_dir = settings.deep_chat_index_dir
        else:
            normalized_dir = root_dir / "normalized" / "deep_chat"
            index_dir = root_dir / "indexes" / "deep_chat"
        self.normalized_dir = normalized_dir
        self.index_dir = index_dir
        self.evidence_unit_path = normalized_dir / "evidence_units.jsonl"
        self.index_meta_path = index_dir / "evidence_unit_index_meta.json"
        self.vector_path = index_dir / "evidence_unit_vectors.npz"

    def save_evidence_units(self, units: Iterable[EvidenceUnit]) -> None:
        self._write_jsonl(self.evidence_unit_path, units)

    def load_evidence_units(self) -> list[EvidenceUnit]:
        return self._read_jsonl(self.evidence_unit_path, EvidenceUnit)

    def save_index_meta(self, payload: dict) -> None:
        self.index_meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_index_meta(self) -> dict:
        if not self.index_meta_path.exists():
            return {}
        return json.loads(self.index_meta_path.read_text(encoding="utf-8"))

    def save_vectors(self, ids: list[str], matrix: np.ndarray) -> None:
        self.vector_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.vector_path, ids=np.array(ids, dtype=object), matrix=matrix)

    def load_vectors(self) -> tuple[list[str], np.ndarray]:
        if not self.vector_path.exists():
            return [], np.empty((0, 0), dtype=float)
        payload = np.load(self.vector_path, allow_pickle=True)
        return [str(item) for item in payload["ids"].tolist()], np.asarray(payload["matrix"], dtype=np.float32)

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
