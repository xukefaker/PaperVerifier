from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import CorpusSpec, Settings
from .utils import now_iso

NORMALIZED_JSONL_SPECS = (
    ("papers", "paper_id"),
    ("sections", "section_id"),
    ("objects", "object_id"),
    ("chunks", "chunk_id"),
    ("parse_failures", "paper_id"),
)

LAYOUT_INDEX_NAMES = ("paper", "section", "chunk", "text_chunk", "table_chunk", "figure_chunk")

REQUIRED_RELEASE_FILES = (
    "normalized/papers.jsonl",
    "normalized/sections.jsonl",
    "normalized/objects.jsonl",
    "normalized/chunks.jsonl",
    "normalized/parse_failures.jsonl",
    "normalized/deep_chat/evidence_units.jsonl",
    "indexes/layout/index_state.json",
    "indexes/layout/paper_index_meta.json",
    "indexes/layout/paper_vectors.npz",
    "indexes/layout/section_index_meta.json",
    "indexes/layout/section_vectors.npz",
    "indexes/layout/chunk_index_meta.json",
    "indexes/layout/chunk_vectors.npz",
    "indexes/layout/text_chunk_index_meta.json",
    "indexes/layout/text_chunk_vectors.npz",
    "indexes/layout/table_chunk_index_meta.json",
    "indexes/layout/table_chunk_vectors.npz",
    "indexes/layout/figure_chunk_index_meta.json",
    "indexes/layout/figure_chunk_vectors.npz",
    "indexes/deep_chat/evidence_unit_index_meta.json",
    "indexes/deep_chat/evidence_unit_vectors.npz",
)

INDEX_STATE_SUM_KEYS = (
    "total_papers",
    "papers",
    "indexed_papers",
    "failed_papers",
    "sections",
    "objects",
    "chunks",
    "text_chunks",
    "table_chunks",
    "figure_chunks",
    "deep_chat_evidence_units",
)

INDEX_STATE_EQUAL_KEYS = (
    "paper_dense_backend",
    "chunk_dense_backend",
    "paper_dense_model",
    "chunk_dense_model",
    "paper_vector_dim",
    "chunk_vector_dim",
    "pdf_parser_backend",
)

META_SCALAR_KEYS = ("encoder_backend", "encoder_model", "vector_dim")
META_NONMERGED_KEYS = (*META_SCALAR_KEYS, "built_at")


@dataclass(slots=True)
class CorpusRelease:
    venue: str
    year: int
    track: str
    corpus_root: Path
    release_root: Path
    state_path: Path
    state_payload: dict[str, Any]

    @property
    def key(self) -> str:
        return f"{self.venue}/{self.year}/{self.track}"


def rebuild_search_current(
    root_dir: str | Path,
    *,
    corpora: list[CorpusSpec] | None = None,
    allow_uncompleted_selected: bool = False,
) -> dict[str, Any]:
    settings = Settings.from_env(root_dir=root_dir)
    releases = discover_completed_releases(
        settings,
        corpora=corpora,
        allow_uncompleted_selected=allow_uncompleted_selected,
    )
    if not releases:
        raise RuntimeError("No completed corpus release is available for data/search_current.")

    staging_dir = settings.search_current_staging_dir
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    (staging_dir / "normalized" / "deep_chat").mkdir(parents=True, exist_ok=True)
    (staging_dir / "indexes" / "layout").mkdir(parents=True, exist_ok=True)
    (staging_dir / "indexes" / "deep_chat").mkdir(parents=True, exist_ok=True)
    (staging_dir / "traces").mkdir(parents=True, exist_ok=True)
    _copy_existing_traces(existing_root=settings.search_current_dir, staging_root=staging_dir)

    jsonl_counts: dict[str, int] = {}
    for name, record_id_key in NORMALIZED_JSONL_SPECS:
        jsonl_counts[name] = _merge_jsonl(
            releases=releases,
            relative_path=Path("normalized") / f"{name}.jsonl",
            destination=staging_dir / "normalized" / f"{name}.jsonl",
            record_id_key=record_id_key,
        )
    jsonl_counts["evidence_units"] = _merge_jsonl(
        releases=releases,
        relative_path=Path("normalized") / "deep_chat" / "evidence_units.jsonl",
        destination=staging_dir / "normalized" / "deep_chat" / "evidence_units.jsonl",
        record_id_key="evidence_id",
    )

    merged_index_counts: dict[str, int] = {}
    for index_name in LAYOUT_INDEX_NAMES:
        merged_index_counts[index_name] = _merge_index_bundle(
            releases=releases,
            meta_relative_path=Path("indexes") / "layout" / f"{index_name}_index_meta.json",
            vectors_relative_path=Path("indexes") / "layout" / f"{index_name}_vectors.npz",
            destination_meta=staging_dir / "indexes" / "layout" / f"{index_name}_index_meta.json",
            destination_vectors=staging_dir / "indexes" / "layout" / f"{index_name}_vectors.npz",
        )
    merged_index_counts["evidence_unit"] = _merge_index_bundle(
        releases=releases,
        meta_relative_path=Path("indexes") / "deep_chat" / "evidence_unit_index_meta.json",
        vectors_relative_path=Path("indexes") / "deep_chat" / "evidence_unit_vectors.npz",
        destination_meta=staging_dir / "indexes" / "deep_chat" / "evidence_unit_index_meta.json",
        destination_vectors=staging_dir / "indexes" / "deep_chat" / "evidence_unit_vectors.npz",
    )

    merged_index_state = _merge_index_state(releases, jsonl_counts=jsonl_counts, index_counts=merged_index_counts)
    (staging_dir / "indexes" / "layout" / "index_state.json").write_text(
        json.dumps(merged_index_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = _build_manifest(releases=releases, index_state=merged_index_state, jsonl_counts=jsonl_counts)
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if settings.search_current_dir.exists() or settings.search_current_dir.is_symlink():
        if settings.search_current_dir.is_symlink() or settings.search_current_dir.is_file():
            settings.search_current_dir.unlink()
        else:
            shutil.rmtree(settings.search_current_dir)
    staging_dir.rename(settings.search_current_dir)
    return manifest


def discover_completed_releases(
    settings: Settings,
    *,
    corpora: list[CorpusSpec] | None = None,
    allow_uncompleted_selected: bool = False,
) -> list[CorpusRelease]:
    corpora_root = settings.data_dir / "corpora"
    if corpora:
        releases = [
            _load_selected_release(
                settings=settings,
                corpus=corpus,
                allow_uncompleted_selected=allow_uncompleted_selected,
            )
            for corpus in _dedupe_corpora(corpora)
        ]
        return sorted(releases, key=_release_sort_key)

    releases: list[CorpusRelease] = []
    for state_path in sorted(corpora_root.glob("*/*/*/state/job_state.json")):
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        relative = state_path.relative_to(corpora_root)
        venue, year_text, track = relative.parts[:3]
        corpus_root = state_path.parents[1]
        release_root = corpus_root / "release" / "current"
        _validate_release(release_root)
        releases.append(
            CorpusRelease(
                venue=venue,
                year=int(year_text),
                track=track,
                corpus_root=corpus_root,
                release_root=release_root,
                state_path=state_path,
                state_payload=payload,
            )
        )
    return sorted(releases, key=_release_sort_key)


def _load_selected_release(
    *,
    settings: Settings,
    corpus: CorpusSpec,
    allow_uncompleted_selected: bool,
) -> CorpusRelease:
    corpus_root = settings.data_dir / "corpora" / corpus.venue / str(corpus.year) / corpus.track
    state_path = corpus_root / "state" / "job_state.json"
    if not state_path.exists():
        raise RuntimeError(f"Selected corpus has no job state file: {corpus.key}")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" and not allow_uncompleted_selected:
        raise RuntimeError(f"Selected corpus is not completed and cannot be published yet: {corpus.key}")
    release_root = corpus_root / "release" / "current"
    _validate_release(release_root)
    return CorpusRelease(
        venue=corpus.venue,
        year=corpus.year,
        track=corpus.track,
        corpus_root=corpus_root,
        release_root=release_root,
        state_path=state_path,
        state_payload=payload,
    )


def _dedupe_corpora(corpora: list[CorpusSpec]) -> list[CorpusSpec]:
    seen: set[str] = set()
    output: list[CorpusSpec] = []
    for corpus in corpora:
        if corpus.key in seen:
            continue
        seen.add(corpus.key)
        output.append(corpus)
    return output


def _release_sort_key(release: CorpusRelease) -> tuple[str, int, int, str]:
    return (release.venue, release.year, _track_priority(release.track), release.track)


def _track_priority(track: str) -> int:
    return 1 if track == "all" else 0


def _copy_existing_traces(*, existing_root: Path, staging_root: Path) -> None:
    trace_dir = existing_root / "traces"
    if not trace_dir.exists():
        return
    destination = staging_root / "traces"
    destination.mkdir(parents=True, exist_ok=True)
    for candidate in trace_dir.iterdir():
        if candidate.is_file():
            shutil.copy2(candidate, destination / candidate.name)


def _validate_release(release_root: Path) -> None:
    missing = [relative for relative in REQUIRED_RELEASE_FILES if not (release_root / relative).exists()]
    if missing:
        raise RuntimeError(
            f"Release {release_root} is incomplete; missing files: {', '.join(missing[:5])}"
            + (" ..." if len(missing) > 5 else "")
        )


def _merge_jsonl(
    *,
    releases: list[CorpusRelease],
    relative_path: Path,
    destination: Path,
    record_id_key: str,
) -> int:
    seen_ids: set[str] = set()
    count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for release in releases:
            source = release.release_root / relative_path
            with source.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    payload = json.loads(raw_line)
                    record_id = str(payload.get(record_id_key) or "")
                    if not record_id:
                        raise RuntimeError(f"{source} contains an entry without {record_id_key}.")
                    if record_id in seen_ids:
                        continue
                    seen_ids.add(record_id)
                    output.write(json.dumps(payload, ensure_ascii=False))
                    output.write("\n")
                    count += 1
    return count


def _merge_index_bundle(
    *,
    releases: list[CorpusRelease],
    meta_relative_path: Path,
    vectors_relative_path: Path,
    destination_meta: Path,
    destination_vectors: Path,
) -> int:
    merged_ids: list[str] = []
    merged_lists: dict[str, list[Any]] = {}
    matrices: list[np.ndarray] = []
    meta_key_order: list[str] | None = None
    scalar_reference: dict[str, Any] | None = None
    vector_dim: int | None = None
    seen_ids: set[str] = set()

    for release in releases:
        meta = json.loads((release.release_root / meta_relative_path).read_text(encoding="utf-8"))
        payload = np.load(release.release_root / vectors_relative_path, allow_pickle=True)
        ids = [str(item) for item in payload["ids"].tolist()]
        matrix = np.array(payload["matrix"])
        _validate_index_meta(meta=meta, ids=ids, matrix=matrix, source=release.release_root / meta_relative_path)

        current_scalars = {key: meta[key] for key in META_SCALAR_KEYS}
        if scalar_reference is None:
            scalar_reference = current_scalars
            meta_key_order = list(meta.keys())
            vector_dim = int(meta["vector_dim"])
            merged_lists = {
                key: [] for key, value in meta.items() if isinstance(value, list) and key not in META_NONMERGED_KEYS
            }
        elif current_scalars != scalar_reference:
            raise RuntimeError(
                f"Encoder mismatch while merging {meta_relative_path.name}: "
                f"expected {scalar_reference}, got {current_scalars} from {release.key}."
            )

        assert meta_key_order is not None
        keep_indexes = [index for index, record_id in enumerate(ids) if record_id not in seen_ids]
        if len(keep_indexes) != len(ids):
            ids = [ids[index] for index in keep_indexes]
            matrix = matrix[keep_indexes]

        for key in merged_lists:
            values = meta.get(key)
            if not isinstance(values, list):
                raise RuntimeError(f"Missing list field {key} in {meta_relative_path} for {release.key}.")
            if len(values) != len(meta["ids"]):
                raise RuntimeError(
                    f"Length mismatch for field {key} in {meta_relative_path} for {release.key}: "
                    f"{len(values)} vs {len(meta['ids'])}."
                )
            if len(keep_indexes) != len(values):
                values = [values[index] for index in keep_indexes]
            merged_lists[key].extend(values)

        for record_id in ids:
            seen_ids.add(record_id)
        merged_ids.extend(ids)
        if vector_dim is not None and matrix.shape[1] != vector_dim:
            raise RuntimeError(
                f"Vector dimension mismatch in {vectors_relative_path.name}: "
                f"expected {vector_dim}, got {matrix.shape[1]} from {release.key}."
            )
        matrices.append(matrix)

    if scalar_reference is None or meta_key_order is None or vector_dim is None:
        raise RuntimeError(f"No index payloads found for {meta_relative_path.name}.")

    merged_meta: dict[str, Any] = {}
    for key in meta_key_order:
        if key == "built_at":
            merged_meta[key] = now_iso()
        elif key == "ids":
            merged_meta[key] = merged_ids
        elif key in scalar_reference:
            merged_meta[key] = scalar_reference[key]
        else:
            merged_meta[key] = merged_lists[key]

    if matrices:
        merged_matrix = np.concatenate(matrices, axis=0)
    else:
        merged_matrix = np.empty((0, vector_dim), dtype=np.float32)
    destination_meta.parent.mkdir(parents=True, exist_ok=True)
    destination_meta.write_text(json.dumps(merged_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(destination_vectors, ids=np.array(merged_ids, dtype=object), matrix=merged_matrix)
    return len(merged_ids)


def _validate_index_meta(*, meta: dict[str, Any], ids: list[str], matrix: np.ndarray, source: Path) -> None:
    if matrix.ndim != 2:
        raise RuntimeError(f"{source} has invalid vector matrix shape {matrix.shape}.")
    meta_ids = [str(item) for item in meta.get("ids", [])]
    if meta_ids != ids:
        raise RuntimeError(f"{source} ids do not align with vector payload ids.")
    if len(meta_ids) != matrix.shape[0]:
        raise RuntimeError(
            f"{source} count mismatch: meta ids={len(meta_ids)} but vectors rows={matrix.shape[0]}."
        )
    for key, value in meta.items():
        if isinstance(value, list) and key not in META_NONMERGED_KEYS and len(value) != len(meta_ids):
            raise RuntimeError(
                f"{source} field {key} has length {len(value)} but expected {len(meta_ids)}."
            )


def _merge_index_state(
    releases: list[CorpusRelease],
    *,
    jsonl_counts: dict[str, int],
    index_counts: dict[str, int],
) -> dict[str, Any]:
    merged: dict[str, Any] = {"built_at": now_iso()}
    equal_reference: dict[str, Any] | None = None
    for release in releases:
        payload = json.loads(
            (release.release_root / "indexes" / "layout" / "index_state.json").read_text(encoding="utf-8")
        )
        if equal_reference is None:
            equal_reference = {key: payload.get(key) for key in INDEX_STATE_EQUAL_KEYS}
            merged.update(equal_reference)
            merged["parse_failure_path"] = ""
        else:
            current = {key: payload.get(key) for key in INDEX_STATE_EQUAL_KEYS}
            if current != equal_reference:
                raise RuntimeError(
                    f"Index-state mismatch while merging {release.key}: expected {equal_reference}, got {current}."
                )
        for key in INDEX_STATE_SUM_KEYS:
            merged[key] = int(merged.get(key, 0)) + int(payload.get(key, 0))
    merged["total_papers"] = int(jsonl_counts["papers"] + jsonl_counts["parse_failures"])
    merged["papers"] = int(index_counts["paper"])
    merged["indexed_papers"] = int(index_counts["paper"])
    merged["failed_papers"] = int(jsonl_counts["parse_failures"])
    merged["sections"] = int(jsonl_counts["sections"])
    merged["objects"] = int(jsonl_counts["objects"])
    merged["chunks"] = int(jsonl_counts["chunks"])
    merged["text_chunks"] = int(index_counts["text_chunk"])
    merged["table_chunks"] = int(index_counts["table_chunk"])
    merged["figure_chunks"] = int(index_counts["figure_chunk"])
    merged["deep_chat_evidence_units"] = int(jsonl_counts["evidence_units"])
    merged["parse_failure_path"] = "normalized/parse_failures.jsonl"
    merged["corpora"] = [release.key for release in releases]
    return merged


def _build_manifest(
    *,
    releases: list[CorpusRelease],
    index_state: dict[str, Any],
    jsonl_counts: dict[str, int],
) -> dict[str, Any]:
    corpora_payload = []
    fingerprint_payload = []
    for release in releases:
        current_path = release.release_root.resolve(strict=False)
        state_updated_at = release.state_payload.get("updated_at")
        index_state_path = release.release_root / "indexes" / "layout" / "index_state.json"
        release_index_state = json.loads(index_state_path.read_text(encoding="utf-8"))
        item = {
            "corpus": release.key,
            "release_path": str(current_path),
            "job_state_path": str(release.state_path),
            "job_updated_at": state_updated_at,
            "release_built_at": release_index_state.get("built_at"),
            "papers": int(release_index_state.get("papers", 0)),
            "chunks": int(release_index_state.get("chunks", 0)),
            "deep_chat_evidence_units": int(release_index_state.get("deep_chat_evidence_units", 0)),
        }
        corpora_payload.append(item)
        fingerprint_payload.append(item)
    serialized = json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False)
    build_id = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]
    return {
        "build_id": build_id,
        "built_at": now_iso(),
        "corpora": corpora_payload,
        "counts": {
            "papers": int(index_state.get("papers", 0)),
            "chunks": int(index_state.get("chunks", 0)),
            "deep_chat_evidence_units": int(index_state.get("deep_chat_evidence_units", 0)),
            "jsonl": jsonl_counts,
        },
    }
