from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .acl_anthology import ACLAnthologyIngestor
from .config import CorpusSpec, Settings
from .deep_chat.evidence import DeepChatEvidenceMaterializer, build_evidence_search_text
from .deep_chat.models import EvidenceUnit
from .encoders import SentenceTransformerEncoder
from .grobid_client import GrobidHeaderClient
from .indexer import ENCODE_LABELS, IndexBuilder
from .mineru_pipeline import (
    artifact_complete,
    normalize_failure_entries,
    remove_artifacts,
    remove_failure_entries,
    run_mineru_pipeline,
)
from .models import (
    BuildIndexSummary,
    ChunkRecord,
    ObjectRecord,
    PaperEnrichmentRecord,
    PaperRecord,
    PaperReferencesRecord,
    ParseFailureRecord,
    SectionRecord,
)
from .pdf_parser import PDFParser
from .presentation import (
    extract_reference_entries_from_markdown,
    save_cached_paper_enrichment_record,
    save_cached_paper_references,
)
from .search_current import rebuild_search_current
from .storage import LocalStore
from .utils import now_iso, tokenize

logger = logging.getLogger(__name__)

BUILD_FINGERPRINT_VERSION = "offline_build_v2"
SHARD_SIZE_BY_DATASET = {
    "paper": 256,
    "section": 2048,
    "chunk": 2048,
    "text_chunk": 2048,
    "table_chunk": 1024,
    "figure_chunk": 1024,
    "deep_chat_evidence": 4096,
}


class PauseRequested(RuntimeError):
    pass


@dataclass(slots=True)
class OfflineRunResult:
    status: str
    corpus: str
    message: str
    build_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "corpus": self.corpus,
            "message": self.message,
        }
        if self.build_summary is not None:
            payload["build_summary"] = self.build_summary
        return payload


@dataclass(slots=True)
class OfflineEnrichmentResult:
    status: str
    corpus: str
    message: str
    enrichment_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "corpus": self.corpus,
            "message": self.message,
        }
        if self.enrichment_summary is not None:
            payload["enrichment_summary"] = self.enrichment_summary
        return payload


class OfflineJobController:
    def __init__(self, settings: Settings, *, mode: str) -> None:
        self.settings = settings
        self.mode = mode
        self.job_id = (
            f"offline_{settings.corpus.venue}_{settings.corpus.year}_{settings.corpus.track}_{time.time_ns()}"
        )
        self.control_path = settings.state_dir / "control.json"
        self.job_state_path = settings.state_dir / "job_state.json"
        self.lock_path = settings.global_state_dir / "offline.lock"
        self._lock_handle: Any | None = None

    def start(self) -> None:
        self.settings.ensure_dirs()
        self._acquire_lock()
        self._clear_stale_active_job()
        if self.settings.active_job_path.exists():
            payload = json.loads(self.settings.active_job_path.read_text(encoding="utf-8"))
            raise RuntimeError(f"Another offline job is already active: {payload.get('job_id')}")
        self._write_json(self.control_path, {"pause_requested": False, "requested_at": None})
        descriptor = {
            "job_id": self.job_id,
            "pid": os.getpid(),
            "corpus": self.settings.corpus.to_dict(),
            "state_path": str(self.job_state_path),
            "control_path": str(self.control_path),
            "started_at": now_iso(),
        }
        self._write_json(self.settings.active_job_path, descriptor)
        self._write_json(self.settings.last_job_path, descriptor)
        self._write_json(
            self.job_state_path,
            {
                "job_id": self.job_id,
                "corpus": self.settings.corpus.to_dict(),
                "mode": self.mode,
                "started_at": descriptor["started_at"],
                "updated_at": descriptor["started_at"],
                "status": "running",
                "phase": "initializing",
                "message": "Offline job started.",
            },
        )

    def close(self) -> None:
        try:
            if self.settings.active_job_path.exists():
                payload = json.loads(self.settings.active_job_path.read_text(encoding="utf-8"))
                if payload.get("job_id") == self.job_id:
                    self.settings.active_job_path.unlink()
        finally:
            if self._lock_handle is not None:
                try:
                    fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                finally:
                    self._lock_handle.close()
                    self._lock_handle = None

    def update_state(
        self,
        *,
        status: str | None = None,
        phase: str | None = None,
        message: str | None = None,
        progress: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = self._read_json(self.job_state_path)
        payload["job_id"] = self.job_id
        payload["corpus"] = self.settings.corpus.to_dict()
        payload["mode"] = self.mode
        payload.setdefault("started_at", now_iso())
        payload["updated_at"] = now_iso()
        if status is not None:
            payload["status"] = status
        if phase is not None:
            payload["phase"] = phase
        if message is not None:
            payload["message"] = message
        if progress is not None:
            payload["progress"] = progress
        if extra:
            payload.update(extra)
        self._write_json(self.job_state_path, payload)
        descriptor = self._read_json(self.settings.last_job_path)
        descriptor.update(
            {
                "job_id": self.job_id,
                "pid": os.getpid(),
                "corpus": self.settings.corpus.to_dict(),
                "state_path": str(self.job_state_path),
                "control_path": str(self.control_path),
                "started_at": payload.get("started_at"),
                "updated_at": payload["updated_at"],
                "status": payload.get("status"),
            }
        )
        self._write_json(self.settings.last_job_path, descriptor)

    def update_progress(
        self,
        *,
        phase: str,
        message: str,
        completed: int,
        total: int,
        unit: str,
        rate_per_min: float | None = None,
        eta_seconds: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        progress = {
            "completed": completed,
            "total": total,
            "unit": unit,
            "percent": (completed / total * 100.0) if total else 100.0,
            "rate_per_min": rate_per_min,
            "eta_seconds": eta_seconds,
            "eta": _format_duration(eta_seconds),
        }
        if extra:
            progress.update(extra)
        self.update_state(status="running", phase=phase, message=message, progress=progress)

    def mark_paused(self, message: str) -> None:
        self.update_state(status="paused", message=message)

    def mark_completed(self, *, message: str, extra: dict[str, Any] | None = None) -> None:
        self.update_state(status="completed", phase="completed", message=message, extra=extra)

    def mark_failed(self, message: str) -> None:
        self.update_state(status="failed", message=message)

    def request_pause(self) -> None:
        self._write_json(
            self.control_path,
            {
                "pause_requested": True,
                "requested_at": now_iso(),
            },
        )

    def check_pause_requested(self) -> None:
        payload = self._read_json(self.control_path)
        if payload.get("pause_requested"):
            raise PauseRequested("Pause requested. Exiting at the next safe boundary.")

    def write_active_corpus(self) -> None:
        self._write_json(self.settings.active_corpus_path, self.settings.corpus.to_dict())

    def _clear_stale_active_job(self) -> None:
        if not self.settings.active_job_path.exists():
            return
        payload = self._read_json(self.settings.active_job_path)
        pid = int(payload.get("pid") or 0)
        if pid <= 0 or not _pid_exists(pid):
            self.settings.active_job_path.unlink(missing_ok=True)

    def _acquire_lock(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_handle = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another offline job is already holding the global lock.") from exc

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


class IncrementalIndexBuilder:
    def __init__(self, settings: Settings, controller: OfflineJobController) -> None:
        self.settings = settings
        self.controller = controller
        self.store = LocalStore(settings)
        self.indexer = IndexBuilder(settings, self.store)
        self.bundle_dir = settings.work_dir / "build" / "bundles"
        self.shard_root = settings.work_dir / "build" / "shards"
        self.meta_path = settings.work_dir / "build" / "build_state.json"
        self.prepare_failure_path = settings.work_dir / "build" / "prepare_failures.jsonl"
        self.snapshot_root = settings.work_dir / "build" / "finalize_snapshot"

    def build(
        self,
        *,
        source_papers: list[PaperRecord],
        all_manifest_papers: list[PaperRecord],
        terminal_parse_failures: list[ParseFailureRecord],
        mode: str,
        total_manifest_papers: int,
    ) -> BuildIndexSummary:
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.shard_root.mkdir(parents=True, exist_ok=True)
        fingerprint = self._build_fingerprint(source_papers)
        release_signature = self._release_signature(
            all_manifest_papers=all_manifest_papers,
            terminal_parse_failures=terminal_parse_failures,
            total_manifest_papers=total_manifest_papers,
        )

        if mode == "rebuild":
            self._clear_workdirs()
        else:
            meta = self._load_meta()
            if meta:
                if meta.get("fingerprint") != fingerprint:
                    raise RuntimeError("Build work fingerprint changed. Re-run with --mode rebuild.")
                if (
                    meta.get("status") == "completed"
                    and meta.get("release_signature") == release_signature
                    and self._final_outputs_exist()
                ):
                    state = self.store.load_index_state()
                    logger.info("[bold blue]Build[/] | skipped reason=completed_index_matches_fingerprint")
                    return BuildIndexSummary.model_validate(state)

        self._save_meta(
            {
                "fingerprint": fingerprint,
                "release_signature": release_signature,
                "status": "running",
                "updated_at": now_iso(),
            }
        )
        self._prepare_bundles(source_papers)
        records = self._aggregate_records(source_papers)
        self._encode_all(records)
        summary = self._finalize(
            records,
            all_manifest_papers=all_manifest_papers,
            terminal_parse_failures=terminal_parse_failures,
            total_manifest_papers=total_manifest_papers,
        )
        self._save_meta(
            {
                "fingerprint": fingerprint,
                "release_signature": release_signature,
                "status": "completed",
                "built_at": summary.built_at,
                "updated_at": now_iso(),
            }
        )
        return summary

    def _prepare_bundles(self, source_papers: list[PaperRecord]) -> None:
        failure_map = self._load_prepare_failure_map()
        total = len(source_papers)
        completed = 0
        start_time = time.time()
        for paper in source_papers:
            bundle_path = self._bundle_path(paper.paper_id)
            if bundle_path.exists() or paper.paper_id in failure_map:
                completed += 1
                continue
            self.controller.check_pause_requested()
            try:
                bundle = self.indexer.parser.parse(paper)
                evidence_units = self.indexer.deep_chat_evidence_materializer.build(
                    papers=[bundle.paper],
                    sections=bundle.sections,
                    objects=bundle.objects,
                    chunks=bundle.chunks,
                )
                _atomic_write_text(
                    bundle_path,
                    json.dumps(
                        {
                            "paper": bundle.paper.model_dump(),
                            "sections": [section.model_dump() for section in bundle.sections],
                            "objects": [obj.model_dump() for obj in bundle.objects],
                            "chunks": [chunk.model_dump() for chunk in bundle.chunks],
                            "evidence_units": [unit.model_dump() for unit in evidence_units],
                        },
                        ensure_ascii=False,
                    ),
                )
            except Exception as exc:
                failure = self.indexer.parser.build_failure_record(paper, exc)
                self._append_prepare_failure(failure)
                logger.error("[bold blue]Build[/] | prepare_failed paper=%s error=%s", paper.paper_id, failure.error_message)
            completed += 1
            rate = completed / max(time.time() - start_time, 1e-6)
            eta_seconds = (total - completed) / rate if rate > 0 else None
            logger.info(
                "[bold blue]Build[/] | prepare_progress completed=%s total=%s percent=%.2f rate_papers_per_min=%.2f eta=%s",
                completed,
                total,
                (completed / total) * 100 if total else 100.0,
                rate * 60,
                _format_duration(eta_seconds),
            )
            self.controller.update_progress(
                phase="build_prepare",
                message=f"Preparing paper bundle: {paper.paper_id}",
                completed=completed,
                total=total,
                unit="papers",
                rate_per_min=rate * 60,
                eta_seconds=eta_seconds,
            )

    def _aggregate_records(self, source_papers: list[PaperRecord]) -> dict[str, list[Any]]:
        papers: list[PaperRecord] = []
        sections: list[SectionRecord] = []
        objects: list[ObjectRecord] = []
        chunks: list[ChunkRecord] = []
        evidence_units: list[EvidenceUnit] = []
        prepare_failures = list(self._load_prepare_failure_map().values())
        for paper in source_papers:
            bundle_path = self._bundle_path(paper.paper_id)
            if not bundle_path.exists():
                continue
            payload = json.loads(bundle_path.read_text(encoding="utf-8"))
            papers.append(PaperRecord.model_validate(payload["paper"]))
            sections.extend(SectionRecord.model_validate(item) for item in payload["sections"])
            objects.extend(ObjectRecord.model_validate(item) for item in payload["objects"])
            chunks.extend(ChunkRecord.model_validate(item) for item in payload["chunks"])
            evidence_units.extend(EvidenceUnit.model_validate(item) for item in payload["evidence_units"])
        return {
            "papers": papers,
            "sections": sections,
            "objects": objects,
            "chunks": chunks,
            "evidence_units": evidence_units,
            "prepare_failures": prepare_failures,
        }

    def _encode_all(self, records: dict[str, list[Any]]) -> None:
        papers = records["papers"]
        sections = records["sections"]
        chunks = records["chunks"]
        text_chunks = [chunk for chunk in chunks if chunk.chunk_type == "text_chunk"]
        table_chunks = [chunk for chunk in chunks if chunk.chunk_type == "table_chunk"]
        figure_chunks = [chunk for chunk in chunks if chunk.chunk_type == "figure_chunk"]
        evidence_units = records["evidence_units"]

        self._encode_dataset(
            dataset_name="paper",
            ids=[paper.paper_id for paper in papers],
            texts=[self.indexer._paper_search_text(paper) for paper in papers],
            encoder=self.indexer.paper_encoder,
            extras={},
        )
        self._encode_dataset(
            dataset_name="section",
            ids=[section.section_id for section in sections],
            texts=[self.indexer._section_search_text(section) for section in sections],
            encoder=self.indexer.chunk_encoder,
            extras={
                "paper_ids": [section.paper_id for section in sections],
                "section_titles": [section.section_title for section in sections],
                "section_paths": [section.section_path for section in sections],
            },
        )
        self._encode_dataset(
            dataset_name="chunk",
            ids=[chunk.chunk_id for chunk in chunks],
            texts=[self.indexer._chunk_search_text(chunk) for chunk in chunks],
            encoder=self.indexer.chunk_encoder,
            extras={
                "paper_ids": [chunk.paper_id for chunk in chunks],
                "section_ids": [chunk.section_id for chunk in chunks],
                "chunk_types": [chunk.chunk_type for chunk in chunks],
            },
        )
        self._encode_dataset(
            dataset_name="text_chunk",
            ids=[chunk.chunk_id for chunk in text_chunks],
            texts=[self.indexer._chunk_search_text(chunk) for chunk in text_chunks],
            encoder=self.indexer.chunk_encoder,
            extras={
                "paper_ids": [chunk.paper_id for chunk in text_chunks],
                "section_ids": [chunk.section_id for chunk in text_chunks],
            },
        )
        self._encode_dataset(
            dataset_name="table_chunk",
            ids=[chunk.chunk_id for chunk in table_chunks],
            texts=[self.indexer._chunk_search_text(chunk) for chunk in table_chunks],
            encoder=self.indexer.chunk_encoder,
            extras={
                "paper_ids": [chunk.paper_id for chunk in table_chunks],
                "section_ids": [chunk.section_id for chunk in table_chunks],
            },
        )
        self._encode_dataset(
            dataset_name="figure_chunk",
            ids=[chunk.chunk_id for chunk in figure_chunks],
            texts=[self.indexer._chunk_search_text(chunk) for chunk in figure_chunks],
            encoder=self.indexer.chunk_encoder,
            extras={
                "paper_ids": [chunk.paper_id for chunk in figure_chunks],
                "section_ids": [chunk.section_id for chunk in figure_chunks],
            },
        )
        self._encode_dataset(
            dataset_name="deep_chat_evidence",
            ids=[unit.evidence_id for unit in evidence_units],
            texts=[build_evidence_search_text(unit) for unit in evidence_units],
            encoder=self.indexer.chunk_encoder,
            extras={
                "paper_ids": [unit.paper_id for unit in evidence_units],
                "evidence_types": [unit.evidence_type for unit in evidence_units],
                "section_ids": [unit.section_id for unit in evidence_units],
            },
        )

    def _encode_dataset(
        self,
        *,
        dataset_name: str,
        ids: list[str],
        texts: list[str],
        encoder: SentenceTransformerEncoder,
        extras: dict[str, list[Any]],
    ) -> None:
        total = len(ids)
        shard_size = SHARD_SIZE_BY_DATASET[dataset_name]
        shard_dir = self.shard_root / dataset_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        display_name = ENCODE_LABELS.get(dataset_name, dataset_name)
        logger.info("[bold blue]Build[/] | encode_start dataset=%s total=%s", display_name, total)
        if total == 0:
            return
        encode_started_at = time.perf_counter()
        for shard_start in range(0, total, shard_size):
            shard_end = min(shard_start + shard_size, total)
            shard_index = shard_start // shard_size
            data_path = shard_dir / f"part-{shard_index:05d}.npz"
            meta_path = shard_dir / f"part-{shard_index:05d}.json"
            if data_path.exists() and meta_path.exists():
                meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_payload.get("start") == shard_start and meta_payload.get("end") == shard_end:
                    continue
            self.controller.check_pause_requested()

            def progress_callback(done_in_shard: int, shard_total: int) -> None:
                overall_done = shard_start + done_in_shard
                rate = overall_done / max(time.perf_counter() - encode_started_at, 1e-6)
                eta_seconds = (total - overall_done) / rate if rate > 0 else None
                logger.info(
                    "[bold blue]Build[/] | encode_progress dataset=%s completed=%s total=%s percent=%.2f rate_items_per_sec=%.2f eta=%s",
                    display_name,
                    overall_done,
                    total,
                    (overall_done / total) * 100 if total else 100.0,
                    rate,
                    _format_duration(eta_seconds),
                )
                self.controller.update_progress(
                    phase="build_encode",
                    message=f"Encoding {display_name}",
                    completed=overall_done,
                    total=total,
                    unit=f"{dataset_name}_records",
                    rate_per_min=rate * 60,
                    eta_seconds=eta_seconds,
                    extra={"dataset": dataset_name},
                )
                self.controller.check_pause_requested()

            shard_matrix = encoder.encode(
                texts[shard_start:shard_end],
                progress_callback=progress_callback,
            )
            shard_ids = ids[shard_start:shard_end]
            shard_extras = {key: values[shard_start:shard_end] for key, values in extras.items()}
            _atomic_save_npz(data_path, ids=np.array(shard_ids, dtype=object), matrix=shard_matrix)
            _atomic_write_text(
                meta_path,
                json.dumps(
                    {
                        "dataset": dataset_name,
                        "start": shard_start,
                        "end": shard_end,
                        "ids": shard_ids,
                        "texts": texts[shard_start:shard_end],
                        "tokens": [tokenize(text) for text in texts[shard_start:shard_end]],
                        "extras": shard_extras,
                        "encoder_backend": encoder.backend_name,
                        "encoder_model": encoder.model_name,
                        "vector_dim": int(shard_matrix.shape[1]) if shard_matrix.size else 0,
                        "built_at": now_iso(),
                    },
                    ensure_ascii=False,
                ),
            )
        logger.info("[bold blue]Build[/] | encode_done dataset=%s total=%s", display_name, total)

    def _finalize(
        self,
        records: dict[str, list[Any]],
        *,
        all_manifest_papers: list[PaperRecord],
        terminal_parse_failures: list[ParseFailureRecord],
        total_manifest_papers: int,
    ) -> BuildIndexSummary:
        self.controller.check_pause_requested()
        papers: list[PaperRecord] = records["papers"]
        sections: list[SectionRecord] = records["sections"]
        objects: list[ObjectRecord] = records["objects"]
        chunks: list[ChunkRecord] = records["chunks"]
        evidence_units: list[EvidenceUnit] = records["evidence_units"]
        build_prepare_failures: list[ParseFailureRecord] = records["prepare_failures"]
        merged_failures = list(terminal_parse_failures) + list(build_prepare_failures)

        text_chunks = [chunk for chunk in chunks if chunk.chunk_type == "text_chunk"]
        table_chunks = [chunk for chunk in chunks if chunk.chunk_type == "table_chunk"]
        figure_chunks = [chunk for chunk in chunks if chunk.chunk_type == "figure_chunk"]

        built_at = now_iso()
        snapshot_name = _snapshot_name(built_at)
        snapshot_root = self.settings.release_snapshots_dir / snapshot_name
        if snapshot_root.exists():
            shutil.rmtree(snapshot_root)
        snapshot_normalized = snapshot_root / "normalized"
        snapshot_deep_chat_normalized = snapshot_normalized / "deep_chat"
        snapshot_indexes = snapshot_root / "indexes"
        snapshot_layout_index = snapshot_indexes / "layout"
        snapshot_deep_chat_index = snapshot_indexes / "deep_chat"
        snapshot_normalized.mkdir(parents=True, exist_ok=True)
        snapshot_deep_chat_normalized.mkdir(parents=True, exist_ok=True)
        snapshot_layout_index.mkdir(parents=True, exist_ok=True)
        snapshot_deep_chat_index.mkdir(parents=True, exist_ok=True)

        paper_lookup = {paper.paper_id: paper for paper in all_manifest_papers}
        updated_papers = list(papers)
        existing_paper_ids = {paper.paper_id for paper in papers}
        for failure in merged_failures:
            source_paper = paper_lookup.get(failure.paper_id)
            if source_paper is None or failure.paper_id in existing_paper_ids:
                continue
            updated_papers.append(_failure_to_updated_paper(source_paper, failure))
        updated_papers.sort(key=lambda item: item.paper_id)

        _write_models_jsonl(snapshot_normalized / "papers.jsonl", updated_papers)
        _write_models_jsonl(snapshot_normalized / "sections.jsonl", sections)
        _write_models_jsonl(snapshot_normalized / "objects.jsonl", objects)
        _write_models_jsonl(snapshot_normalized / "chunks.jsonl", chunks)
        _write_models_jsonl(snapshot_normalized / "parse_failures.jsonl", merged_failures)
        _write_models_jsonl(snapshot_deep_chat_normalized / "evidence_units.jsonl", evidence_units)

        paper_ids, paper_vectors, paper_meta = self._merge_shards("paper")
        section_ids, section_vectors, section_meta = self._merge_shards("section")
        chunk_ids, chunk_vectors, chunk_meta = self._merge_shards("chunk")
        text_chunk_ids, text_chunk_vectors, text_chunk_meta = self._merge_shards("text_chunk")
        table_chunk_ids, table_chunk_vectors, table_chunk_meta = self._merge_shards("table_chunk")
        figure_chunk_ids, figure_chunk_vectors, figure_chunk_meta = self._merge_shards("figure_chunk")
        evidence_ids, evidence_vectors, evidence_meta = self._merge_shards("deep_chat_evidence")

        _save_index_payload(snapshot_layout_index, "paper", paper_ids, paper_vectors, paper_meta)
        _save_index_payload(snapshot_layout_index, "section", section_ids, section_vectors, section_meta)
        _save_index_payload(snapshot_layout_index, "chunk", chunk_ids, chunk_vectors, chunk_meta)
        _save_index_payload(snapshot_layout_index, "text_chunk", text_chunk_ids, text_chunk_vectors, text_chunk_meta)
        _save_index_payload(snapshot_layout_index, "table_chunk", table_chunk_ids, table_chunk_vectors, table_chunk_meta)
        _save_index_payload(snapshot_layout_index, "figure_chunk", figure_chunk_ids, figure_chunk_vectors, figure_chunk_meta)
        _save_index_payload(snapshot_deep_chat_index, "evidence_unit", evidence_ids, evidence_vectors, evidence_meta)

        parser_backend_counts: dict[str, int] = defaultdict(int)
        for paper in papers:
            parser_backend_counts[paper.parser_backend or self.settings.pdf_parser_backend] += 1
        parse_failure_counts: dict[str, int] = defaultdict(int)
        for failure in merged_failures:
            parse_failure_counts[failure.error_type] += 1

        index_state = {
            "built_at": built_at,
            "total_papers": total_manifest_papers,
            "papers": len(papers),
            "indexed_papers": len(papers),
            "failed_papers": len(merged_failures),
            "sections": len(sections),
            "objects": len(objects),
            "chunks": len(chunks),
            "text_chunks": len(text_chunks),
            "table_chunks": len(table_chunks),
            "figure_chunks": len(figure_chunks),
            "deep_chat_evidence_units": len(evidence_units),
            "paper_dense_backend": paper_meta.get("encoder_backend"),
            "chunk_dense_backend": chunk_meta.get("encoder_backend"),
            "paper_dense_model": paper_meta.get("encoder_model"),
            "chunk_dense_model": chunk_meta.get("encoder_model"),
            "paper_vector_dim": int(paper_vectors.shape[1]) if paper_vectors.size else 0,
            "chunk_vector_dim": int(chunk_vectors.shape[1]) if chunk_vectors.size else 0,
            "pdf_parser_backend": self.settings.pdf_parser_backend,
            "parser_backend_counts": dict(sorted(parser_backend_counts.items())),
            "parse_failure_counts": dict(sorted(parse_failure_counts.items())),
            "parse_failure_path": str(self.settings.normalized_dir / "parse_failures.jsonl"),
        }
        _atomic_write_text(snapshot_layout_index / "index_state.json", json.dumps(index_state, ensure_ascii=False, indent=2))

        self.controller.update_state(status="running", phase="build_finalize", message="Publishing the new release index snapshot.")
        _publish_release_snapshot(snapshot_root=snapshot_root, current_link=self.settings.current_release_path)

        return BuildIndexSummary(
            papers=len(papers),
            total_papers=total_manifest_papers,
            indexed_papers=len(papers),
            failed_papers=len(merged_failures),
            sections=len(sections),
            objects=len(objects),
            chunks=len(chunks),
            text_chunks=len(text_chunks),
            table_chunks=len(table_chunks),
            figure_chunks=len(figure_chunks),
            deep_chat_evidence_units=len(evidence_units),
            paper_vector_dim=int(paper_vectors.shape[1]) if paper_vectors.size else 0,
            chunk_vector_dim=int(chunk_vectors.shape[1]) if chunk_vectors.size else 0,
            paper_dense_backend=str(paper_meta.get("encoder_backend")),
            chunk_dense_backend=str(chunk_meta.get("encoder_backend")),
            pdf_parser_backend=self.settings.pdf_parser_backend,
            parser_backend_counts=dict(sorted(parser_backend_counts.items())),
            parse_failure_counts=dict(sorted(parse_failure_counts.items())),
            built_at=built_at,
        )

    def _merge_shards(self, dataset_name: str) -> tuple[list[str], np.ndarray, dict[str, Any]]:
        shard_dir = self.shard_root / dataset_name
        ids: list[str] = []
        texts: list[str] = []
        tokens: list[list[str]] = []
        matrices: list[np.ndarray] = []
        extras: dict[str, list[Any]] = defaultdict(list)
        encoder_backend = None
        encoder_model = None
        for meta_path in sorted(shard_dir.glob("part-*.json")):
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            data = np.load(meta_path.with_suffix(".npz"), allow_pickle=True)
            ids.extend(str(item) for item in data["ids"].tolist())
            texts.extend(payload["texts"])
            tokens.extend(payload["tokens"])
            matrices.append(np.array(data["matrix"], dtype=np.float32))
            for key, values in payload.get("extras", {}).items():
                extras[key].extend(values)
            encoder_backend = payload.get("encoder_backend")
            encoder_model = payload.get("encoder_model")
        matrix = np.vstack(matrices).astype(np.float32, copy=False) if matrices else np.empty((0, 0), dtype=np.float32)
        meta = {
            "ids": ids,
            "texts": texts,
            "tokens": tokens,
            "encoder_backend": encoder_backend,
            "encoder_model": encoder_model,
            "vector_dim": int(matrix.shape[1]) if matrix.size else 0,
            "built_at": now_iso(),
        }
        meta.update(extras)
        return ids, matrix, meta

    def _load_prepare_failure_map(self) -> dict[str, ParseFailureRecord]:
        if not self.prepare_failure_path.exists():
            return {}
        output: dict[str, ParseFailureRecord] = {}
        for line in self.prepare_failure_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            record = ParseFailureRecord.model_validate_json(line)
            output[record.paper_id] = record
        return output

    def _append_prepare_failure(self, failure: ParseFailureRecord) -> None:
        records = self._load_prepare_failure_map()
        records[failure.paper_id] = failure
        self.prepare_failure_path.parent.mkdir(parents=True, exist_ok=True)
        with self.prepare_failure_path.open("w", encoding="utf-8") as handle:
            for record in records.values():
                handle.write(record.model_dump_json())
                handle.write("\n")

    def _bundle_path(self, paper_id: str) -> Path:
        return self.bundle_dir / f"{paper_id}.json"

    def _build_fingerprint(self, source_papers: list[PaperRecord]) -> str:
        payload = {
            "version": BUILD_FINGERPRINT_VERSION,
            "corpus": self.settings.corpus.to_dict(),
            "papers": [_paper_build_signature(paper) for paper in source_papers],
            "pdf_parser_backend": self.settings.pdf_parser_backend,
            "paper_dense_model": self.settings.paper_dense_model,
            "chunk_dense_model": self.settings.chunk_dense_model,
            "chunk_target_tokens": self.settings.chunk_target_tokens,
            "chunk_overlap_tokens": self.settings.chunk_overlap_tokens,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _release_signature(
        self,
        *,
        all_manifest_papers: list[PaperRecord],
        terminal_parse_failures: list[ParseFailureRecord],
        total_manifest_papers: int,
    ) -> str:
        payload = {
            "version": "offline_release_v1",
            "corpus": self.settings.corpus.to_dict(),
            "total_manifest_papers": total_manifest_papers,
            "manifest_papers": [_paper_build_signature(paper) for paper in all_manifest_papers],
            "terminal_parse_failures": [
                {
                    "paper_id": failure.paper_id,
                    "stage": failure.stage,
                    "error_type": failure.error_type,
                    "error_message": failure.error_message,
                    "parser_backend": failure.parser_backend,
                }
                for failure in sorted(terminal_parse_failures, key=lambda item: item.paper_id)
            ],
            "pdf_parser_backend": self.settings.pdf_parser_backend,
            "paper_dense_model": self.settings.paper_dense_model,
            "chunk_dense_model": self.settings.chunk_dense_model,
            "chunk_target_tokens": self.settings.chunk_target_tokens,
            "chunk_overlap_tokens": self.settings.chunk_overlap_tokens,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _clear_workdirs(self) -> None:
        for path in (
            self.bundle_dir.parent,
        ):
            if path.exists():
                shutil.rmtree(path)

    def _load_meta(self) -> dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _save_meta(self, payload: dict[str, Any]) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.meta_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _final_outputs_exist(self) -> bool:
        return self.settings.current_release_path.is_symlink() and all(
            path.exists()
            for path in (
                self.settings.normalized_dir / "papers.jsonl",
                self.settings.index_dir / "paper_vectors.npz",
                self.settings.index_dir / "index_state.json",
                self.settings.deep_chat_index_dir / "evidence_unit_vectors.npz",
            )
        )


class OfflineRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = LocalStore(settings)

    def run(self, *, mode: str) -> OfflineRunResult:
        controller = OfflineJobController(self.settings, mode=mode)
        controller.start()
        try:
            controller.update_state(status="running", phase="manifest_sync", message="Syncing corpus manifest and PDFs.")
            track_arg = ["all"] if self.settings.corpus.track == "all" else [self.settings.corpus.track]
            summary = ACLAnthologyIngestor(self.settings, self.store).ingest_event(
                venue=self.settings.corpus.venue,
                year=self.settings.corpus.year,
                tracks=track_arg,
                download_pdfs=True,
            )
            source_papers = self.store.load_source_papers()
            self._ensure_local_pdfs(source_papers)
            if mode == "rebuild":
                controller.update_state(status="running", phase="rebuild_cleanup", message="Cleaning parse/build artifacts for this corpus.")
                self._clear_for_rebuild(source_papers)

            parse_success, parse_failures = self._collect_parse_status(source_papers)
            if len(parse_success) + len(parse_failures) < len(source_papers):
                controller.update_state(status="running", phase="parse", message="Running MinerU parse.")
                run_mineru_pipeline(settings=self.settings, papers=source_papers, controller=controller)
                parse_success, parse_failures = self._collect_parse_status(source_papers)

            pending = len(source_papers) - len(parse_success) - len(parse_failures)
            if pending > 0:
                raise RuntimeError(f"Parse ended with {pending} paper(s) neither completed nor recorded as failed.")

            controller.update_state(status="running", phase="build", message="Building resumable indexes.")
            builder = IncrementalIndexBuilder(self.settings, controller)
            build_summary = builder.build(
                source_papers=parse_success,
                all_manifest_papers=source_papers,
                terminal_parse_failures=parse_failures,
                mode=mode,
                total_manifest_papers=len(source_papers),
            )
            controller.update_state(
                status="running",
                phase="refresh_search_current",
                message="Refreshing the online search collection at data/search_current.",
            )
            search_current_manifest = rebuild_search_current(
                self.settings.root_dir,
                corpora=[self.settings.corpus],
                allow_uncompleted_selected=True,
            )
            controller.write_active_corpus()
            controller.mark_completed(
                message="Offline job completed and the online search collection was refreshed.",
                extra={
                    "build_summary": build_summary.model_dump(),
                    "search_current_manifest": search_current_manifest,
                },
            )
            return OfflineRunResult(
                status="completed",
                corpus=self.settings.corpus.key,
                message=(
                    f"Offline job completed: fetched={summary.fetched_papers} "
                    f"downloaded={summary.downloaded_pdfs} indexed={build_summary.indexed_papers}"
                ),
                build_summary=build_summary.model_dump(),
            )
        except PauseRequested as exc:
            controller.mark_paused(str(exc))
            return OfflineRunResult(status="paused", corpus=self.settings.corpus.key, message=str(exc))
        except Exception as exc:
            controller.mark_failed(str(exc))
            raise
        finally:
            controller.close()

    def _clear_for_rebuild(self, source_papers: list[PaperRecord]) -> None:
        paper_ids = [paper.paper_id for paper in source_papers]
        remove_artifacts(self.settings.mineru_output_dir, paper_ids)
        remove_failure_entries(self.settings.mineru_failure_manifest_path, set(paper_ids))
        for path in (
            self.settings.work_dir,
        ):
            if path.exists():
                shutil.rmtree(path)

    def _ensure_local_pdfs(self, source_papers: list[PaperRecord]) -> None:
        missing = [
            paper.paper_id
            for paper in source_papers
            if not paper.local_pdf_path or not Path(paper.local_pdf_path).exists()
        ]
        if missing:
            preview = ", ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            raise RuntimeError(f"These papers are missing local PDFs and cannot be parsed: {preview}{suffix}")

    def _collect_parse_status(
        self,
        source_papers: list[PaperRecord],
    ) -> tuple[list[PaperRecord], list[ParseFailureRecord]]:
        success: list[PaperRecord] = []
        for paper in source_papers:
            if artifact_complete(self.settings.mineru_output_dir, paper.paper_id):
                success.append(paper)
        failures = normalize_failure_entries(self.settings, source_papers)
        failure_ids = {failure.paper_id for failure in failures}
        success = [paper for paper in success if paper.paper_id not in failure_ids]
        return success, failures


class OfflineEnrichmentRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = LocalStore(settings)
        self.parser = PDFParser(settings)
        self.grobid = GrobidHeaderClient(settings)

    def run(self, *, mode: str) -> OfflineEnrichmentResult:
        controller = OfflineJobController(self.settings, mode=f"enrich:{mode}")
        controller.start()
        try:
            controller.update_state(status="running", phase="manifest_load", message="Loading the current corpus manifest.")
            source_papers = self.store.load_source_papers()
            if not source_papers:
                raise RuntimeError("The current corpus has no source papers, so enrichment cannot run.")
            self._ensure_local_pdfs(source_papers)

            parse_success, parse_failures = self._collect_parse_status(source_papers)
            pending = len(source_papers) - len(parse_success) - len(parse_failures)
            if pending > 0:
                raise RuntimeError(
                    f"The current corpus still has {pending} paper(s) without completed MinerU parse. Run offline-run before offline-enrich."
                )

            if mode == "rebuild":
                controller.update_state(
                    status="running",
                    phase="enrich_cleanup",
                    message="Cleaning enrichment cache for this corpus.",
                )
                self._clear_for_rebuild(parse_success)

            total = len(parse_success)
            processed = 0
            skipped = 0
            failed = 0
            start_time = time.time()

            for paper in parse_success:
                controller.check_pause_requested()
                if mode == "resume" and self._enrichment_complete(paper.paper_id):
                    skipped += 1
                    processed += 1
                    self._emit_progress(
                        controller=controller,
                        paper_id=paper.paper_id,
                        processed=processed,
                        total=total,
                        started_at=start_time,
                    )
                    continue

                try:
                    if not paper.local_pdf_path:
                        raise RuntimeError(f"Paper is missing local_pdf_path: {paper.paper_id}")
                    header_result = self.grobid.extract_header(Path(paper.local_pdf_path))
                    markdown_path = self.parser.resolve_artifacts(paper.paper_id).markdown_path
                    markdown_text = ""
                    if markdown_path is not None and markdown_path.exists():
                        markdown_text = markdown_path.read_text(encoding="utf-8", errors="replace")
                    references = extract_reference_entries_from_markdown(markdown_text)

                    if header_result.title and _normalized_title_key(header_result.title) != _normalized_title_key(paper.title):
                        logger.warning(
                            "[bold yellow]Enrichment[/] | grobid_title_mismatch paper=%s manifest=%r grobid=%r",
                            paper.paper_id,
                            paper.title,
                            header_result.title,
                        )

                    save_cached_paper_enrichment_record(
                        self.settings,
                        PaperEnrichmentRecord(
                            paper_id=paper.paper_id,
                            affiliations=header_result.affiliations,
                            authors_structured=header_result.authors_structured,
                            reference_count=len(references),
                            updated_at=now_iso(),
                        ),
                    )
                    save_cached_paper_references(
                        self.settings,
                        PaperReferencesRecord(
                            paper_id=paper.paper_id,
                            references=references,
                            updated_at=now_iso(),
                        ),
                    )
                except Exception as exc:
                    failed += 1
                    logger.exception("[bold yellow]Enrichment[/] | failed paper=%s error=%r", paper.paper_id, exc)
                processed += 1
                self._emit_progress(
                    controller=controller,
                    paper_id=paper.paper_id,
                    processed=processed,
                    total=total,
                    started_at=start_time,
                )

            summary = {
                "total": total,
                "processed": processed,
                "skipped": skipped,
                "failed": failed,
                "parse_failures": len(parse_failures),
                "updated_at": now_iso(),
            }
            controller.mark_completed(
                message="Offline enrichment completed.",
                extra={"enrichment_summary": summary},
            )
            return OfflineEnrichmentResult(
                status="completed",
                corpus=self.settings.corpus.key,
                message=(
                    f"Offline enrichment completed: processed={processed} skipped={skipped} "
                    f"failed={failed} parse_failures={len(parse_failures)}"
                ),
                enrichment_summary=summary,
            )
        except PauseRequested as exc:
            controller.mark_paused(str(exc))
            return OfflineEnrichmentResult(status="paused", corpus=self.settings.corpus.key, message=str(exc))
        except Exception as exc:
            controller.mark_failed(str(exc))
            raise
        finally:
            controller.close()

    def _ensure_local_pdfs(self, source_papers: list[PaperRecord]) -> None:
        missing = [
            paper.paper_id
            for paper in source_papers
            if not paper.local_pdf_path or not Path(paper.local_pdf_path).exists()
        ]
        if missing:
            preview = ", ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            raise RuntimeError(f"These papers are missing local PDFs and cannot be enriched: {preview}{suffix}")

    def _collect_parse_status(
        self,
        source_papers: list[PaperRecord],
    ) -> tuple[list[PaperRecord], list[ParseFailureRecord]]:
        success: list[PaperRecord] = []
        for paper in source_papers:
            if artifact_complete(self.settings.mineru_output_dir, paper.paper_id):
                success.append(paper)
        failures = normalize_failure_entries(self.settings, source_papers)
        failure_ids = {failure.paper_id for failure in failures}
        success = [paper for paper in success if paper.paper_id not in failure_ids]
        return success, failures

    def _enrichment_complete(self, paper_id: str) -> bool:
        paper_path = self.settings.data_dir / "enrichment" / "papers" / f"{paper_id}.json"
        references_path = self.settings.data_dir / "enrichment" / "references" / f"{paper_id}.json"
        return paper_path.exists() and references_path.exists()

    def _clear_for_rebuild(self, source_papers: list[PaperRecord]) -> None:
        for paper in source_papers:
            paper_path = self.settings.data_dir / "enrichment" / "papers" / f"{paper.paper_id}.json"
            references_path = self.settings.data_dir / "enrichment" / "references" / f"{paper.paper_id}.json"
            paper_path.unlink(missing_ok=True)
            references_path.unlink(missing_ok=True)

    def _emit_progress(
        self,
        *,
        controller: OfflineJobController,
        paper_id: str,
        processed: int,
        total: int,
        started_at: float,
    ) -> None:
        elapsed = max(time.time() - started_at, 1e-6)
        rate = processed / elapsed
        eta_seconds = (total - processed) / rate if rate > 0 else None
        logger.info(
            "[bold yellow]Enrichment[/] | progress completed=%s total=%s percent=%.2f rate_papers_per_min=%.2f eta=%s last=%s",
            processed,
            total,
            (processed / total) * 100 if total else 100.0,
            rate * 60,
            _format_duration(eta_seconds),
            paper_id,
        )
        controller.update_progress(
            phase="enrich",
            message=f"Writing paper enrichment: {paper_id}",
            completed=processed,
            total=total,
            unit="papers",
            rate_per_min=rate * 60,
            eta_seconds=eta_seconds,
        )


def _normalized_title_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def request_pause(settings: Settings) -> dict[str, Any]:
    if not settings.active_job_path.exists():
        return {"status": "idle", "message": "There is no active offline job."}
    payload = json.loads(settings.active_job_path.read_text(encoding="utf-8"))
    pid = int(payload.get("pid") or 0)
    if pid <= 0 or not _pid_exists(pid):
        settings.active_job_path.unlink(missing_ok=True)
        return {"status": "idle", "message": "The active job record was stale and has been cleared."}
    control_path = Path(str(payload["control_path"]))
    _atomic_write_text(
        control_path,
        json.dumps({"pause_requested": True, "requested_at": now_iso()}, ensure_ascii=False, indent=2),
    )
    return {"status": "pause_requested", "job_id": payload.get("job_id"), "corpus": payload.get("corpus")}


def render_status(settings: Settings) -> None:
    console = Console()
    active_job = json.loads(settings.active_job_path.read_text(encoding="utf-8")) if settings.active_job_path.exists() else None
    if active_job is not None:
        active_pid = int(active_job.get("pid") or 0)
        if active_pid <= 0 or not _pid_exists(active_pid):
            settings.active_job_path.unlink(missing_ok=True)
            active_job = None
    last_job = json.loads(settings.last_job_path.read_text(encoding="utf-8")) if settings.last_job_path.exists() else None
    job_descriptor = active_job or last_job
    state_path = Path(str(job_descriptor["state_path"])) if job_descriptor else settings.state_dir / "job_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if active_job is None and last_job is not None:
        last_pid = int(last_job.get("pid") or 0)
        if last_pid > 0 and not _pid_exists(last_pid) and str(state.get("status") or "") == "running":
            state = {
                **state,
                "status": "stale",
                "phase": str(state.get("phase") or "unknown"),
                "message": "The latest offline job process no longer exists; the state file was not finalized.",
            }
    table = Table(box=box.ROUNDED, show_header=False, expand=True)
    table.add_column("Field", style="bold cyan", width=18)
    table.add_column("Value", style="white")
    descriptor_corpus = job_descriptor.get("corpus") if job_descriptor else None
    if descriptor_corpus:
        corpus_label = f"{descriptor_corpus['venue']}/{descriptor_corpus['year']}/{descriptor_corpus['track']}"
    else:
        corpus_label = settings.corpus.key
    table.add_row("Current corpus", corpus_label)
    table.add_row("Active job", str(active_job.get("job_id")) if active_job else "none")
    table.add_row("Status", str(state.get("status") or "idle"))
    table.add_row("Phase", str(state.get("phase") or "none"))
    table.add_row("Message", str(state.get("message") or "none"))
    progress = state.get("progress") or {}
    if progress and str(state.get("status") or "idle") in {"running", "paused"}:
        table.add_row("Progress", f"{progress.get('completed')}/{progress.get('total')} ({float(progress.get('percent') or 0.0):.2f}%)")
        table.add_row("Rate", f"{float(progress.get('rate_per_min') or 0.0):.2f}/min")
        table.add_row("ETA", str(progress.get("eta") or "unknown"))
    active_corpus = (
        json.loads(settings.active_corpus_path.read_text(encoding="utf-8"))
        if settings.active_corpus_path.exists()
        else None
    )
    if active_corpus:
        table.add_row("Online corpus", f"{active_corpus['venue']}/{active_corpus['year']}/{active_corpus['track']}")
    console.print(Panel(table, title="Offline Status", border_style="green"))


def _failure_to_updated_paper(source_paper: PaperRecord, failure: ParseFailureRecord) -> PaperRecord:
    return source_paper.model_copy(
        deep=True,
        update={
            "parser_backend": failure.parser_backend,
            "text": "",
            "intro_summary": "",
            "section_headings": [],
            "sections": [],
            "section_ids": [],
            "object_ids": [],
            "chunk_ids": [],
            "typed_evidence_summary": {},
            "metadata": {
                **source_paper.metadata,
                "parse_status": "failed",
                "pdf_parser_backend": failure.parser_backend,
                "parse_failure_stage": failure.stage,
                "parse_failure_error_type": failure.error_type,
                "parse_failure_message": failure.error_message,
            },
        },
    )


def _paper_build_signature(paper: PaperRecord) -> dict[str, Any]:
    pdf_signature: dict[str, Any] | None = None
    local_pdf_path = Path(str(paper.local_pdf_path)) if paper.local_pdf_path else None
    if local_pdf_path is not None and local_pdf_path.exists():
        stat = local_pdf_path.stat()
        pdf_signature = {
            "path": str(local_pdf_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return {
        "paper_id": paper.paper_id,
        "anthology_id": paper.anthology_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": list(paper.authors),
        "venue": paper.venue,
        "year": paper.year,
        "track": paper.track,
        "volume_id": paper.volume_id,
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "keywords": list(paper.keywords),
        "local_pdf": pdf_signature,
    }


def _write_models_jsonl(path: Path, items: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        temp_path = Path(handle.name)
        for item in items:
            if hasattr(item, "model_dump_json"):
                handle.write(item.model_dump_json())
            else:
                handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")
    temp_path.replace(path)


def _save_index_payload(root: Path, name: str, ids: list[str], matrix: np.ndarray, meta: dict[str, Any]) -> None:
    _atomic_save_npz(root / f"{name}_vectors.npz", ids=np.array(ids, dtype=object), matrix=matrix)
    _atomic_write_text(root / f"{name}_index_meta.json", json.dumps(meta, ensure_ascii=False, indent=2))


def _publish_release_snapshot(*, snapshot_root: Path, current_link: Path) -> None:
    current_link.parent.mkdir(parents=True, exist_ok=True)
    if current_link.exists() and not current_link.is_symlink():
        raise RuntimeError(
            f"Release pointer path is not a symlink: {current_link}. "
            "Remove the legacy directory before running unified offline build."
        )
    temp_link = current_link.with_name(f"{current_link.name}.tmp")
    temp_link.unlink(missing_ok=True)
    relative_target = os.path.relpath(snapshot_root, start=current_link.parent)
    os.symlink(relative_target, temp_link)
    os.replace(temp_link, current_link)


def _snapshot_name(built_at: str) -> str:
    compact = built_at.replace("-", "").replace(":", "").replace("+", "_").replace("T", "_")
    return compact.replace(".", "_")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", suffix=".npz", delete=False, dir=path.parent) as handle:
        temp_path = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    temp_path.replace(path)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    total_seconds = max(0, int(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
