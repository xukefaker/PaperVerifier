from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings
from .deep_chat.evidence import DeepChatEvidenceMaterializer, build_evidence_search_text
from .deep_chat.store import DeepChatStore
from .encoders import EncoderConfig, SentenceTransformerEncoder
from .models import BuildIndexSummary, ChunkRecord, ObjectRecord, PaperRecord, ParseFailureRecord, SectionRecord
from .pdf_parser import PDFParser
from .storage import LocalStore
from .utils import now_iso, tokenize

logger = logging.getLogger(__name__)

ENCODE_LABELS = {
    "paper": "paper vectors",
    "section": "section vectors",
    "chunk": "chunk vectors",
    "text_chunk": "text chunk vectors",
    "table_chunk": "table chunk vectors",
    "figure_chunk": "figure chunk vectors",
    "deep_chat_evidence": "deep-chat evidence vectors",
}


class IndexBuilder:
    def __init__(
        self,
        settings: Settings,
        store: LocalStore,
        *,
        cancel_check: Callable[[], None] | None = None,
        progress: Any | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.cancel_check = cancel_check
        self.progress = progress
        self.deep_chat_store = DeepChatStore(settings)
        self.parser = PDFParser(settings)
        self.deep_chat_evidence_materializer = DeepChatEvidenceMaterializer()
        self.paper_encoder = SentenceTransformerEncoder(
            EncoderConfig(
                settings.paper_dense_model,
                device=settings.dense_device,
                batch_size=settings.dense_batch_size,
            )
        )
        self.chunk_encoder = SentenceTransformerEncoder(
            EncoderConfig(
                settings.chunk_dense_model,
                device=settings.dense_device,
                batch_size=settings.dense_batch_size,
            )
        )

    def build(
        self,
        *,
        max_papers: int | None = None,
        paper_ids: list[str] | None = None,
    ) -> BuildIndexSummary:
        source_papers = self._select_source_papers(max_papers=max_papers, paper_ids=paper_ids)
        parse_stage_start = time.perf_counter()
        updated_papers: list[PaperRecord] = []
        parsed_papers: list[PaperRecord] = []
        sections: list[SectionRecord] = []
        objects: list[ObjectRecord] = []
        chunks: list[ChunkRecord] = []
        parse_failures: list[ParseFailureRecord] = []
        parser_backend_counts: dict[str, int] = defaultdict(int)
        parse_failure_counts: dict[str, int] = defaultdict(int)

        total_papers = len(source_papers)
        if self.progress is not None:
            self.progress.index_parse_start(total=total_papers)
        for index, paper in enumerate(source_papers, start=1):
            self._check_cancel()
            try:
                bundle = self.parser.parse(paper)
            except Exception as exc:
                failure = self.parser.build_failure_record(paper, exc)
                failed_paper = paper.model_copy(
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
                            **paper.metadata,
                            "parse_status": "failed",
                            "pdf_parser_backend": failure.parser_backend,
                            "parse_failure_stage": failure.stage,
                            "parse_failure_error_type": failure.error_type,
                            "parse_failure_message": failure.error_message,
                        },
                    },
                )
                updated_papers.append(failed_paper)
                parse_failures.append(failure)
                parse_failure_counts[failure.error_type] += 1
                logger.warning(
                    "Parse failed for %s (%s/%s/%s): %s",
                    paper.paper_id,
                    paper.venue,
                    paper.year,
                    paper.track or "unknown",
                    failure.error_message,
                )
            else:
                updated_papers.append(bundle.paper)
                parsed_papers.append(bundle.paper)
                sections.extend(bundle.sections)
                objects.extend(bundle.objects)
                chunks.extend(bundle.chunks)
                backend_name = bundle.paper.parser_backend or self.settings.pdf_parser_backend
                parser_backend_counts[backend_name] += 1

            if self.progress is not None:
                self.progress.index_parse_update(
                    completed=index,
                    total=total_papers,
                    indexed=len(parsed_papers),
                    failed=len(parse_failures),
                    sections=len(sections),
                    chunks=len(chunks),
                )
            elif index % 25 == 0 or index == total_papers:
                elapsed = time.perf_counter() - parse_stage_start
                rate = index / elapsed if elapsed > 0 else 0.0
                remaining = max(total_papers - index, 0)
                eta_seconds = remaining / rate if rate > 0 else None
                logger.info(
                    "[bold blue]Index[/] | parse_progress completed=%s total=%s percent=%.2f indexed=%s failed=%s sections=%s objects=%s chunks=%s rate_papers_per_min=%.2f eta=%s",
                    index,
                    total_papers,
                    (index / total_papers) * 100 if total_papers else 100.0,
                    len(parsed_papers),
                    len(parse_failures),
                    len(sections),
                    len(objects),
                    len(chunks),
                    rate * 60,
                    self._format_duration(eta_seconds),
                )

        self._check_cancel()
        if self.progress is not None:
            self.progress.prepare("building section, chunk, and evidence records")
        text_chunks = [chunk for chunk in chunks if chunk.chunk_type == "text_chunk"]
        table_chunks = [chunk for chunk in chunks if chunk.chunk_type == "table_chunk"]
        figure_chunks = [chunk for chunk in chunks if chunk.chunk_type == "figure_chunk"]
        deep_chat_evidence_units = self.deep_chat_evidence_materializer.build(
            papers=parsed_papers,
            sections=sections,
            objects=objects,
            chunks=chunks,
        )

        paper_texts = [self._paper_search_text(paper) for paper in parsed_papers]
        section_texts = [self._section_search_text(section) for section in sections]
        all_chunk_texts = [self._chunk_search_text(chunk) for chunk in chunks]
        text_chunk_texts = [self._chunk_search_text(chunk) for chunk in text_chunks]
        table_chunk_texts = [self._chunk_search_text(chunk) for chunk in table_chunks]
        figure_chunk_texts = [self._chunk_search_text(chunk) for chunk in figure_chunks]
        deep_chat_evidence_texts = [build_evidence_search_text(unit) for unit in deep_chat_evidence_units]

        self._check_cancel()
        paper_vectors = self._encode(self.paper_encoder, paper_texts, "paper")
        section_vectors = self._encode(self.chunk_encoder, section_texts, "section")
        chunk_vectors = self._encode(self.chunk_encoder, all_chunk_texts, "chunk")
        text_chunk_vectors = self._encode(self.chunk_encoder, text_chunk_texts, "text_chunk")
        table_chunk_vectors = self._encode(self.chunk_encoder, table_chunk_texts, "table_chunk")
        figure_chunk_vectors = self._encode(self.chunk_encoder, figure_chunk_texts, "figure_chunk")
        deep_chat_evidence_vectors = self._encode(
            self.chunk_encoder,
            deep_chat_evidence_texts,
            "deep_chat_evidence",
        )

        self._check_cancel()
        if self.progress is not None:
            self.progress.save_start()
        self.store.save_papers(updated_papers)
        self.store.save_sections(sections)
        self.store.save_objects(objects)
        self.store.save_chunks(chunks)
        self.store.save_parse_failures(parse_failures)
        self.deep_chat_store.save_evidence_units(deep_chat_evidence_units)

        self._save_index_payload(
            "paper",
            [paper.paper_id for paper in parsed_papers],
            paper_texts,
            paper_vectors,
            encoder=self.paper_encoder,
        )
        self._save_index_payload(
            "section",
            [section.section_id for section in sections],
            section_texts,
            section_vectors,
            encoder=self.chunk_encoder,
            extra={
                "paper_ids": [section.paper_id for section in sections],
                "section_titles": [section.section_title for section in sections],
                "section_paths": [section.section_path for section in sections],
            },
        )
        self._save_index_payload(
            "chunk",
            [chunk.chunk_id for chunk in chunks],
            all_chunk_texts,
            chunk_vectors,
            encoder=self.chunk_encoder,
            extra={
                "paper_ids": [chunk.paper_id for chunk in chunks],
                "section_ids": [chunk.section_id for chunk in chunks],
                "chunk_types": [chunk.chunk_type for chunk in chunks],
            },
        )
        self._save_index_payload(
            "text_chunk",
            [chunk.chunk_id for chunk in text_chunks],
            text_chunk_texts,
            text_chunk_vectors,
            encoder=self.chunk_encoder,
            extra={
                "paper_ids": [chunk.paper_id for chunk in text_chunks],
                "section_ids": [chunk.section_id for chunk in text_chunks],
            },
        )
        self._save_index_payload(
            "table_chunk",
            [chunk.chunk_id for chunk in table_chunks],
            table_chunk_texts,
            table_chunk_vectors,
            encoder=self.chunk_encoder,
            extra={
                "paper_ids": [chunk.paper_id for chunk in table_chunks],
                "section_ids": [chunk.section_id for chunk in table_chunks],
            },
        )
        self._save_index_payload(
            "figure_chunk",
            [chunk.chunk_id for chunk in figure_chunks],
            figure_chunk_texts,
            figure_chunk_vectors,
            encoder=self.chunk_encoder,
            extra={
                "paper_ids": [chunk.paper_id for chunk in figure_chunks],
                "section_ids": [chunk.section_id for chunk in figure_chunks],
            },
        )
        self.deep_chat_store.save_vectors(
            [unit.evidence_id for unit in deep_chat_evidence_units],
            deep_chat_evidence_vectors,
        )
        self.deep_chat_store.save_index_meta(
            {
                "ids": [unit.evidence_id for unit in deep_chat_evidence_units],
                "texts": deep_chat_evidence_texts,
                "tokens": [tokenize(text) for text in deep_chat_evidence_texts],
                "paper_ids": [unit.paper_id for unit in deep_chat_evidence_units],
                "evidence_types": [unit.evidence_type for unit in deep_chat_evidence_units],
                "section_ids": [unit.section_id for unit in deep_chat_evidence_units],
                "encoder_backend": self.chunk_encoder.backend_name,
                "encoder_model": self.chunk_encoder.model_name,
                "vector_dim": int(deep_chat_evidence_vectors.shape[1]) if deep_chat_evidence_vectors.size else 0,
                "built_at": now_iso(),
            }
        )
        if self.progress is not None:
            self.progress.save_done()

        built_at = now_iso()
        state = {
            "built_at": built_at,
            "total_papers": len(updated_papers),
            "papers": len(parsed_papers),
            "indexed_papers": len(parsed_papers),
            "failed_papers": len(parse_failures),
            "sections": len(sections),
            "objects": len(objects),
            "chunks": len(chunks),
            "text_chunks": len(text_chunks),
            "table_chunks": len(table_chunks),
            "figure_chunks": len(figure_chunks),
            "deep_chat_evidence_units": len(deep_chat_evidence_units),
            "paper_dense_backend": self.paper_encoder.backend_name,
            "chunk_dense_backend": self.chunk_encoder.backend_name,
            "paper_dense_model": self.paper_encoder.model_name,
            "chunk_dense_model": self.chunk_encoder.model_name,
            "paper_vector_dim": int(paper_vectors.shape[1]) if paper_vectors.size else 0,
            "chunk_vector_dim": int(chunk_vectors.shape[1]) if chunk_vectors.size else 0,
            "pdf_parser_backend": self.settings.pdf_parser_backend,
            "parser_backend_counts": dict(sorted(parser_backend_counts.items())),
            "parse_failure_counts": dict(sorted(parse_failure_counts.items())),
            "parse_failure_path": str(self.store.parse_failure_path),
        }
        self.store.save_index_state(state)

        return BuildIndexSummary(
            papers=len(parsed_papers),
            total_papers=len(updated_papers),
            indexed_papers=len(parsed_papers),
            failed_papers=len(parse_failures),
            sections=len(sections),
            objects=len(objects),
            chunks=len(chunks),
            text_chunks=len(text_chunks),
            table_chunks=len(table_chunks),
            figure_chunks=len(figure_chunks),
            deep_chat_evidence_units=len(deep_chat_evidence_units),
            paper_vector_dim=int(paper_vectors.shape[1]) if paper_vectors.size else 0,
            chunk_vector_dim=int(chunk_vectors.shape[1]) if chunk_vectors.size else 0,
            paper_dense_backend=self.paper_encoder.backend_name,
            chunk_dense_backend=self.chunk_encoder.backend_name,
            pdf_parser_backend=self.settings.pdf_parser_backend,
            parser_backend_counts=dict(sorted(parser_backend_counts.items())),
            parse_failure_counts=dict(sorted(parse_failure_counts.items())),
            built_at=built_at,
        )

    def load_paper_ids(self, path: Path) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            paper_id = line.strip()
            if not paper_id or paper_id.startswith("#") or paper_id in seen:
                continue
            seen.add(paper_id)
            ids.append(paper_id)
        if not ids:
            raise RuntimeError(f"Paper id file is empty: {path}")
        return ids

    def _select_source_papers(
        self,
        *,
        max_papers: int | None,
        paper_ids: list[str] | None,
    ) -> list[PaperRecord]:
        source_papers = self.store.load_source_papers()
        if paper_ids is not None:
            paper_lookup = {paper.paper_id: paper for paper in source_papers}
            missing_ids: list[str] = []
            selected: list[PaperRecord] = []
            for paper_id in paper_ids:
                paper = paper_lookup.get(paper_id)
                if paper is None:
                    missing_ids.append(paper_id)
                    continue
                selected.append(paper)
            if missing_ids:
                preview = ", ".join(missing_ids[:5])
                suffix = " ..." if len(missing_ids) > 5 else ""
                raise RuntimeError(
                    f"{len(missing_ids)} paper ids were not found in data/raw/papers.jsonl: {preview}{suffix}"
                )
            source_papers = selected
        if max_papers is not None:
            if max_papers <= 0:
                raise RuntimeError(f"max_papers must be positive, got {max_papers}")
            source_papers = source_papers[:max_papers]
        if not source_papers:
            raise RuntimeError("No source papers selected for build-index.")
        return source_papers

    def _encode(
        self,
        encoder: SentenceTransformerEncoder,
        texts: list[str],
        name: str,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=float)
        display_name = ENCODE_LABELS.get(name, name)
        total = len(texts)
        start_time = time.perf_counter()
        checkpoint_count = min(40, total)
        next_checkpoint = 1

        if self.progress is not None:
            self.progress.encode_start(name=display_name, total=total, backend=encoder.backend_name)
        else:
            logger.info(
                "[bold blue]Index[/] | encode_start dataset=%s total=%s backend=%s",
                display_name,
                total,
                encoder.backend_name,
            )

        def progress_callback(completed: int, total_items: int) -> None:
            nonlocal next_checkpoint
            self._check_cancel()
            if total_items <= 0:
                return
            if self.progress is not None:
                self.progress.encode_update(name=display_name, completed=completed, total=total_items)
                return
            progress_ratio = completed / total_items
            target_checkpoint = math.floor(progress_ratio * checkpoint_count)
            if completed < total_items and target_checkpoint < next_checkpoint:
                return
            elapsed = time.perf_counter() - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = max(total_items - completed, 0)
            eta_seconds = remaining / rate if rate > 0 else None
            logger.info(
                "[bold blue]Index[/] | encode_progress dataset=%s completed=%s total=%s percent=%.2f rate_items_per_sec=%.2f eta=%s",
                name,
                completed,
                total_items,
                progress_ratio * 100,
                rate,
                self._format_duration(eta_seconds),
            )
            next_checkpoint = min(checkpoint_count + 1, target_checkpoint + 1)

        vectors = encoder.encode(texts, progress_callback=progress_callback)
        elapsed = time.perf_counter() - start_time
        rate = total / elapsed if elapsed > 0 else 0.0
        if self.progress is not None:
            self.progress.encode_done(name=display_name, total=total)
        else:
            logger.info(
                "[bold blue]Index[/] | encode_done dataset=%s total=%s elapsed=%s avg_rate_items_per_sec=%.2f",
                display_name,
                total,
                self._format_duration(elapsed),
                rate,
            )
        return vectors

    def _check_cancel(self) -> None:
        if self.cancel_check is not None:
            self.cancel_check()

    def _format_duration(self, seconds: float | None) -> str:
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

    def _save_index_payload(
        self,
        name: str,
        ids: list[str],
        texts: list[str],
        vectors: np.ndarray,
        encoder: SentenceTransformerEncoder,
        extra: dict[str, list] | None = None,
    ) -> None:
        self.store.save_vectors(name, ids, vectors)
        payload = {
            "ids": ids,
            "texts": texts,
            "tokens": [tokenize(text) for text in texts],
            "encoder_backend": encoder.backend_name,
            "encoder_model": encoder.model_name,
            "vector_dim": int(vectors.shape[1]) if vectors.size else 0,
            "built_at": now_iso(),
        }
        if extra:
            payload.update(extra)
        self.store.save_index_meta(name, payload)

    def _paper_search_text(self, paper: PaperRecord) -> str:
        evidence_terms = []
        for evidence_type, count in sorted(paper.typed_evidence_summary.items()):
            evidence_terms.extend([f"{evidence_type} evidence"] * min(count, 3))
        parts = [
            paper.title,
            paper.abstract,
            " ".join(paper.section_headings),
            paper.intro_summary,
            " ".join(paper.keywords),
            " ".join(evidence_terms),
        ]
        return " ".join(part for part in parts if part)

    def _section_search_text(self, section: SectionRecord) -> str:
        parts = [
            section.section_title,
            " ".join(section.section_path),
            section.text_summary,
            section.text,
        ]
        return " ".join(part for part in parts if part)

    def _chunk_search_text(self, chunk: ChunkRecord) -> str:
        parts = [
            chunk.heading,
            " ".join(chunk.section_path),
            chunk.text,
            " ".join(f"{name} evidence" for name in chunk.evidence_types),
        ]
        return " ".join(part for part in parts if part)
