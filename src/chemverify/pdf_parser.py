from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from .config import Settings
from .models import (
    ChunkRecord,
    ObjectRecord,
    PaperRecord,
    ParseFailureRecord,
    ParsedPaperBundle,
    SectionNode,
    SectionRecord,
)
from .utils import (
    build_typed_evidence_summary,
    extract_keywords,
    infer_evidence_scores,
    make_stable_id,
    normalize_whitespace,
    now_iso,
    select_evidence_types,
    tokenize,
    truncate_text,
)

logger = logging.getLogger(__name__)

_REFERENCE_SECTION_PATTERN = re.compile(
    r"^(?:(?:acknowledg?ements?|acknowledgments?)\s+and\s+)?(?:references?|bibliography|works cited)(?:\s+and\s+.*)?$",
    re.IGNORECASE,
)
_TEXT_BLOCK_TYPES = {"text", "list"}
_TABLE_BLOCK_TYPES = {"table"}
_FIGURE_BLOCK_TYPES = {"image", "figure"}
_EQUATION_BLOCK_TYPES = {"equation", "formula", "interline_equation", "inline_equation"}
_SKIP_BLOCK_TYPES = {
    "discarded",
    "abandon",
    "header",
    "footer",
    "page_number",
    "page_header",
    "page_footer",
    "footnote",
}
_NOISE_HEADING_PHRASES = (
    "conference on",
    "association for computational linguistics",
    "proceedings of",
    "arxiv:",
    "copyright",
)


@dataclass(slots=True)
class _ArtifactBundle:
    parse_dir: Path
    middle_path: Path
    content_list_path: Path | None
    markdown_path: Path | None
    images_dir: Path | None


@dataclass(slots=True)
class _SectionAccumulator:
    section_id: str
    section_title: str
    section_path: list[str]
    level: int
    ordinal: int
    page_start: int
    page_end: int
    member_object_ids: list[str] = field(default_factory=list)
    text_fragments: list[str] = field(default_factory=list)

    def add_object(self, obj: ObjectRecord) -> None:
        self.member_object_ids.append(obj.object_id)
        self.page_start = min(self.page_start, obj.page_idx)
        self.page_end = max(self.page_end, obj.page_idx)
        fragment = _object_text_for_section(obj)
        if fragment:
            self.text_fragments.append(fragment)

    def finalize(self) -> SectionRecord:
        text = normalize_whitespace("\n\n".join(fragment for fragment in self.text_fragments if fragment))
        return SectionRecord(
            section_id=self.section_id,
            paper_id="",
            section_title=self.section_title,
            section_path=self.section_path,
            level=self.level,
            ordinal=self.ordinal,
            page_start=self.page_start,
            page_end=self.page_end,
            text=text,
            text_summary=truncate_text(text, limit=900),
            member_object_ids=list(self.member_object_ids),
            member_chunk_ids=[],
        )


@dataclass(slots=True)
class _TextPiece:
    object_id: str
    page_idx: int
    text: str
    token_count: int


class PDFParseError(RuntimeError):
    def __init__(
        self,
        *,
        parser_backend: str,
        stage: str,
        error_type: str,
        error_message: str,
        analysis: str,
        suggestion: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(error_message)
        self.parser_backend = parser_backend
        self.stage = stage
        self.error_type = error_type
        self.error_message = error_message
        self.analysis = analysis
        self.suggestion = suggestion
        self.details = details or {}


class BasePDFParser:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.parser_backend = "base"

    def build_failure_record(self, paper: PaperRecord, exc: Exception) -> ParseFailureRecord:
        if isinstance(exc, PDFParseError):
            parser_backend = exc.parser_backend
            stage = exc.stage
            error_type = exc.error_type
            error_message = exc.error_message
            analysis = exc.analysis
            suggestion = exc.suggestion
            details = exc.details
        else:
            parser_backend = self.parser_backend
            stage = "unexpected"
            error_type = exc.__class__.__name__
            error_message = str(exc).strip() or repr(exc)
            analysis = "The parser raised an unexpected exception before producing layout normalized outputs."
            suggestion = "Inspect the stack trace and the MinerU artifacts for this paper before rerunning the parser."
            details = {"exception_repr": repr(exc)}
        return ParseFailureRecord(
            paper_id=paper.paper_id,
            venue=paper.venue,
            year=paper.year,
            track=paper.track,
            parser_backend=parser_backend,
            stage=stage,
            error_type=error_type,
            error_message=error_message,
            local_pdf_path=paper.local_pdf_path,
            analysis=analysis,
            suggestion=suggestion,
            details=details,
            occurred_at=now_iso(),
        )

    def _parse_error(
        self,
        *,
        stage: str,
        error_type: str,
        error_message: str,
        analysis: str,
        suggestion: str,
        details: dict[str, Any] | None = None,
    ) -> PDFParseError:
        return PDFParseError(
            parser_backend=self.parser_backend,
            stage=stage,
            error_type=error_type,
            error_message=error_message,
            analysis=analysis,
            suggestion=suggestion,
            details=details,
        )


class MinerULayoutV2Parser(BasePDFParser):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.parser_backend = "mineru_layout"

    def parse(self, paper: PaperRecord) -> ParsedPaperBundle:
        if not paper.local_pdf_path:
            raise self._parse_error(
                stage="pdf_lookup",
                error_type="missing_local_pdf_path",
                error_message="Paper is missing local_pdf_path, so MinerU artifacts cannot be resolved.",
                analysis="The index builder expects ACL Anthology ingestion to download the PDF before normalized parsing.",
                suggestion="Re-run ingestion with PDF download enabled and make sure the paper points to a local PDF file.",
            )

        pdf_path = Path(paper.local_pdf_path)
        if not pdf_path.exists():
            raise self._parse_error(
                stage="pdf_lookup",
                error_type="local_pdf_missing",
                error_message=f"Local PDF does not exist: {pdf_path}",
                analysis="The paper metadata points to a PDF path that is missing on disk.",
                suggestion="Check corpus download status for this paper and re-download the PDF before rebuilding indexes.",
                details={"local_pdf_path": str(pdf_path)},
            )

        artifacts = self._resolve_artifacts(paper.paper_id)
        middle_payload = json.loads(artifacts.middle_path.read_text(encoding="utf-8"))
        content_items = []
        if artifacts.content_list_path is not None and artifacts.content_list_path.exists():
            content_items = json.loads(artifacts.content_list_path.read_text(encoding="utf-8"))

        sections, objects = self._extract_sections_and_objects(paper, middle_payload, content_items, artifacts)
        if not sections:
            raise self._parse_error(
                stage="section_construction",
                error_type="no_sections_constructed",
                error_message="No sections could be constructed from MinerU middle.json.",
                analysis="The parser found the artifacts but could not build any usable sections before hitting the reference boundary or empty content.",
                suggestion="Inspect the markdown and middle.json for this paper; check whether title blocks were misdetected or body text is missing.",
                details={"middle_path": str(artifacts.middle_path)},
            )
        if not objects:
            raise self._parse_error(
                stage="object_construction",
                error_type="no_objects_constructed",
                error_message="No retrievable objects could be constructed from MinerU output.",
                analysis="The parser built section boundaries but failed to recover text, table, figure, or list objects.",
                suggestion="Inspect the para_blocks inside middle.json and verify that the target paper was parsed with middle.json export enabled.",
                details={"middle_path": str(artifacts.middle_path)},
            )

        chunks, sections = self._build_chunks(paper.paper_id, sections, objects)
        if not chunks:
            raise self._parse_error(
                stage="chunk_construction",
                error_type="no_chunks_constructed",
                error_message="No chunks were produced from the layout objects.",
                analysis="Objects were extracted, but chunk assembly produced no retrievable text/table/figure units.",
                suggestion="Inspect section/object construction and verify whether text-bearing objects survived normalization.",
            )

        updated_paper = self._finalize_paper(paper, sections, objects, chunks, artifacts)
        return ParsedPaperBundle(paper=updated_paper, sections=sections, objects=objects, chunks=chunks)

    def _resolve_artifacts(self, paper_id: str) -> _ArtifactBundle:
        root = self.settings.mineru_output_dir / paper_id
        if not root.exists():
            raise self._parse_error(
                stage="artifact_lookup",
                error_type="mineru_artifact_root_missing",
                error_message=f"MinerU artifact directory is missing for {paper_id}: {root}",
                analysis="The new layout parser expects a pre-generated MinerU artifact directory for every paper.",
                suggestion="Run the MinerU cache pipeline for ACL 2025 long papers before building the layout normalized corpus.",
                details={"artifact_root": str(root)},
            )

        middle_candidates = sorted(root.rglob(f"{paper_id}_middle.json"))
        if not middle_candidates and self.settings.mineru_require_middle_json:
            raise self._parse_error(
                stage="artifact_lookup",
                error_type="missing_middle_json",
                error_message=f"Missing required MinerU middle.json for {paper_id}.",
                analysis="Layout V2 uses middle.json as the primary structural source; the old content_list-only export is insufficient.",
                suggestion="Re-run the MinerU pipeline with middle.json export enabled for this paper.",
                details={"artifact_root": str(root)},
            )
        if not middle_candidates:
            raise self._parse_error(
                stage="artifact_lookup",
                error_type="missing_middle_json",
                error_message=f"Could not find a usable middle.json for {paper_id}.",
                analysis="The parser could not locate the structural MinerU artifact required by layout.",
                suggestion="Re-run MinerU for this paper and verify that middle.json is emitted into the parse directory.",
                details={"artifact_root": str(root)},
            )
        middle_path = middle_candidates[0]
        parse_dir = middle_path.parent

        content_list_path = parse_dir / f"{paper_id}_content_list.json"
        if not content_list_path.exists():
            fallback_candidates = sorted(root.rglob(f"{paper_id}_content_list.json"))
            content_list_path = fallback_candidates[0] if fallback_candidates else None
        if content_list_path is None:
            raise self._parse_error(
                stage="artifact_lookup",
                error_type="missing_content_list",
                error_message=f"Missing required MinerU content_list.json for {paper_id}.",
                analysis="Layout V2 uses middle.json as the primary structural input, but still relies on content_list.json to supplement table, figure, and image metadata.",
                suggestion="Re-run the MinerU pipeline with content_list export enabled for this paper.",
                details={"artifact_root": str(root)},
            )

        markdown_path = parse_dir / f"{paper_id}.md"
        if not markdown_path.exists():
            markdown_candidates = sorted(root.rglob(f"{paper_id}.md"))
            markdown_path = markdown_candidates[0] if markdown_candidates else None
        if self.settings.mineru_require_markdown and markdown_path is None:
            raise self._parse_error(
                stage="artifact_lookup",
                error_type="missing_markdown",
                error_message=f"Missing required MinerU markdown export for {paper_id}.",
                analysis="The new workflow requires markdown output to support human inspection of parsing quality.",
                suggestion="Re-run the MinerU pipeline with markdown export enabled for this paper.",
                details={"artifact_root": str(root)},
            )

        images_dir = parse_dir / "images"
        if not images_dir.exists():
            images_dir = None
        return _ArtifactBundle(
            parse_dir=parse_dir,
            middle_path=middle_path,
            content_list_path=content_list_path,
            markdown_path=markdown_path,
            images_dir=images_dir,
        )

    def _extract_sections_and_objects(
        self,
        paper: PaperRecord,
        middle_payload: dict[str, Any],
        content_items: list[dict[str, Any]],
        artifacts: _ArtifactBundle,
    ) -> tuple[list[SectionRecord], list[ObjectRecord]]:
        pages = middle_payload.get("pdf_info")
        if not isinstance(pages, list) or not pages:
            raise self._parse_error(
                stage="middle_json_validation",
                error_type="empty_pdf_info",
                error_message="middle.json does not contain a usable pdf_info page list.",
                analysis="The structural MinerU export exists, but its page-level payload is empty or malformed.",
                suggestion="Inspect the exported middle.json and rerun MinerU for this paper if pdf_info is missing.",
                details={"middle_path": str(artifacts.middle_path)},
            )

        content_queues = self._build_content_queues(content_items)
        section_accumulators: list[_SectionAccumulator] = []
        objects: list[ObjectRecord] = []
        section_stack: list[tuple[int, str]] = []
        current_section: _SectionAccumulator | None = None
        document_title_consumed = False
        object_ordinal = 0
        reference_reached = False

        for page_offset, page in enumerate(pages):
            page_idx = int(page.get("page_idx", page_offset)) + 1
            para_blocks = page.get("para_blocks", [])
            if not isinstance(para_blocks, list):
                continue

            for block in para_blocks:
                if not isinstance(block, dict):
                    continue
                block_type = _normalize_block_type(block.get("type"))
                if block_type in _SKIP_BLOCK_TYPES:
                    continue

                if block_type == "title":
                    heading_text = _extract_block_text(block)
                    if not heading_text:
                        continue
                    if (
                        not document_title_consumed
                        and current_section is None
                        and not section_accumulators
                        and page_idx == 1
                        and _is_likely_document_title(heading_text)
                    ):
                        document_title_consumed = True
                        continue
                    if self._is_reference_heading(heading_text):
                        reference_reached = True
                        break
                    current_section = self._start_section(
                        paper.paper_id,
                        section_accumulators,
                        section_stack,
                        heading_text,
                        page_idx,
                    )
                    continue

                object_record = self._build_object_record(
                    paper_id=paper.paper_id,
                    current_section=current_section,
                    section_accumulators=section_accumulators,
                    section_stack=section_stack,
                    page_idx=page_idx,
                    block=block,
                    content_queues=content_queues,
                    artifact_dir=artifacts.parse_dir,
                    next_ordinal=object_ordinal + 1,
                )
                if object_record is None:
                    continue
                if current_section is None:
                    continue
                object_ordinal += 1
                object_record.section_id = current_section.section_id
                object_record.section_path = list(current_section.section_path)
                current_section.add_object(object_record)
                objects.append(object_record)

            if reference_reached:
                break

        sections: list[SectionRecord] = []
        for accumulator in section_accumulators:
            section = accumulator.finalize()
            section.paper_id = paper.paper_id
            if section.text or section.member_object_ids:
                sections.append(section)
        return sections, objects

    def _start_section(
        self,
        paper_id: str,
        section_accumulators: list[_SectionAccumulator],
        section_stack: list[tuple[int, str]],
        heading_text: str,
        page_idx: int,
    ) -> _SectionAccumulator:
        cleaned_heading = normalize_whitespace(heading_text)
        level = _infer_heading_level(cleaned_heading)
        while section_stack and section_stack[-1][0] >= level:
            section_stack.pop()
        section_path = [title for _, title in section_stack] + [cleaned_heading]
        ordinal = len(section_accumulators) + 1
        accumulator = _SectionAccumulator(
            section_id=make_stable_id("section", f"{paper_id}:{ordinal}:{cleaned_heading}:{page_idx}"),
            section_title=cleaned_heading,
            section_path=section_path,
            level=level,
            ordinal=ordinal,
            page_start=page_idx,
            page_end=page_idx,
        )
        section_accumulators.append(accumulator)
        section_stack.append((level, cleaned_heading))
        return accumulator

    def _build_object_record(
        self,
        *,
        paper_id: str,
        current_section: _SectionAccumulator | None,
        section_accumulators: list[_SectionAccumulator],
        section_stack: list[tuple[int, str]],
        page_idx: int,
        block: dict[str, Any],
        content_queues: dict[tuple[int, str], list[dict[str, Any]]],
        artifact_dir: Path,
        next_ordinal: int,
    ) -> ObjectRecord | None:
        block_type = _normalize_block_type(block.get("type"))
        source_fields: list[str] = []
        bbox = _extract_bbox(block)

        if block_type in _TEXT_BLOCK_TYPES:
            text = _extract_block_text(block)
            if not text:
                return None
            source_fields.append("middle.text")
            return ObjectRecord(
                object_id=make_stable_id("obj", f"{paper_id}:{next_ordinal}:{block_type}:{page_idx}:{text[:80]}"),
                paper_id=paper_id,
                section_id=current_section.section_id if current_section else "",
                object_type="text_block" if block_type == "text" else "list_block",
                ordinal=next_ordinal,
                page_idx=page_idx,
                bbox=bbox,
                text=text,
                source_fields=source_fields,
            )

        if block_type in _TABLE_BLOCK_TYPES:
            supplement_match = _match_content_supplement(content_queues, page_idx, block_type, bbox)
            supplement = supplement_match[1] if supplement_match is not None else None
            caption = _extract_caption(block, supplement, kind="table")
            footnote = _extract_footnote(block, supplement, kind="table")
            html = _extract_html(block, supplement)
            body_text = _extract_table_text(block, supplement, html)
            text = normalize_whitespace("\n\n".join(part for part in [caption, body_text, footnote] if part))
            if not text:
                return None
            source_fields.extend(_non_empty_source_fields(("caption", caption), ("html", html), ("text", body_text), ("footnote", footnote)))
            if supplement_match is not None:
                _consume_content_supplement(content_queues, page_idx, block_type, supplement_match[0])
            return ObjectRecord(
                object_id=make_stable_id("obj", f"{paper_id}:{next_ordinal}:table:{page_idx}:{caption[:80] or body_text[:80]}"),
                paper_id=paper_id,
                section_id=current_section.section_id if current_section else "",
                object_type="table_block",
                ordinal=next_ordinal,
                page_idx=page_idx,
                bbox=bbox,
                text=text,
                caption=caption,
                footnote=footnote,
                html=html,
                source_fields=source_fields,
            )

        if block_type in _FIGURE_BLOCK_TYPES:
            supplement_match = _match_content_supplement(content_queues, page_idx, block_type, bbox)
            supplement = supplement_match[1] if supplement_match is not None else None
            caption = _extract_caption(block, supplement, kind="image")
            footnote = _extract_footnote(block, supplement, kind="image")
            text = normalize_whitespace("\n\n".join(part for part in [caption, footnote] if part))
            if not text:
                text = _extract_block_text(block)
            if not text:
                return None
            image_path = _resolve_image_path(supplement, artifact_dir)
            source_fields.extend(_non_empty_source_fields(("caption", caption), ("footnote", footnote), ("image_path", image_path or "")))
            if supplement_match is not None:
                _consume_content_supplement(content_queues, page_idx, block_type, supplement_match[0])
            return ObjectRecord(
                object_id=make_stable_id("obj", f"{paper_id}:{next_ordinal}:figure:{page_idx}:{text[:80]}"),
                paper_id=paper_id,
                section_id=current_section.section_id if current_section else "",
                object_type="figure_block",
                ordinal=next_ordinal,
                page_idx=page_idx,
                bbox=bbox,
                text=text,
                caption=caption,
                footnote=footnote,
                image_path=image_path,
                source_fields=source_fields,
            )

        if block_type in _EQUATION_BLOCK_TYPES:
            text = _extract_block_text(block) or _extract_math_text(block)
            if not text:
                return None
            source_fields.append("middle.equation")
            return ObjectRecord(
                object_id=make_stable_id("obj", f"{paper_id}:{next_ordinal}:equation:{page_idx}:{text[:80]}"),
                paper_id=paper_id,
                section_id=current_section.section_id if current_section else "",
                object_type="equation_block",
                ordinal=next_ordinal,
                page_idx=page_idx,
                bbox=bbox,
                text=text,
                source_fields=source_fields,
            )
        return None

    def _build_chunks(
        self,
        paper_id: str,
        sections: list[SectionRecord],
        objects: list[ObjectRecord],
    ) -> tuple[list[ChunkRecord], list[SectionRecord]]:
        objects_by_section: dict[str, list[ObjectRecord]] = defaultdict(list)
        for obj in objects:
            objects_by_section[obj.section_id].append(obj)

        chunks: list[ChunkRecord] = []
        updated_sections: list[SectionRecord] = []
        for section in sections:
            section_objects = sorted(objects_by_section.get(section.section_id, []), key=lambda item: item.ordinal)
            section_chunks = self._build_section_chunks(paper_id, section, section_objects)
            section.member_chunk_ids = [chunk.chunk_id for chunk in section_chunks]
            updated_sections.append(section)
            chunks.extend(section_chunks)
        return chunks, updated_sections

    def _build_section_chunks(
        self,
        paper_id: str,
        section: SectionRecord,
        section_objects: list[ObjectRecord],
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        buffer: list[_TextPiece] = []
        buffer_tokens = 0

        def flush_buffer(*, keep_overlap: bool) -> None:
            nonlocal buffer, buffer_tokens
            if not buffer:
                return
            text = normalize_whitespace("\n\n".join(piece.text for piece in buffer))
            if text:
                chunks.append(
                    self._make_chunk(
                        paper_id=paper_id,
                        section=section,
                        chunk_type="text_chunk",
                        member_object_ids=[piece.object_id for piece in buffer],
                        page_start=min(piece.page_idx for piece in buffer),
                        page_end=max(piece.page_idx for piece in buffer),
                        text=text,
                    )
                )
            buffer = _tail_overlap(buffer, self.settings.chunk_overlap_tokens) if keep_overlap else []
            buffer_tokens = sum(piece.token_count for piece in buffer)

        for obj in section_objects:
            if obj.object_type in {"text_block", "list_block"}:
                token_count = len(tokenize(obj.text))
                if token_count == 0:
                    continue
                if buffer and buffer_tokens + token_count > self.settings.chunk_target_tokens and buffer_tokens >= max(80, self.settings.chunk_target_tokens // 2):
                    flush_buffer(keep_overlap=True)
                buffer.append(_TextPiece(object_id=obj.object_id, page_idx=obj.page_idx, text=obj.text, token_count=token_count))
                buffer_tokens += token_count
                continue

            if obj.object_type == "equation_block":
                continue

            flush_buffer(keep_overlap=False)
            chunk_type = "table_chunk" if obj.object_type == "table_block" else "figure_chunk"
            chunks.append(
                self._make_chunk(
                    paper_id=paper_id,
                    section=section,
                    chunk_type=chunk_type,
                    member_object_ids=[obj.object_id],
                    page_start=obj.page_idx,
                    page_end=obj.page_idx,
                    text=obj.text,
                    metadata={
                        "object_type": obj.object_type,
                        "caption": obj.caption,
                        "footnote": obj.footnote,
                        "image_path": obj.image_path,
                    },
                )
            )

        flush_buffer(keep_overlap=False)
        return chunks

    def _make_chunk(
        self,
        *,
        paper_id: str,
        section: SectionRecord,
        chunk_type: str,
        member_object_ids: list[str],
        page_start: int,
        page_end: int,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChunkRecord:
        evidence_scores = infer_evidence_scores(
            heading=section.section_title,
            section_path=section.section_path,
            text=text,
        )
        return ChunkRecord(
            chunk_id=make_stable_id("chunk", f"{paper_id}:{section.section_id}:{chunk_type}:{page_start}:{page_end}:{'|'.join(member_object_ids)}"),
            paper_id=paper_id,
            section_id=section.section_id,
            chunk_type=chunk_type,
            member_object_ids=member_object_ids,
            heading=section.section_title,
            section_path=section.section_path,
            page_start=page_start,
            page_end=page_end,
            page_span=[page_start, page_end],
            token_count=len(tokenize(text)),
            evidence_types=select_evidence_types(evidence_scores),
            evidence_scores=evidence_scores,
            text=text,
            metadata=metadata or {},
        )

    def _finalize_paper(
        self,
        paper: PaperRecord,
        sections: list[SectionRecord],
        objects: list[ObjectRecord],
        chunks: list[ChunkRecord],
        artifacts: _ArtifactBundle,
    ) -> PaperRecord:
        full_text, section_text_offsets = _compose_paper_text_and_offsets(sections)
        intro_summary = _select_intro_summary(sections, paper.abstract)
        parser_backend = self.parser_backend
        _assign_chunk_char_offsets(chunks, sections, section_text_offsets)
        section_tree = _build_section_tree(sections, section_text_offsets)
        return paper.model_copy(
            deep=True,
            update={
                "parser_backend": parser_backend,
                "text": full_text,
                "intro_summary": intro_summary,
                "section_headings": [section.section_title for section in sections],
                "keywords": extract_keywords(f"{paper.title} {paper.abstract} {intro_summary}", limit=12),
                "sections": section_tree,
                "section_ids": [section.section_id for section in sections],
                "object_ids": [obj.object_id for obj in objects],
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "typed_evidence_summary": build_typed_evidence_summary(chunk.evidence_scores for chunk in chunks),
                "metadata": {
                    **paper.metadata,
                    "parse_status": "success",
                    "pdf_parser_backend": parser_backend,
                    "layout": {
                        "sections": len(sections),
                        "objects": len(objects),
                        "chunks": len(chunks),
                    },
                    "mineru_artifacts": {
                        "middle_json": str(artifacts.middle_path),
                        "content_list_json": str(artifacts.content_list_path) if artifacts.content_list_path else None,
                        "markdown": str(artifacts.markdown_path) if artifacts.markdown_path else None,
                        "images_dir": str(artifacts.images_dir) if artifacts.images_dir else None,
                    },
                },
            },
        )

    def _build_content_queues(self, content_items: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
        queues: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for item in content_items:
            if not isinstance(item, dict):
                continue
            block_type = _normalize_block_type(item.get("type"))
            if block_type in _SKIP_BLOCK_TYPES:
                continue
            page_idx = int(item.get("page_idx", 0)) + 1
            queues[(page_idx, block_type)].append(item)
        return queues

    def _is_reference_heading(self, heading: str) -> bool:
        normalized = normalize_whitespace(re.sub(r"^\d+(?:\.\d+)*\s*", "", heading)).lower()
        return bool(_REFERENCE_SECTION_PATTERN.match(normalized))


class PDFParser:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mineru_parser = MinerULayoutV2Parser(settings)

    def parse(self, paper: PaperRecord) -> ParsedPaperBundle:
        if self.settings.pdf_parser_backend != "mineru_layout":
            raise RuntimeError(f"Unsupported pdf_parser_backend: {self.settings.pdf_parser_backend}")
        return self.mineru_parser.parse(paper)

    def build_failure_record(self, paper: PaperRecord, exc: Exception) -> ParseFailureRecord:
        return self.mineru_parser.build_failure_record(paper, exc)

    def resolve_artifacts(self, paper_id: str) -> _ArtifactBundle:
        return self.mineru_parser._resolve_artifacts(paper_id)


def _normalize_block_type(raw: Any) -> str:
    return normalize_whitespace(str(raw or "")).lower().replace(" ", "_")


def _extract_block_text(payload: Any) -> str:
    if isinstance(payload, str):
        return normalize_whitespace(payload)
    if isinstance(payload, list):
        combined = [_extract_block_text(item) for item in payload]
        return normalize_whitespace(" ".join(item for item in combined if item))
    if not isinstance(payload, dict):
        return ""

    direct_text = payload.get("text")
    if isinstance(direct_text, str) and normalize_whitespace(direct_text):
        return normalize_whitespace(direct_text)

    pieces: list[str] = []
    for line in payload.get("lines", []) if isinstance(payload.get("lines"), list) else []:
        if isinstance(line, dict):
            line_text = line.get("text")
            if isinstance(line_text, str) and normalize_whitespace(line_text):
                pieces.append(normalize_whitespace(line_text))
            spans = line.get("spans")
            if isinstance(spans, list):
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    span_text = span.get("content") or span.get("text") or span.get("value") or span.get("latex")
                    if isinstance(span_text, str) and normalize_whitespace(span_text):
                        pieces.append(normalize_whitespace(span_text))
    if pieces:
        return normalize_whitespace(" ".join(pieces))

    nested = []
    for key in ("caption", "content", "blocks", "paragraphs"):
        if key in payload:
            nested_text = _extract_block_text(payload[key])
            if nested_text:
                nested.append(nested_text)
    return normalize_whitespace(" ".join(nested))


def _extract_bbox(block: dict[str, Any]) -> list[float]:
    bbox = block.get("bbox")
    if isinstance(bbox, list):
        values = []
        for item in bbox[:4]:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                break
        if len(values) == 4:
            return values
    return []


def _extract_caption(block: dict[str, Any], supplement: dict[str, Any] | None, *, kind: str) -> str:
    keys = {
        "table": ("caption", "table_caption", "table_caption_text"),
        "image": ("caption", "image_caption", "img_caption", "figure_caption"),
    }[kind]
    for key in keys:
        value = block.get(key)
        if isinstance(value, str) and normalize_whitespace(value):
            return normalize_whitespace(value)
    if supplement is not None:
        for key in keys:
            value = supplement.get(key)
            if isinstance(value, str) and normalize_whitespace(value):
                return normalize_whitespace(value)
    return ""


def _extract_footnote(block: dict[str, Any], supplement: dict[str, Any] | None, *, kind: str) -> str:
    keys = {
        "table": ("footnote", "table_footnote", "table_note"),
        "image": ("footnote", "image_footnote", "img_footnote", "figure_footnote"),
    }[kind]
    for key in keys:
        value = block.get(key)
        if isinstance(value, str) and normalize_whitespace(value):
            return normalize_whitespace(value)
    if supplement is not None:
        for key in keys:
            value = supplement.get(key)
            if isinstance(value, str) and normalize_whitespace(value):
                return normalize_whitespace(value)
    return ""


def _extract_html(block: dict[str, Any], supplement: dict[str, Any] | None) -> str:
    for value in (block.get("html"), (supplement or {}).get("html"), (supplement or {}).get("table_html")):
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_table_text(block: dict[str, Any], supplement: dict[str, Any] | None, html: str) -> str:
    for key in ("table_body", "body", "text"):
        value = block.get(key)
        if isinstance(value, str) and normalize_whitespace(value):
            return normalize_whitespace(value)
    if supplement is not None:
        for key in ("table_body", "body", "text"):
            value = supplement.get(key)
            if isinstance(value, str) and normalize_whitespace(value):
                return normalize_whitespace(value)
    if html:
        html_text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        return normalize_whitespace(html_text)
    return _extract_block_text(block)


def _extract_math_text(block: dict[str, Any]) -> str:
    for key in ("latex", "text", "math"):
        value = block.get(key)
        if isinstance(value, str) and normalize_whitespace(value):
            return normalize_whitespace(value)
    return _extract_block_text(block)


def _resolve_image_path(supplement: dict[str, Any] | None, artifact_dir: Path) -> str | None:
    if supplement is None:
        return None
    for key in ("img_path", "image_path", "path"):
        value = supplement.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = artifact_dir / candidate
        return str(candidate.resolve())
    return None


def _match_content_supplement(
    content_queues: dict[tuple[int, str], list[dict[str, Any]]],
    page_idx: int,
    block_type: str,
    block_bbox: list[float],
) -> tuple[int, dict[str, Any]] | None:
    queue = content_queues.get((page_idx, block_type))
    if not queue:
        return None
    if len(queue) == 1 or not block_bbox:
        return 0, queue[0]
    best_index = 0
    best_distance = float("inf")
    for index, item in enumerate(queue):
        item_bbox = _extract_bbox(item)
        if not item_bbox:
            continue
        distance = _bbox_center_distance(block_bbox, item_bbox)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index, queue[best_index]


def _consume_content_supplement(
    content_queues: dict[tuple[int, str], list[dict[str, Any]]],
    page_idx: int,
    block_type: str,
    index: int,
) -> dict[str, Any] | None:
    queue = content_queues.get((page_idx, block_type))
    if not queue or index < 0 or index >= len(queue):
        return None
    return queue.pop(index)


def _build_section_tree(sections: list[SectionRecord], section_text_offsets: dict[str, tuple[int, int]]) -> list[SectionNode]:
    roots: list[SectionNode] = []
    stack: list[SectionNode] = []
    for section in sections:
        start, end = section_text_offsets.get(section.section_id, (0, 0))
        node = SectionNode(
            node_id=section.section_id,
            heading=section.section_title,
            level=section.level,
            page_start=section.page_start,
            page_end=section.page_end,
            char_start=start,
            char_end=end,
        )
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def _infer_heading_level(heading: str) -> int:
    cleaned = normalize_whitespace(heading)
    match = re.match(r"^(?:section\s+)?(\d+(?:\.\d+)*)\s+", cleaned, flags=re.IGNORECASE)
    if match:
        return min(4, match.group(1).count(".") + 1)
    if re.match(r"^(appendix|appendices)\b", cleaned, flags=re.IGNORECASE):
        return 1
    if re.match(r"^[A-Z]\.\s+", cleaned):
        return 2
    return 1


def _is_likely_document_title(heading: str) -> bool:
    cleaned = normalize_whitespace(heading)
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered in {"abstract", "keywords"}:
        return False
    if _REFERENCE_SECTION_PATTERN.match(lowered):
        return False
    if re.match(r"^(?:section\s+)?\d+(?:\.\d+)*\s+", cleaned, flags=re.IGNORECASE):
        return False
    if re.match(r"^(appendix|appendices|acknowledg?ements?|introduction|conclusion|related work)\b", lowered):
        return False
    if any(phrase in lowered for phrase in _NOISE_HEADING_PHRASES):
        return False
    return True


def _object_text_for_section(obj: ObjectRecord) -> str:
    if obj.object_type in {"text_block", "list_block", "table_block", "figure_block"}:
        return obj.text
    return ""


def _tail_overlap(pieces: list[_TextPiece], overlap_tokens: int) -> list[_TextPiece]:
    if overlap_tokens <= 0 or not pieces:
        return []
    kept: list[_TextPiece] = []
    running = 0
    for piece in reversed(pieces):
        kept.insert(0, piece)
        running += piece.token_count
        if running >= overlap_tokens:
            break
    return kept


def _compose_paper_text_and_offsets(sections: list[SectionRecord]) -> tuple[str, dict[str, tuple[int, int]]]:
    section_offsets: dict[str, tuple[int, int]] = {}
    parts: list[str] = []
    cursor = 0
    for index, section in enumerate(sections):
        prefix = f"{section.section_title}\n"
        part = f"{prefix}{section.text}"
        text_start = cursor + len(prefix)
        text_end = text_start + len(section.text)
        section_offsets[section.section_id] = (text_start, text_end)
        parts.append(part)
        cursor += len(part)
        if index != len(sections) - 1:
            cursor += 2
    return "\n\n".join(parts), section_offsets


def _assign_chunk_char_offsets(
    chunks: list[ChunkRecord],
    sections: list[SectionRecord],
    section_text_offsets: dict[str, tuple[int, int]],
) -> None:
    chunks_by_section: dict[str, list[ChunkRecord]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_section[chunk.section_id].append(chunk)

    for section in sections:
        if not section.text:
            continue
        text_start, _ = section_text_offsets.get(section.section_id, (0, 0))
        search_cursor = 0
        for chunk in chunks_by_section.get(section.section_id, []):
            relative_index = section.text.find(chunk.text, search_cursor)
            if relative_index < 0:
                relative_index = section.text.find(chunk.text)
            if relative_index < 0:
                chunk.char_start = text_start
                chunk.char_end = text_start + min(len(chunk.text), len(section.text))
                continue
            chunk.char_start = text_start + relative_index
            chunk.char_end = chunk.char_start + len(chunk.text)
            search_cursor = relative_index + 1


def _select_intro_summary(sections: list[SectionRecord], abstract: str) -> str:
    preferred = ("abstract", "introduction", "overview")
    for name in preferred:
        for section in sections:
            if name in section.section_title.lower() and section.text:
                return truncate_text(section.text, limit=1200)
    for section in sections:
        if section.text:
            return truncate_text(section.text, limit=1200)
    return truncate_text(abstract, limit=1200)


def _non_empty_source_fields(*items: tuple[str, str]) -> list[str]:
    return [name for name, value in items if normalize_whitespace(value)]


def _bbox_center_distance(left: list[float], right: list[float]) -> float:
    left_cx = (left[0] + left[2]) / 2.0
    left_cy = (left[1] + left[3]) / 2.0
    right_cx = (right[0] + right[2]) / 2.0
    right_cy = (right[1] + right[3]) / 2.0
    return ((left_cx - right_cx) ** 2 + (left_cy - right_cy) ** 2) ** 0.5
