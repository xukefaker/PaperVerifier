from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from ...deep_chat.store import DeepChatStore
from ...models import (
    EvidenceNavigationTarget,
    ObjectRecord,
    PaperRecord,
    SectionRecord,
    ViewerMarkdownTarget,
    ViewerPdfPageTarget,
    ViewerPdfRect,
    ViewerPdfTarget,
)
from ...presentation import (
    build_main_image_url,
    load_cached_paper_authorship,
    load_cached_paper_references,
    resolve_paper_image_path,
)
from ...zotero_export import (
    build_public_export_path,
    build_zotero_export_payload,
    render_bibtex,
    render_ris,
    render_zotero_metadata_page,
)

router = APIRouter()

_VIEWER_FRONT_MATTER_MAX_Y = 230.0
_VIEWER_AFFILIATION_KEYWORDS = (
    "university",
    "institute",
    "laboratory",
    "lab",
    "school",
    "department",
    "college",
    "faculty",
    "research",
    "amazon",
    "google",
    "meta",
    "microsoft",
    "openai",
    "anthropic",
    "deepmind",
    "nvidia",
    "bytedance",
    "tencent",
    "alibaba",
    "hkust",
)


class _ViewerTableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, object]]] = []
        self._current_row: list[dict[str, object]] | None = None
        self._current_cell: dict[str, object] | None = None
        self._cell_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if tag == "tr":
            self._current_row = []
            return
        if tag in {"td", "th"}:
            self._current_cell = {
                "text": "",
                "colspan": _safe_positive_int(attr_map.get("colspan"), default=1),
                "rowspan": _safe_positive_int(attr_map.get("rowspan"), default=1),
                "is_header": tag == "th",
            }
            self._cell_chunks = []
            return
        if tag == "br" and self._current_cell is not None:
            self._cell_chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._cell_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_cell is not None:
            cell_text = _normalize_viewer_text("".join(self._cell_chunks))
            self._current_cell["text"] = cell_text
            if self._current_row is not None:
                self._current_row.append(self._current_cell)
            self._current_cell = None
            self._cell_chunks = []
            return
        if tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None


def _safe_positive_int(raw_value: object, *, default: int) -> int:
    try:
        parsed = int(str(raw_value or "").strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _normalize_viewer_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_lookup_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_viewer_text(value).lower())


def _build_viewer_display_header(settings, paper: PaperRecord) -> dict[str, object]:
    _, affiliations, authors_structured = load_cached_paper_authorship(settings, paper)
    return {
        "authors_structured": [
            {
                "name": author.name,
                "affiliation": author.affiliation,
            }
            for author in authors_structured
        ],
        "affiliations": affiliations,
    }


def _find_title_section_ids(section_rows: list[dict[str, object]], paper_title: str) -> set[str]:
    title_key = _normalize_lookup_key(paper_title)
    if not title_key:
        return set()

    title_section_ids: set[str] = set()
    for section in section_rows:
        section_id = str(section.get("section_id") or "").strip()
        if not section_id:
            continue
        section_title_key = _normalize_lookup_key(section.get("section_title"))
        section_path = [str(item) for item in section.get("section_path", []) if str(item).strip()]
        path_head_key = _normalize_lookup_key(section_path[0]) if section_path else ""
        if section_title_key == title_key or path_head_key == title_key:
            title_section_ids.add(section_id)
    return title_section_ids


def _looks_like_front_matter_text(text: str, author_names: list[str], affiliations: list[str]) -> bool:
    normalized = _normalize_viewer_text(text)
    if not normalized:
        return False

    lowered = normalized.lower()
    if "@" in normalized:
        return True

    author_hits = sum(1 for name in author_names if name and name.lower() in lowered)
    if author_hits >= 2:
        return True

    affiliation_hits = sum(1 for affiliation in affiliations if affiliation and affiliation.lower() in lowered)
    if affiliation_hits >= 1:
        return True

    if any(keyword in lowered for keyword in _VIEWER_AFFILIATION_KEYWORDS):
        if re.search(r"(^|\s)[0-9*†‡§♠♣♡◊]", normalized) or "," in normalized:
            return True
    return False


def _is_front_matter_object(
    obj: ObjectRecord,
    *,
    title_section_ids: set[str],
    author_names: list[str],
    affiliations: list[str],
) -> bool:
    if obj.section_id in title_section_ids:
        return True
    if obj.page_idx != 1 or obj.object_type not in {"text_block", "list_block"}:
        return False
    bbox_top = float(obj.bbox[1]) if len(obj.bbox) >= 2 else None
    if bbox_top is not None and bbox_top > _VIEWER_FRONT_MATTER_MAX_Y:
        return False
    return _looks_like_front_matter_text(obj.text, author_names, affiliations)


def _parse_table_payload(table_text: str) -> dict[str, object] | None:
    if "<table" not in table_text.lower():
        return None
    parser = _ViewerTableHTMLParser()
    try:
        parser.feed(table_text)
        parser.close()
    except Exception:
        return None

    rows = []
    for row in parser.rows:
        if not row:
            continue
        rows.append({"cells": row})
    if not rows:
        return None
    return {"rows": rows}


def _build_viewer_references(settings, paper_id: str) -> list[dict[str, object]]:
    record = load_cached_paper_references(settings, paper_id)
    if record is None:
        return []

    references: list[dict[str, object]] = []
    seen_texts: set[str] = set()
    for entry in record.references:
        raw_text = _normalize_viewer_text(entry.raw_text)
        if not raw_text:
            continue

        lowered = raw_text.lower()
        if any(tag in lowered for tag in ("<table", "<tr", "<td", "<th", "</table", "</tr", "</td", "</th")):
            continue
        if len(raw_text) < 20 or len(raw_text) > 1200:
            continue

        lookup_key = lowered
        if lookup_key in seen_texts:
            continue
        seen_texts.add(lookup_key)
        references.append(
            {
                "ordinal": int(entry.ordinal),
                "raw_text": raw_text,
                "year": entry.year,
                "doi": entry.doi,
                "arxiv_id": entry.arxiv_id,
                "url": entry.url,
            }
        )
    return references


def _build_viewer_pdf_url(settings, paper: PaperRecord) -> str | None:
    if not paper.local_pdf_path:
        return None
    pdf_path = Path(paper.local_pdf_path).expanduser()
    if not pdf_path.is_file():
        return None

    base_url = settings.public_api_base_url.rstrip("/") if settings.public_api_base_url else None
    if base_url:
        return f"{base_url}/papers/{quote(paper.paper_id, safe='')}/pdf"
    return f"/api/papers/{quote(paper.paper_id, safe='')}/pdf"


def _download_filename(paper_id: str, suffix: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", paper_id).strip("._") or "paper"
    return f"{normalized}.{suffix}"


@lru_cache(maxsize=256)
def _load_viewer_pdf_page_sizes(middle_json_path_str: str, mtime_ns: int) -> dict[int, tuple[float, float]]:
    path = Path(middle_json_path_str)
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    pdf_info = payload.get("pdf_info")
    if not isinstance(pdf_info, list) or not pdf_info:
        return {}

    indexed_pages: list[tuple[int, list[object]]] = []
    for page in pdf_info:
        if not isinstance(page, dict):
            continue
        raw_idx = page.get("page_idx")
        raw_page_size = page.get("page_size")
        if not isinstance(raw_page_size, list) or len(raw_page_size) < 2:
            continue
        try:
            page_idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        indexed_pages.append((page_idx, raw_page_size))

    if not indexed_pages:
        return {}

    page_index_offset = 1 if min(page_idx for page_idx, _ in indexed_pages) == 0 else 0
    page_sizes: dict[int, tuple[float, float]] = {}
    for page_idx, raw_page_size in indexed_pages:
        try:
            width = float(raw_page_size[0])
            height = float(raw_page_size[1])
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        page_sizes[page_idx + page_index_offset] = (width, height)
    return page_sizes


def _load_viewer_pdf_page_size_lookup(paper: PaperRecord) -> dict[int, tuple[float, float]]:
    middle_json_path_str = paper.metadata.get("mineru_artifacts", {}).get("middle_json")
    if not middle_json_path_str:
        return {}

    middle_json_path = Path(middle_json_path_str)
    if not middle_json_path.exists():
        return {}

    return _load_viewer_pdf_page_sizes(
        str(middle_json_path),
        middle_json_path.stat().st_mtime_ns,
    )


def _build_viewer_pdf_target(
    *,
    object_ids: list[str],
    object_by_id: dict[str, ObjectRecord],
    page_size_lookup: dict[int, tuple[float, float]],
) -> ViewerPdfTarget | None:
    page_rects: dict[int, list[ViewerPdfRect]] = defaultdict(list)
    seen_object_ids: set[str] = set()

    for object_id in object_ids:
        normalized_object_id = str(object_id).strip()
        if not normalized_object_id or normalized_object_id in seen_object_ids:
            continue
        seen_object_ids.add(normalized_object_id)

        obj = object_by_id.get(normalized_object_id)
        if obj is None or len(obj.bbox) < 4:
            continue

        try:
            page = int(obj.page_idx)
            x0, y0, x1, y1 = (float(obj.bbox[0]), float(obj.bbox[1]), float(obj.bbox[2]), float(obj.bbox[3]))
        except (TypeError, ValueError):
            continue
        if page not in page_size_lookup and (page + 1) in page_size_lookup:
            page += 1
        if page <= 0:
            continue
        page_rects[page].append(ViewerPdfRect(x0=x0, y0=y0, x1=x1, y1=y1))

    if not page_rects:
        return None

    page_targets: list[ViewerPdfPageTarget] = []
    for page in sorted(page_rects):
        page_size = page_size_lookup.get(page)
        if page_size is None:
            continue
        width, height = page_size
        page_targets.append(
            ViewerPdfPageTarget(
                page=page,
                width=width,
                height=height,
                bboxes=page_rects[page],
            )
        )

    if not page_targets:
        return None

    return ViewerPdfTarget(
        primary_page=page_targets[0].page,
        pages=page_targets,
    )


def _build_evidence_navigation_map(
    *,
    evidence_specs: list[dict[str, object]],
    section_block_id_by_section_id: dict[str, str],
    object_block_id_by_object_id: dict[str, str],
    object_by_id: dict[str, ObjectRecord],
    page_size_lookup: dict[int, tuple[float, float]],
) -> dict[str, dict[str, object]]:
    navigation_map: dict[str, dict[str, object]] = {}
    section_block_ids = set(section_block_id_by_section_id.values())

    for spec in evidence_specs:
        evidence_id = str(spec.get("evidence_id") or "").strip()
        if not evidence_id:
            continue

        markdown_block_id: str | None = None
        for object_id in spec.get("object_ids", []):
            markdown_block_id = object_block_id_by_object_id.get(str(object_id))
            if markdown_block_id is not None:
                break

        section_id = str(spec.get("section_id") or "").strip()
        section_block_id = section_block_id_by_section_id.get(section_id) if section_id else None

        if markdown_block_id is None:
            markdown_block_id = section_block_id
        if markdown_block_id is None:
            continue

        if section_block_id is None and markdown_block_id in section_block_ids:
            section_block_id = markdown_block_id

        pdf_target = _build_viewer_pdf_target(
            object_ids=[str(item) for item in spec.get("object_ids", [])],
            object_by_id=object_by_id,
            page_size_lookup=page_size_lookup,
        )

        navigation_map[evidence_id] = EvidenceNavigationTarget(
            evidence_id=evidence_id,
            markdown_target=ViewerMarkdownTarget(
                block_id=markdown_block_id,
                section_block_id=section_block_id,
            ),
            pdf_target=pdf_target,
        ).model_dump()

    return navigation_map


@lru_cache(maxsize=4)
def _load_evidence_specs(evidence_unit_path_str: str, mtime_ns: int) -> dict[str, list[dict[str, object]]]:
    path = Path(evidence_unit_path_str)
    grouped: dict[str, list[dict[str, object]]] = {}
    if not path.exists():
        return grouped

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            paper_id = str(payload.get("paper_id") or "").strip()
            evidence_id = str(payload.get("evidence_id") or "").strip()
            if not paper_id or not evidence_id:
                continue
            grouped.setdefault(paper_id, []).append(
                {
                    "evidence_id": evidence_id,
                    "section_id": str(payload.get("section_id") or "").strip() or None,
                    "object_ids": [str(item) for item in payload.get("object_ids", []) if str(item).strip()],
                }
            )
    return grouped


def _section_row(section: SectionRecord, index: int) -> dict[str, object]:
    return {
        "section_id": section.section_id,
        "paper_id": section.paper_id,
        "section_title": section.section_title,
        "section_path": section.section_path,
        "page_start": section.page_start,
        "page_end": section.page_end,
        "text": section.text,
        "index": index,
    }


def _collect_paper_layout(
    *,
    services,
    paper_id: str,
) -> tuple[list[dict[str, object]], list[ObjectRecord]]:
    runtime = getattr(services.engine, "runtime", None)
    if runtime is not None:
        return (
            list(runtime.sections_by_paper.get(paper_id, [])),
            list(runtime.objects_by_paper.get(paper_id, [])),
        )

    sections = [
        _section_row(section, index)
        for index, section in enumerate(services.store.load_sections())
        if section.paper_id == paper_id
    ]
    objects = [obj for obj in services.store.load_objects() if obj.paper_id == paper_id]
    return sections, objects


def _build_viewer_blocks(
    *,
    settings,
    paper: PaperRecord,
    paper_id: str,
    section_rows: list[dict[str, object]],
    objects: list[ObjectRecord],
    author_names: list[str],
    affiliations: list[str],
) -> tuple[list[dict[str, object]], dict[str, str], dict[str, str]]:
    blocks: list[dict[str, object]] = []
    section_block_id_by_section_id: dict[str, str] = {}
    object_block_id_by_object_id: dict[str, str] = {}
    title_section_ids = _find_title_section_ids(section_rows, paper.title)

    objects_by_section: dict[str, list[ObjectRecord]] = defaultdict(list)
    for obj in objects:
        objects_by_section[obj.section_id].append(obj)
    for grouped_objects in objects_by_section.values():
        grouped_objects.sort(key=lambda item: (item.page_idx, item.ordinal, item.object_id))

    seen_sections: set[str] = set()
    for section in sorted(section_rows, key=lambda item: int(item.get("index", 0) or 0)):
        section_id = str(section.get("section_id") or "")
        if not section_id:
            continue
        seen_sections.add(section_id)
        if section_id in title_section_ids:
            continue
        section_block_id = f"section:{section_id}"
        section_block_id_by_section_id[section_id] = section_block_id
        blocks.append(
            {
                "block_id": section_block_id,
                "block_type": "section_heading",
                "page_start": int(section.get("page_start", 1) or 1),
                "page_end": int(section.get("page_end", 1) or 1),
                "section_path": list(section.get("section_path", [])),
                "text": str(section.get("section_title", "") or ""),
                "caption": None,
                "footnote": None,
                "image_url": None,
            }
        )

        for obj in objects_by_section.get(section_id, []):
            if _is_front_matter_object(
                obj,
                title_section_ids=title_section_ids,
                author_names=author_names,
                affiliations=affiliations,
            ):
                continue
            block_id = f"object:{obj.object_id}"
            object_block_id_by_object_id[obj.object_id] = block_id
            image_url = build_main_image_url(settings, paper_id, obj.image_path) if obj.image_path else None
            blocks.append(
                {
                    "block_id": block_id,
                    "block_type": obj.object_type,
                    "page_start": int(obj.page_idx),
                    "page_end": int(obj.page_idx),
                    "section_path": list(obj.section_path),
                    "text": obj.text,
                    "caption": obj.caption or None,
                    "footnote": obj.footnote or None,
                    "image_url": image_url or None,
                    "table": _parse_table_payload(obj.text) if obj.object_type == "table_block" else None,
                }
            )

    residual_objects = [obj for obj in objects if obj.section_id not in seen_sections]
    residual_objects.sort(key=lambda item: (item.page_idx, item.ordinal, item.object_id))
    for obj in residual_objects:
        if _is_front_matter_object(
            obj,
            title_section_ids=title_section_ids,
            author_names=author_names,
            affiliations=affiliations,
        ):
            continue
        block_id = f"object:{obj.object_id}"
        object_block_id_by_object_id[obj.object_id] = block_id
        image_url = build_main_image_url(settings, paper_id, obj.image_path) if obj.image_path else None
        blocks.append(
            {
                "block_id": block_id,
                "block_type": obj.object_type,
                "page_start": int(obj.page_idx),
                "page_end": int(obj.page_idx),
                "section_path": list(obj.section_path),
                "text": obj.text,
                "caption": obj.caption or None,
                "footnote": obj.footnote or None,
                "image_url": image_url or None,
                "table": _parse_table_payload(obj.text) if obj.object_type == "table_block" else None,
            }
        )

    return blocks, section_block_id_by_section_id, object_block_id_by_object_id


@router.get("/papers/{paper_id}/markdown")
def get_markdown(request: Request, paper_id: str):
    with request.app.state.service_manager.acquire_services() as services:
        paper = services.store.get_paper(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")

        md_path_str = paper.metadata.get("mineru_artifacts", {}).get("markdown")
        if not md_path_str:
            raise HTTPException(status_code=404, detail="Markdown artifact not found for this paper")

        md_path = Path(md_path_str)
        if not md_path.exists():
            raise HTTPException(status_code=404, detail="Markdown file missing on disk")

        content = md_path.read_text(encoding="utf-8")
        return Response(content=content, media_type="text/markdown")


@router.get("/papers/{paper_id}/viewer")
def get_paper_viewer(request: Request, paper_id: str):
    with request.app.state.service_manager.acquire_services() as services:
        paper = services.store.get_paper(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")

        authors, affiliations, _ = load_cached_paper_authorship(services.settings, paper)
        display_header = _build_viewer_display_header(services.settings, paper)
        references = _build_viewer_references(services.settings, paper_id)
        section_rows, objects = _collect_paper_layout(services=services, paper_id=paper_id)
        object_by_id = {obj.object_id: obj for obj in objects}
        page_size_lookup = _load_viewer_pdf_page_size_lookup(paper)
        blocks, section_block_id_by_section_id, object_block_id_by_object_id = _build_viewer_blocks(
            settings=services.settings,
            paper=paper,
            paper_id=paper_id,
            section_rows=section_rows,
            objects=objects,
            author_names=authors,
            affiliations=affiliations,
        )

        deep_chat_store = DeepChatStore(services.settings, root_dir=services.settings.search_current_dir)
        evidence_path = deep_chat_store.evidence_unit_path
        evidence_specs = _load_evidence_specs(
            str(evidence_path),
            evidence_path.stat().st_mtime_ns if evidence_path.exists() else 0,
        ).get(paper_id, [])
        evidence_navigation_map = _build_evidence_navigation_map(
            evidence_specs=evidence_specs,
            section_block_id_by_section_id=section_block_id_by_section_id,
            object_block_id_by_object_id=object_block_id_by_object_id,
            object_by_id=object_by_id,
            page_size_lookup=page_size_lookup,
        )

        return JSONResponse(
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "pdf_url": _build_viewer_pdf_url(services.settings, paper),
                "display_header": display_header,
                "blocks": blocks,
                "evidence_navigation_map": evidence_navigation_map,
                "references": references,
            }
        )


@router.get("/papers/{paper_id}/pdf")
def get_paper_pdf(request: Request, paper_id: str) -> FileResponse:
    with request.app.state.service_manager.acquire_services() as services:
        paper = services.store.get_paper(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        if not paper.local_pdf_path:
            raise HTTPException(status_code=404, detail="PDF artifact not found for this paper")

        pdf_path = Path(paper.local_pdf_path).expanduser()
        if not pdf_path.is_file():
            raise HTTPException(status_code=404, detail="PDF file missing on disk")

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"{paper.paper_id}.pdf",
        )


@router.get("/papers/{paper_id}/zotero")
def get_paper_zotero_metadata_page(request: Request, paper_id: str) -> HTMLResponse:
    with request.app.state.service_manager.acquire_services() as services:
        paper = services.store.get_paper(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")

        payload = build_zotero_export_payload(
            services.settings,
            paper,
            public_pdf_url=_build_viewer_pdf_url(services.settings, paper),
        )
        html_content = render_zotero_metadata_page(
            payload,
            bibtex_url=build_public_export_path(paper_id, "export.bib"),
            ris_url=build_public_export_path(paper_id, "export.ris"),
        )
        return HTMLResponse(content=html_content)


@router.get("/papers/{paper_id}/export.bib")
def get_paper_bibtex_export(request: Request, paper_id: str) -> Response:
    with request.app.state.service_manager.acquire_services() as services:
        paper = services.store.get_paper(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")

        payload = build_zotero_export_payload(
            services.settings,
            paper,
            public_pdf_url=_build_viewer_pdf_url(services.settings, paper),
        )
        bibtex = render_bibtex(payload)
        return Response(
            content=bibtex,
            media_type="text/x-bibtex; charset=utf-8",
            headers={
                "content-disposition": f'inline; filename="{_download_filename(paper_id, "bib")}"',
            },
        )


@router.get("/papers/{paper_id}/export.ris")
def get_paper_ris_export(request: Request, paper_id: str) -> Response:
    with request.app.state.service_manager.acquire_services() as services:
        paper = services.store.get_paper(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")

        payload = build_zotero_export_payload(
            services.settings,
            paper,
            public_pdf_url=_build_viewer_pdf_url(services.settings, paper),
        )
        ris = render_ris(payload)
        return Response(
            content=ris,
            media_type="application/x-research-info-systems; charset=utf-8",
            headers={
                "content-disposition": f'inline; filename="{_download_filename(paper_id, "ris")}"',
            },
        )


@router.get("/papers/{paper_id}/content_list")
def get_content_list(request: Request, paper_id: str):
    with request.app.state.service_manager.acquire_services() as services:
        paper = services.store.get_paper(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")

        json_path_str = paper.metadata.get("mineru_artifacts", {}).get("content_list_json")
        if not json_path_str:
            raise HTTPException(status_code=404, detail="Content list artifact not found")

        json_path = Path(json_path_str)
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="Content list file missing on disk")

        return JSONResponse(json.loads(json_path.read_text(encoding="utf-8")))


@router.get("/papers/{paper_id}/images/{image_name:path}")
def get_paper_image(request: Request, paper_id: str, image_name: str) -> FileResponse:
    with request.app.state.service_manager.acquire_services() as services:
        paper = services.store.get_paper(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        if Path(image_name).is_absolute() or ".." in Path(image_name).parts:
            raise HTTPException(status_code=404, detail="Image not found")

        runtime = getattr(services.engine, "runtime", None)
        objects = runtime.objects_by_paper.get(paper_id, []) if runtime is not None else [
            obj for obj in services.store.load_objects() if obj.paper_id == paper_id
        ]
        image_path = resolve_paper_image_path(services.settings, paper_id, image_name, objects)
        if image_path is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_path)


@router.get("/papers/{paper_id}", response_model=PaperRecord)
def get_paper(request: Request, paper_id: str) -> PaperRecord:
    with request.app.state.service_manager.acquire_services() as services:
        paper = services.store.get_paper(paper_id)
        if paper is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return paper
