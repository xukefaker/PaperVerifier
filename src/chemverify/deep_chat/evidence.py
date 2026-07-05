from __future__ import annotations

import re
from collections import defaultdict

from ..models import ChunkRecord, ObjectRecord, PaperRecord, SectionRecord
from ..utils import make_stable_id, normalize_whitespace, truncate_text
from .models import EvidenceUnit

_APPENDIX_PATTERN = re.compile(r"\b(appendix|supplementary|supplement)\b", re.IGNORECASE)
_LIST_SPLIT_PATTERN = re.compile(r"(?:(?<=^)|(?<=\s))(?:(?:[-*•])|(?:\d+\.))\s+")


def build_evidence_search_text(unit: EvidenceUnit) -> str:
    evidence_terms = [unit.evidence_type.replace("_", " ")]
    if unit.metadata.get("section_kind"):
        evidence_terms.append(str(unit.metadata["section_kind"]).replace("_", " "))
    parts = [
        unit.heading,
        " ".join(unit.section_path),
        unit.text,
        unit.html,
        " ".join(evidence_terms),
    ]
    return " ".join(part for part in parts if part)


class DeepChatEvidenceMaterializer:
    def build(
        self,
        *,
        papers: list[PaperRecord],
        sections: list[SectionRecord],
        objects: list[ObjectRecord],
        chunks: list[ChunkRecord],
    ) -> list[EvidenceUnit]:
        paper_lookup = {paper.paper_id: paper for paper in papers}
        sections_by_paper: dict[str, list[SectionRecord]] = defaultdict(list)
        section_lookup = {section.section_id: section for section in sections}
        objects_by_section: dict[str, list[ObjectRecord]] = defaultdict(list)
        chunks_by_section: dict[str, list[ChunkRecord]] = defaultdict(list)
        for section in sections:
            sections_by_paper[section.paper_id].append(section)
        for obj in objects:
            objects_by_section[obj.section_id].append(obj)
        for chunk in chunks:
            chunks_by_section[chunk.section_id].append(chunk)

        units: list[EvidenceUnit] = []
        for paper_id, paper_sections in sections_by_paper.items():
            if paper_id not in paper_lookup:
                continue
            for section in sorted(paper_sections, key=lambda item: item.ordinal):
                units.extend(
                    self._build_section_units(
                        paper_id=paper_id,
                        section=section,
                        section_objects=sorted(objects_by_section.get(section.section_id, []), key=lambda item: item.ordinal),
                        section_chunks=sorted(
                            chunks_by_section.get(section.section_id, []),
                            key=lambda item: (item.page_start, item.page_end, item.chunk_id),
                        ),
                    )
                )
        return units

    def _build_section_units(
        self,
        *,
        paper_id: str,
        section: SectionRecord,
        section_objects: list[ObjectRecord],
        section_chunks: list[ChunkRecord],
    ) -> list[EvidenceUnit]:
        units: list[EvidenceUnit] = []
        section_kind = "appendix" if _is_appendix_section(section) else "body"
        if section.text:
            units.append(
                EvidenceUnit(
                    evidence_id=make_stable_id("evidence", f"{paper_id}:{section.section_id}:section"),
                    paper_id=paper_id,
                    evidence_type="section_unit",
                    section_id=section.section_id,
                    heading=section.section_title,
                    section_path=list(section.section_path),
                    page_start=section.page_start,
                    page_end=section.page_end,
                    text=section.text,
                    object_ids=list(section.member_object_ids),
                    chunk_ids=list(section.member_chunk_ids),
                    metadata={
                        "section_kind": section_kind,
                        "ordinal": section.ordinal,
                    },
                )
            )

        for obj in section_objects:
            if not obj.text:
                continue
            base_metadata = {
                "section_kind": section_kind,
                "object_type": obj.object_type,
                "source_fields": list(obj.source_fields),
            }
            if obj.object_type == "text_block":
                units.append(
                    EvidenceUnit(
                        evidence_id=make_stable_id("evidence", f"{paper_id}:{obj.object_id}:paragraph"),
                        paper_id=paper_id,
                        evidence_type="paragraph_unit",
                        section_id=section.section_id,
                        heading=section.section_title,
                        section_path=list(section.section_path),
                        page_start=obj.page_idx,
                        page_end=obj.page_idx,
                        text=obj.text,
                        object_ids=[obj.object_id],
                        metadata=base_metadata,
                    )
                )
            elif obj.object_type == "list_block":
                units.append(
                    EvidenceUnit(
                        evidence_id=make_stable_id("evidence", f"{paper_id}:{obj.object_id}:list"),
                        paper_id=paper_id,
                        evidence_type="list_unit",
                        section_id=section.section_id,
                        heading=section.section_title,
                        section_path=list(section.section_path),
                        page_start=obj.page_idx,
                        page_end=obj.page_idx,
                        text=obj.text,
                        object_ids=[obj.object_id],
                        metadata=base_metadata,
                    )
                )
                for index, item_text in enumerate(_split_list_items(obj.text), start=1):
                    units.append(
                        EvidenceUnit(
                            evidence_id=make_stable_id("evidence", f"{paper_id}:{obj.object_id}:list_item:{index}"),
                            paper_id=paper_id,
                            evidence_type="list_item_unit",
                            section_id=section.section_id,
                            heading=section.section_title,
                            section_path=list(section.section_path),
                            page_start=obj.page_idx,
                            page_end=obj.page_idx,
                            text=item_text,
                            object_ids=[obj.object_id],
                            metadata={**base_metadata, "list_item_index": index},
                        )
                    )
            elif obj.object_type == "table_block":
                units.append(
                    EvidenceUnit(
                        evidence_id=make_stable_id("evidence", f"{paper_id}:{obj.object_id}:table"),
                        paper_id=paper_id,
                        evidence_type="table_unit",
                        section_id=section.section_id,
                        heading=obj.caption or section.section_title,
                        section_path=list(section.section_path),
                        page_start=obj.page_idx,
                        page_end=obj.page_idx,
                        text=obj.text,
                        html=obj.html,
                        object_ids=[obj.object_id],
                        metadata={
                            **base_metadata,
                            "caption": obj.caption,
                            "footnote": obj.footnote,
                        },
                    )
                )

        for chunk in section_chunks:
            if not chunk.text:
                continue
            units.append(
                EvidenceUnit(
                    evidence_id=make_stable_id("evidence", f"{paper_id}:{chunk.chunk_id}:chunk"),
                    paper_id=paper_id,
                    evidence_type="chunk_unit",
                    section_id=section.section_id,
                    heading=chunk.heading,
                    section_path=list(chunk.section_path),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    object_ids=list(chunk.member_object_ids),
                    chunk_ids=[chunk.chunk_id],
                    metadata={
                        "section_kind": section_kind,
                        "chunk_type": chunk.chunk_type,
                        "evidence_types": list(chunk.evidence_types),
                    },
                )
            )
        return units


def summarize_evidence_snippet(unit: EvidenceUnit, *, limit: int = 260) -> str:
    return truncate_text(unit.text, limit=limit)


def _is_appendix_section(section: SectionRecord) -> bool:
    combined = " ".join([section.section_title, *section.section_path])
    return bool(_APPENDIX_PATTERN.search(combined))


def _split_list_items(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    matches = list(_LIST_SPLIT_PATTERN.finditer(normalized))
    if not matches:
        if "; " in normalized:
            parts = [part.strip() for part in normalized.split("; ") if part.strip()]
            return parts if len(parts) >= 2 else []
        return []
    starts = [match.start() for match in matches] + [len(normalized)]
    items: list[str] = []
    for left, right in zip(starts, starts[1:], strict=False):
        item_text = normalize_whitespace(normalized[left:right])
        if item_text:
            items.append(item_text)
    return items
