from __future__ import annotations

from chemverify.deep_chat.evidence import DeepChatEvidenceMaterializer
from chemverify.models import ChunkRecord, ObjectRecord, PaperRecord, SectionRecord


def test_evidence_materializer_builds_structured_units() -> None:
    paper = PaperRecord(
        paper_id="paper-1",
        title="Structured Deep Chat",
        authors=["Ada Lovelace"],
        venue="acl",
        year=2025,
        url="https://example.com/paper-1",
    )
    section = SectionRecord(
        section_id="section-1",
        paper_id="paper-1",
        section_title="Experiments",
        section_path=["Experiments"],
        ordinal=1,
        page_start=2,
        page_end=3,
        text="We evaluate on GAIA and report results in a table.",
        member_object_ids=["obj-text", "obj-list", "obj-table"],
        member_chunk_ids=["chunk-1"],
    )
    objects = [
        ObjectRecord(
            object_id="obj-text",
            paper_id="paper-1",
            section_id="section-1",
            object_type="text_block",
            ordinal=1,
            page_idx=2,
            section_path=["Experiments"],
            text="We evaluate on GAIA and report strong results.",
        ),
        ObjectRecord(
            object_id="obj-list",
            paper_id="paper-1",
            section_id="section-1",
            object_type="list_block",
            ordinal=2,
            page_idx=2,
            section_path=["Experiments"],
            text="1. Use GAIA benchmark 2. Compare against GPT-4 3. Report win rate",
        ),
        ObjectRecord(
            object_id="obj-table",
            paper_id="paper-1",
            section_id="section-1",
            object_type="table_block",
            ordinal=3,
            page_idx=3,
            section_path=["Experiments"],
            text="Table 1 shows GAIA results.",
            caption="Table 1: GAIA benchmark results.",
            html="<table><tr><td>GAIA</td><td>72.1</td></tr></table>",
        ),
    ]
    chunks = [
        ChunkRecord(
            chunk_id="chunk-1",
            paper_id="paper-1",
            section_id="section-1",
            chunk_type="text_chunk",
            member_object_ids=["obj-text", "obj-list"],
            heading="Experiments",
            section_path=["Experiments"],
            page_start=2,
            page_end=2,
            text="We evaluate on GAIA and compare against GPT-4.",
        )
    ]

    units = DeepChatEvidenceMaterializer().build(
        papers=[paper],
        sections=[section],
        objects=objects,
        chunks=chunks,
    )
    unit_types = [unit.evidence_type for unit in units]

    assert "section_unit" in unit_types
    assert "paragraph_unit" in unit_types
    assert "list_unit" in unit_types
    assert "list_item_unit" in unit_types
    assert "table_unit" in unit_types
    assert "chunk_unit" in unit_types
