from __future__ import annotations

from chemverify.models import ObjectRecord, PaperRecord
from chemverify.presentation import (
    build_matched_sections_summary,
    extract_author_metadata,
    structure_rationale_text,
)


def test_extract_author_metadata_from_front_matter_markers() -> None:
    paper = PaperRecord(
        paper_id="2024.acl-long.1",
        title="Quantized Side Tuning",
        authors=["Zhengxin Zhang", "Dan Zhao", "Xupeng Miao"],
        venue="acl",
        year=2024,
        track="long",
        abstract="Abstract",
        url="https://aclanthology.org/2024.acl-long.1/",
        text="",
    )
    objects = [
        ObjectRecord(
            object_id="obj_front",
            paper_id=paper.paper_id,
            section_id="section_front",
            object_type="text_block",
            page_idx=1,
            section_path=["Front Matter"],
            text=(
                "Zhengxin Zhang‡§, Dan Zhao♭, Xupeng Miao‡ "
                "‡Carnegie Mellon University, §Tsinghua University, ♭Peng Cheng Laboratory "
                "zhang@example.com"
            ),
        )
    ]

    authors, affiliations, structured = extract_author_metadata(paper, objects)

    assert authors == ["Zhengxin Zhang", "Dan Zhao", "Xupeng Miao"]
    assert affiliations == [
        "Carnegie Mellon University",
        "Tsinghua University",
        "Peng Cheng Laboratory",
    ]
    assert structured[0].affiliation == "Carnegie Mellon University; Tsinghua University"
    assert structured[1].affiliation == "Peng Cheng Laboratory"
    assert structured[2].affiliation == "Carnegie Mellon University"


def test_extract_author_metadata_falls_back_to_single_affiliation() -> None:
    paper = PaperRecord(
        paper_id="2025.acl-long.2",
        title="Example",
        authors=["Alice Smith", "Bob Lee"],
        venue="acl",
        year=2025,
        track="long",
        abstract="Abstract",
        url="https://aclanthology.org/2025.acl-long.2/",
        text="Alice Smith, Bob Lee Example University alice@example.com",
    )

    authors, affiliations, structured = extract_author_metadata(paper, [])

    assert authors == ["Alice Smith", "Bob Lee"]
    assert affiliations == ["Example University"]
    assert [author.affiliation for author in structured] == ["Example University", "Example University"]


def test_structure_rationale_and_matched_sections_summary() -> None:
    rationale = (
        "This paper directly evaluates on the GAIA benchmark. "
        "It reports dataset-specific results in the experiments section. "
        "It also compares against strong baselines."
    )
    structured = structure_rationale_text(rationale)

    assert structured is not None
    assert structured.main_reason == "This paper directly evaluates on the GAIA benchmark."
    assert structured.matching_points == [
        "It reports dataset-specific results in the experiments section.",
        "It also compares against strong baselines.",
    ]

    summary = build_matched_sections_summary(
        [
            {"section_title": "Experiments"},
            {"section_title": "Experiments"},
            {"section_title": "Introduction"},
        ]
    )
    assert summary == {"Experiments": 2, "Introduction": 1}
