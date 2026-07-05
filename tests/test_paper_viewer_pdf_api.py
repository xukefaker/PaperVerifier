from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from chemverify.api import app
from chemverify.config import Settings
from chemverify.indexer import IndexBuilder
from chemverify.models import PaperRecord
from chemverify.search_current import rebuild_search_current
from chemverify.storage import LocalStore


def _json_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _write_mineru_artifacts(
    settings: Settings,
    *,
    paper_id: str,
    title: str,
    sections: list[tuple[str, str]],
) -> Path:
    parse_dir = settings.mineru_output_dir / paper_id / "auto"
    parse_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = settings.root_dir / f"{paper_id}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    (parse_dir / f"{paper_id}.md").write_text(f"# {title}\n", encoding="utf-8")

    first_page_blocks = [{"type": "title", "text": title}]
    for heading, text in sections:
        first_page_blocks.append({"type": "title", "text": heading})
        first_page_blocks.append({"type": "text", "text": text})

    (parse_dir / f"{paper_id}_middle.json").write_text(
        _json_dumps(
            {
                "pdf_info": [
                    {"page_idx": 0, "page_size": [595, 841], "para_blocks": first_page_blocks},
                    {"page_idx": 1, "page_size": [595, 841], "para_blocks": []},
                ]
            }
        ),
        encoding="utf-8",
    )
    (parse_dir / f"{paper_id}_content_list.json").write_text("[]", encoding="utf-8")
    return pdf_path


def _seed_fixture_data(tmp_path: Path) -> Settings:
    (tmp_path / "config.toml").write_text(
        """
[openai]
enabled = true
base_url = "https://api.openai.com/v1"
model = "gpt-5.4-mini"
""",
        encoding="utf-8",
    )
    settings = Settings.from_env(tmp_path)
    store = LocalStore(settings)
    pdf_path = _write_mineru_artifacts(
        settings,
        paper_id="acl2025.test.3",
        title="Agentic Reasoning and Planning with Large Language Models on GAIA",
        sections=[
            ("1 Introduction", "We study agentic reasoning."),
            ("2 Experiments", "We evaluate on the GAIA benchmark and report results in Table 3."),
            ("3 Results", "GAIA benchmark results show the model improves over strong baselines."),
        ],
    )
    store.save_raw_papers(
        [
            PaperRecord.model_validate(
                {
                    "paper_id": "acl2025.test.3",
                    "anthology_id": "acl2025.test.3",
                    "title": "Agentic Reasoning and Planning with Large Language Models on GAIA",
                    "authors": ["Katherine Johnson"],
                    "venue": "acl",
                    "year": 2025,
                    "track": "long",
                    "abstract": "We evaluate agentic reasoning systems on the GAIA benchmark and report detailed results.",
                    "url": "https://example.com/3",
                    "local_pdf_path": str(pdf_path),
                }
            )
        ]
    )
    return settings


def _build_and_refresh_search_current(settings: Settings) -> None:
    store = LocalStore(settings)
    IndexBuilder(settings, store).build()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    (settings.state_dir / "job_state.json").write_text(
        json.dumps(
            {
                "job_id": "test_job",
                "corpus": settings.corpus.to_dict(),
                "status": "completed",
                "updated_at": "2026-03-28T00:00:00+00:00",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    rebuild_search_current(settings.root_dir)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _augment_with_multi_page_evidence(settings: Settings) -> None:
    objects_path = settings.search_current_dir / "normalized" / "objects.jsonl"
    evidence_path = settings.search_current_dir / "normalized" / "deep_chat" / "evidence_units.jsonl"

    objects = _read_jsonl(objects_path)
    first_object = next(row for row in objects if row.get("paper_id") == "acl2025.test.3")
    first_object["page_idx"] = 1
    first_object["bbox"] = [84.0, 228.0, 286.0, 612.0]
    second_object = dict(first_object)
    second_object["object_id"] = "obj_multi_page_fixture"
    second_object["page_idx"] = 2
    second_object["bbox"] = [112.0, 146.0, 448.0, 372.0]
    objects.append(second_object)
    _write_jsonl(objects_path, objects)

    evidence_units = _read_jsonl(evidence_path)
    evidence_units.append(
        {
            "evidence_id": "evidence_multi_page_fixture",
            "paper_id": "acl2025.test.3",
            "evidence_type": "chunk_unit",
            "section_id": first_object["section_id"],
            "heading": "Results",
            "section_path": list(first_object.get("section_path", [])),
            "page_start": 1,
            "page_end": 2,
            "text": "Combined evidence across two pages for PDF navigation validation.",
            "html": "",
            "object_ids": [first_object["object_id"], second_object["object_id"]],
            "chunk_ids": [],
            "metadata": {"fixture": True},
        }
    )
    _write_jsonl(evidence_path, evidence_units)


def test_paper_viewer_api_returns_pdf_navigation_targets(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    _build_and_refresh_search_current(settings)
    _augment_with_multi_page_evidence(settings)

    with TestClient(app) as client:
        response = client.get("/api/papers/acl2025.test.3/viewer")

    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_id"] == "acl2025.test.3"
    assert payload["pdf_url"] == "/api/papers/acl2025.test.3/pdf"

    navigation_target = payload["evidence_navigation_map"]["evidence_multi_page_fixture"]
    assert navigation_target["markdown_target"]["block_id"]

    pdf_target = navigation_target["pdf_target"]
    assert pdf_target["primary_page"] == 1
    assert [page["page"] for page in pdf_target["pages"]] == [1, 2]
    assert all(page["width"] == 595.0 and page["height"] == 841.0 for page in pdf_target["pages"])
    assert all(page["bboxes"] for page in pdf_target["pages"])


def test_paper_pdf_api_serves_pdf_bytes(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    _build_and_refresh_search_current(settings)

    with TestClient(app) as client:
        response = client.get("/api/papers/acl2025.test.3/pdf")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF-1.4")
