from __future__ import annotations

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from paper_search_agent.config import Settings


class _NoopSearchEngine:
    def __init__(self, settings, store) -> None:
        self.settings = settings
        self.store = store


class _NoopDeepChatService:
    def __init__(self, settings, store, engine, **_: object) -> None:
        self.settings = settings
        self.store = store
        self.engine = engine

    def close(self) -> None:
        return None


def _write_search_current_snapshot(root_dir: Path, *, build_id: str) -> Settings:
    settings = Settings.from_env(root_dir)
    index_dir = settings.search_current_dir / "indexes" / "layout"
    normalized_dir = settings.search_current_dir / "normalized"
    trace_dir = settings.search_current_dir / "traces"
    index_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index_state.json").write_text(
        json.dumps({"papers": 1, "chunks": 1}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paper_pdf = root_dir / "2024.acl-long.1.pdf"
    paper_pdf.write_bytes(b"%PDF-1.4\n%fake zotero export\n")
    (normalized_dir / "papers.jsonl").write_text(
        json.dumps(
            {
                "paper_id": "2024.acl-long.1",
                "anthology_id": "2024.acl-long.1",
                "title": "Zotero Exportable Paper",
                "authors": ["Ada Lovelace", "Grace Hopper"],
                "venue": "acl",
                "year": 2024,
                "track": "long",
                "abstract": "A paper used to verify Zotero export endpoints.",
                "doi": "10.1000/zotero-test",
                "url": "https://aclanthology.org/2024.acl-long.1/",
                "local_pdf_path": str(paper_pdf),
                "pdf_url": "https://aclanthology.org/2024.acl-long.1.pdf",
                "metadata": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    settings.search_current_manifest_path.write_text(
        json.dumps(
            {
                "build_id": build_id,
                "corpora": [],
                "counts": {"papers": 1, "chunks": 1},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return settings


def _create_test_client(tmp_path: Path, monkeypatch):
    (tmp_path / "config.toml").write_text(
        """
[openai]
enabled = false
base_url = "https://api.openai.com/v1"
model = "gpt-5.4-mini"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PAPER_SEARCH_AGENT_DATA_DIR", str(tmp_path / "data"))
    _write_search_current_snapshot(tmp_path, build_id="build-zotero")

    app_module = importlib.import_module("paper_search_agent.api.app")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "SearchEngine", _NoopSearchEngine)
    monkeypatch.setattr(app_module, "DeepChatService", _NoopDeepChatService)
    app = app_module.create_app()
    return TestClient(app)


def test_zotero_export_endpoints_return_expected_content(tmp_path: Path, monkeypatch) -> None:
    with _create_test_client(tmp_path, monkeypatch) as client:
        html_response = client.get("/api/papers/2024.acl-long.1/zotero")
        assert html_response.status_code == 200
        assert "text/html" in html_response.headers["content-type"]
        assert 'citation_title' in html_response.text
        assert 'Download BibTeX' in html_response.text

        bib_response = client.get("/api/papers/2024.acl-long.1/export.bib")
        assert bib_response.status_code == 200
        assert "@inproceedings" in bib_response.text
        assert "Zotero Exportable Paper" in bib_response.text

        ris_response = client.get("/api/papers/2024.acl-long.1/export.ris")
        assert ris_response.status_code == 200
        assert "TY  - CONF" in ris_response.text
        assert "DO  - 10.1000/zotero-test" in ris_response.text
