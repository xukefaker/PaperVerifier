from __future__ import annotations

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from chemverify.config import Settings


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


def _write_search_current_snapshot(root_dir: Path, *, build_id: str, papers: int) -> None:
    settings = Settings.from_env(root_dir)
    index_dir = settings.search_current_dir / "indexes" / "layout"
    normalized_dir = settings.search_current_dir / "normalized"
    trace_dir = settings.search_current_dir / "traces"
    index_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index_state.json").write_text(
        json.dumps({"papers": papers, "chunks": papers * 10}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (normalized_dir / "papers.jsonl").write_text("", encoding="utf-8")
    settings.search_current_manifest_path.write_text(
        json.dumps(
            {
                "build_id": build_id,
                "corpora": [
                    {
                        "corpus": "acl/2024/long",
                        "papers": papers,
                        "chunks": papers * 10,
                        "deep_chat_evidence_units": papers * 20,
                    },
                    {
                        "corpus": "chemqa40/2026/all",
                        "papers": 40,
                        "chunks": 1056,
                        "deep_chat_evidence_units": 3204,
                    },
                ],
                "counts": {"papers": papers, "chunks": papers * 10},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


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
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(tmp_path / "data"))
    _write_search_current_snapshot(tmp_path, build_id="build-a", papers=0)

    app_module = importlib.import_module("chemverify.api.app")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "SearchEngine", _NoopSearchEngine)
    monkeypatch.setattr(app_module, "DeepChatService", _NoopDeepChatService)
    app = app_module.create_app()
    return TestClient(app)


def test_projects_api_crud_and_clear(tmp_path: Path, monkeypatch) -> None:
    with _create_test_client(tmp_path, monkeypatch) as client:
        listed = client.get("/api/projects")
        assert listed.status_code == 200
        projects = listed.json()["projects"]
        assert projects == []

        created = client.post("/api/projects", json={"title": "GAIA Survey"})
        assert created.status_code == 200
        project = created.json()
        assert project["project_id"].startswith("gaia-survey")
        assert project["selected_corpora"] == ["chemqa40/2026/all"]

        thread_upsert = client.put(
            f"/api/projects/{project['project_id']}/threads/search-1",
            json={
                "query": "Find ACL 2024 long papers that evaluate on GAIA.",
                "trace_id": "trace-gaia-1",
                "result_counts": {"satisfied": 1, "partial": 2, "rejected": 3},
                "paper_ids": ["2024.acl-long.1"],
            },
        )
        assert thread_upsert.status_code == 200
        assert thread_upsert.json()["trace_id"] == "trace-gaia-1"

        session_upsert = client.put(
            f"/api/projects/{project['project_id']}/papers/2024.acl-long.1/session",
            json={
                "paper_title": "Test Paper",
                "source_thread_id": "search-1",
                "chat_history": [
                    {
                        "role": "user",
                        "content": "What does this paper do?",
                        "citations": [],
                    }
                ],
                "last_active_evidence_id": "evidence_1",
            },
        )
        assert session_upsert.status_code == 200
        assert session_upsert.json()["paper_id"] == "2024.acl-long.1"
        assert session_upsert.json()["source_thread_id"] == "search-1"

        detail = client.get(f"/api/projects/{project['project_id']}")
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["project"]["search_thread_count"] == 1
        assert payload["project"]["paper_session_count"] == 1
        assert len(payload["threads"]) == 1
        assert len(payload["paper_sessions"]) == 1

        cleared = client.post(f"/api/projects/{project['project_id']}/clear")
        assert cleared.status_code == 200
        assert cleared.json()["status"] == "cleared"

        after_clear = client.get(f"/api/projects/{project['project_id']}")
        assert after_clear.status_code == 200
        assert after_clear.json()["project"]["search_thread_count"] == 0
        assert after_clear.json()["project"]["paper_session_count"] == 0
        assert after_clear.json()["threads"] == []
        assert after_clear.json()["paper_sessions"] == []

        renamed = client.patch(
            f"/api/projects/{project['project_id']}",
            json={"title": "GAIA Workspace"},
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "GAIA Workspace"

        detail_after_rename = client.get(f"/api/projects/{project['project_id']}")
        assert detail_after_rename.status_code == 200
        assert detail_after_rename.json()["project"]["title"] == "GAIA Workspace"


def test_projects_api_allows_deleting_last_workspace_without_recreating_scratch(tmp_path: Path, monkeypatch) -> None:
    with _create_test_client(tmp_path, monkeypatch) as client:
        created = client.post("/api/projects", json={"title": "Scratch"})
        assert created.status_code == 200
        scratch_delete = client.delete("/api/projects/scratch")
        assert scratch_delete.status_code == 200
        assert scratch_delete.json()["status"] == "deleted"

        listed = client.get("/api/projects")
        assert listed.status_code == 200
        projects = listed.json()["projects"]
        assert projects == []


def test_projects_api_allows_renaming_scratch(tmp_path: Path, monkeypatch) -> None:
    with _create_test_client(tmp_path, monkeypatch) as client:
        created = client.post("/api/projects", json={"title": "Scratch"})
        assert created.status_code == 200
        renamed = client.patch("/api/projects/scratch", json={"title": "Default Workspace"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Default Workspace"


def test_projects_api_recovers_from_empty_paper_session_file(tmp_path: Path, monkeypatch) -> None:
    with _create_test_client(tmp_path, monkeypatch) as client:
        created = client.post("/api/projects", json={"title": "PDF Debug"})
        assert created.status_code == 200
        project = created.json()

        session_path = (
            tmp_path
            / "data"
            / "projects"
            / project["project_id"]
            / "paper_sessions"
            / "2024.acl-long.1.json"
        )
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text("", encoding="utf-8")

        session_upsert = client.put(
            f"/api/projects/{project['project_id']}/papers/2024.acl-long.1/session",
            json={
                "paper_title": "Recovered Session",
                "source_thread_id": None,
                "chat_history": [],
                "last_active_evidence_id": None,
            },
        )
        assert session_upsert.status_code == 200
        assert session_upsert.json()["paper_id"] == "2024.acl-long.1"
