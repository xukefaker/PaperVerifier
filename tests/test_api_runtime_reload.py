from __future__ import annotations

import json
import importlib
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from chemverify.config import CorpusSpec, Settings
from chemverify.models import QueryPlan, SearchResponse, SearchTrace
from chemverify.storage import LocalStore


class _NoopSearchEngine:
    def __init__(self, settings: Settings, store) -> None:
        self.settings = settings
        self.store = store


class _SnapshotSearchEngine(_NoopSearchEngine):
    def __init__(self, settings: Settings, store) -> None:
        super().__init__(settings, store)
        papers = int(store.load_index_state().get("papers", 0))
        self.runtime = SimpleNamespace(paper_ids=[f"paper-{idx}" for idx in range(papers)])


class _JobSearchEngine(_SnapshotSearchEngine):
    def search(self, query: str, top_k: int, progress_callback=None) -> SearchResponse:
        if progress_callback is not None:
            progress_callback("done", "completed")
        return SearchResponse(trace_id=f"trace-{query}", mode="test", satisfied=[], partial=[], rejected=[])


class _NoopDeepChatService:
    def __init__(self, settings: Settings, store, engine, **_: object) -> None:
        self.settings = settings
        self.store = store
        self.engine = engine

    def close(self) -> None:
        return None


class _TrackCloseDeepChatService(_NoopDeepChatService):
    def __init__(self, settings: Settings, store, engine, **kwargs: object) -> None:
        super().__init__(settings, store, engine, **kwargs)
        self.closed = False

    def close(self) -> None:
        self.closed = True


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
                "corpora": [],
                "counts": {"papers": papers, "chunks": papers * 10},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_trace(settings: Settings, *, trace_id: str, query: str) -> None:
    store = LocalStore(settings)
    trace = SearchTrace(
        trace_id=trace_id,
        created_at="2026-03-27T00:00:00+00:00",
        mode="test",
        user_query=query,
        query_plan=QueryPlan(user_query=query, global_query=query),
    )
    store.save_trace(trace)


def test_api_reloads_services_after_search_current_switch(tmp_path: Path, monkeypatch) -> None:
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
    app_module = importlib.import_module("chemverify.api.app")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "SearchEngine", _NoopSearchEngine)
    monkeypatch.setattr(app_module, "DeepChatService", _NoopDeepChatService)

    _write_search_current_snapshot(tmp_path, build_id="build-a", papers=1)

    app = app_module.create_app()
    with TestClient(app) as client:
        first = client.get("/api/health")
        assert first.status_code == 200
        assert first.json()["index_state"]["papers"] == 1

        _write_search_current_snapshot(tmp_path, build_id="build-b", papers=2)
        second = client.get("/api/health")
        assert second.status_code == 200
        assert second.json()["index_state"]["papers"] == 2


def test_service_manager_reloads_after_search_current_manifest_switch(tmp_path: Path, monkeypatch) -> None:
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
    app_module = importlib.import_module("chemverify.api.app")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "SearchEngine", _SnapshotSearchEngine)
    monkeypatch.setattr(app_module, "DeepChatService", _NoopDeepChatService)

    _write_search_current_snapshot(tmp_path, build_id="build-a", papers=1)

    manager = app_module.AppServiceManager(tmp_path, completed_job_retention_seconds=0.0)
    services_one = manager.get_services()
    assert len(services_one.engine.runtime.paper_ids) == 1

    _write_search_current_snapshot(tmp_path, build_id="build-b", papers=2)
    services_two = manager.get_services()
    assert services_two is not services_one
    assert len(services_two.engine.runtime.paper_ids) == 2
    assert len(manager._services_by_runtime) == 1
    manager.close()


def test_search_jobs_remain_queryable_after_search_current_switch(tmp_path: Path, monkeypatch) -> None:
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
    app_module = importlib.import_module("chemverify.api.app")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "SearchEngine", _JobSearchEngine)
    monkeypatch.setattr(app_module, "DeepChatService", _NoopDeepChatService)

    _write_search_current_snapshot(tmp_path, build_id="build-a", papers=1)

    app = app_module.create_app()
    with TestClient(app) as client:
        created = client.post("/api/search/jobs", json={"query": "find gaia papers", "top_k": 5})
        assert created.status_code == 200
        job_id = created.json()["job_id"]

        for _ in range(20):
            status = client.get(f"/api/search/jobs/{job_id}")
            assert status.status_code == 200
            if status.json()["status"] in {"running", "completed"}:
                break
            time.sleep(0.01)

        _write_search_current_snapshot(tmp_path, build_id="build-b", papers=1)
        after_switch = client.get(f"/api/search/jobs/{job_id}")
        assert after_switch.status_code == 200


def test_trace_route_finds_old_corpus_trace_after_search_current_refresh(tmp_path: Path, monkeypatch) -> None:
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
    app_module = importlib.import_module("chemverify.api.app")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "SearchEngine", _NoopSearchEngine)
    monkeypatch.setattr(app_module, "DeepChatService", _NoopDeepChatService)

    settings_acl = Settings.from_env(tmp_path, corpus=CorpusSpec.from_values("acl", 2025, "long"))
    _write_trace(settings_acl, trace_id="trace-old", query="find gaia papers")
    _write_search_current_snapshot(tmp_path, build_id="build-a", papers=0)

    app = app_module.create_app()
    with TestClient(app) as client:
        response = client.get("/api/traces/trace-old")
        assert response.status_code == 200
        assert response.json()["trace_id"] == "trace-old"


def test_service_manager_keeps_borrowed_runtime_alive_until_search_current_request_finishes(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    app_module = importlib.import_module("chemverify.api.app")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "SearchEngine", _SnapshotSearchEngine)
    monkeypatch.setattr(app_module, "DeepChatService", _TrackCloseDeepChatService)

    _write_search_current_snapshot(tmp_path, build_id="build-a", papers=1)

    manager = app_module.AppServiceManager(tmp_path, completed_job_retention_seconds=0.0)
    with manager.acquire_services() as services_one:
        assert len(services_one.engine.runtime.paper_ids) == 1
        assert services_one.deep_chat.closed is False

        _write_search_current_snapshot(tmp_path, build_id="build-b", papers=2)
        services_two = manager.get_services()
        assert services_two is not services_one
        assert len(services_two.engine.runtime.paper_ids) == 2
        assert services_one.deep_chat.closed is False

    assert services_one.deep_chat.closed is True
    manager.close()
