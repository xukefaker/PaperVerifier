from __future__ import annotations

import json
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Iterator

from fastapi import FastAPI

from ..config import CorpusSpec, Settings
from ..deep_chat import DeepChatService
from ..search import SearchEngine
from ..storage import LocalStore
from .jobs import SearchJobManager
from .routes.chat import router as chat_router
from .routes.health import router as health_router
from .routes.papers import router as papers_router
from .routes.projects import router as projects_router
from .routes.search import router as search_router
from .routes.traces import router as traces_router


@dataclass(slots=True)
class AppServices:
    settings: Settings
    store: LocalStore
    engine: SearchEngine
    jobs: SearchJobManager
    deep_chat: DeepChatService


@dataclass(slots=True)
class _RuntimeEntry:
    services: AppServices
    borrow_count: int = 0
    pending_close: bool = False


class AppServiceManager:
    def __init__(self, root_dir: Path, *, completed_job_retention_seconds: float = 1800.0) -> None:
        self.root_dir = root_dir
        self.completed_job_retention_seconds = completed_job_retention_seconds
        self._lock = RLock()
        self._services_by_runtime: dict[tuple[str, str], _RuntimeEntry] = {}

    def _runtime_identity(self, settings: Settings) -> tuple[str, str]:
        manifest_payload = self._load_search_current_manifest(settings)
        build_id = str(manifest_payload.get("build_id") or "missing")
        return ("search_current", build_id)

    def _create_services(self, settings: Settings) -> AppServices:
        store = LocalStore(settings, root_dir=settings.search_current_dir)
        engine = SearchEngine(settings, store)
        jobs = SearchJobManager(engine)
        deep_chat = DeepChatService(settings, store, engine, root_dir=settings.search_current_dir)
        return AppServices(settings=settings, store=store, engine=engine, jobs=jobs, deep_chat=deep_chat)

    def _close_services(self, services: AppServices) -> None:
        services.deep_chat.close()
        services.jobs.close()

    def _active_runtime_key(self) -> tuple[str, str]:
        return self._runtime_identity(Settings.from_env(root_dir=self.root_dir))

    def _prune_services_locked(self, *, current_runtime_key: tuple[str, str]) -> list[AppServices]:
        now = perf_counter()
        removable: list[tuple[str, str]] = []
        to_close: list[AppServices] = []
        for runtime_key, entry in self._services_by_runtime.items():
            if runtime_key == current_runtime_key:
                continue
            if entry.services.jobs.should_retain(
                now=now,
                completed_job_retention_seconds=self.completed_job_retention_seconds,
            ):
                continue
            if entry.borrow_count > 0:
                entry.pending_close = True
                continue
            removable.append(runtime_key)
        for runtime_key in removable:
            entry = self._services_by_runtime.pop(runtime_key, None)
            if entry is not None:
                to_close.append(entry.services)
        return to_close

    def _acquire_runtime_locked(self, runtime_key: tuple[str, str], settings: Settings) -> _RuntimeEntry:
        entry = self._services_by_runtime.get(runtime_key)
        if entry is None:
            entry = _RuntimeEntry(services=self._create_services(settings))
            self._services_by_runtime[runtime_key] = entry
        entry.borrow_count += 1
        return entry

    def _release_runtime(self, runtime_key: tuple[str, str]) -> None:
        to_close: list[AppServices] = []
        with self._lock:
            entry = self._services_by_runtime.get(runtime_key)
            if entry is not None and entry.borrow_count > 0:
                entry.borrow_count -= 1
            current_runtime_key = self._active_runtime_key()
            if (
                entry is not None
                and entry.borrow_count == 0
                and entry.pending_close
                and runtime_key != current_runtime_key
                and not entry.services.jobs.should_retain(
                    completed_job_retention_seconds=self.completed_job_retention_seconds,
                )
            ):
                removed = self._services_by_runtime.pop(runtime_key, None)
                if removed is not None:
                    to_close.append(removed.services)
            to_close.extend(self._prune_services_locked(current_runtime_key=current_runtime_key))
        for services in to_close:
            self._close_services(services)

    def _corpora_root(self) -> Path:
        settings = Settings.from_env(root_dir=self.root_dir)
        return settings.data_dir / "corpora"

    @staticmethod
    def _load_search_current_manifest(settings: Settings) -> dict[str, object]:
        if not settings.search_current_manifest_path.exists():
            return {}
        return json.loads(settings.search_current_manifest_path.read_text(encoding="utf-8"))

    def _store_for_trace_path(self, trace_path: Path) -> LocalStore | None:
        corpora_root = self._corpora_root()
        try:
            relative = trace_path.relative_to(corpora_root)
        except ValueError:
            return None
        parts = relative.parts
        if len(parts) != 5 or parts[3] != "traces":
            return None
        corpus = CorpusSpec.from_values(parts[0], int(parts[1]), parts[2])
        settings = Settings.from_env(root_dir=self.root_dir, corpus=corpus)
        return LocalStore(settings)

    def get_services(self) -> AppServices:
        settings = Settings.from_env(root_dir=self.root_dir)
        runtime_key = self._runtime_identity(settings)
        to_close: list[AppServices] = []
        with self._lock:
            entry = self._services_by_runtime.get(runtime_key)
            if entry is None:
                entry = _RuntimeEntry(services=self._create_services(settings))
                self._services_by_runtime[runtime_key] = entry
            to_close = self._prune_services_locked(current_runtime_key=runtime_key)
            services = entry.services
        for service in to_close:
            self._close_services(service)
        return services

    @contextmanager
    def acquire_services(self) -> Iterator[AppServices]:
        settings = Settings.from_env(root_dir=self.root_dir)
        runtime_key = self._runtime_identity(settings)
        to_close: list[AppServices] = []
        with self._lock:
            entry = self._acquire_runtime_locked(runtime_key, settings)
            to_close = self._prune_services_locked(current_runtime_key=runtime_key)
            services = entry.services
        for service in to_close:
            self._close_services(service)
        try:
            yield services
        finally:
            self._release_runtime(runtime_key)

    @contextmanager
    def acquire_job_services(self, job_id: str) -> Iterator[AppServices]:
        settings = Settings.from_env(root_dir=self.root_dir)
        current_runtime_key = self._runtime_identity(settings)
        to_close: list[AppServices] = []
        with self._lock:
            runtime_key: tuple[str, str] | None = None
            entry: _RuntimeEntry | None = None
            for candidate_key, candidate_entry in self._services_by_runtime.items():
                if candidate_entry.services.jobs.get_status(job_id) is not None:
                    runtime_key = candidate_key
                    entry = candidate_entry
                    break
            if entry is None:
                runtime_key = current_runtime_key
                entry = self._acquire_runtime_locked(current_runtime_key, settings)
            else:
                entry.borrow_count += 1
            to_close = self._prune_services_locked(current_runtime_key=current_runtime_key)
            services = entry.services
        for service in to_close:
            self._close_services(service)
        assert runtime_key is not None
        try:
            yield services
        finally:
            self._release_runtime(runtime_key)

    def find_job_services(self, job_id: str) -> AppServices | None:
        to_close: list[AppServices] = []
        with self._lock:
            current_runtime_key = self._active_runtime_key()
            found_entry: _RuntimeEntry | None = None
            for entry in self._services_by_runtime.values():
                if entry.services.jobs.get_status(job_id) is not None:
                    found_entry = entry
                    break
            to_close = self._prune_services_locked(current_runtime_key=current_runtime_key)
            services = found_entry.services if found_entry is not None else None
        for service in to_close:
            self._close_services(service)
        if services is not None:
            return services
        return None

    def find_trace_store(self, trace_id: str) -> LocalStore | None:
        to_close: list[AppServices] = []
        with self._lock:
            current_runtime_key = self._active_runtime_key()
            for entry in self._services_by_runtime.values():
                if entry.services.store.load_trace(trace_id) is not None:
                    to_close = self._prune_services_locked(current_runtime_key=current_runtime_key)
                    store = entry.services.store
                    break
            else:
                to_close = self._prune_services_locked(current_runtime_key=current_runtime_key)
                store = None
        for service in to_close:
            self._close_services(service)
        if store is not None:
            return store

        settings = Settings.from_env(root_dir=self.root_dir)
        online_store = LocalStore(settings, root_dir=settings.search_current_dir)
        if online_store.load_trace(trace_id) is not None:
            return online_store

        corpora_root = self._corpora_root()
        for trace_path in corpora_root.glob(f"*/*/*/traces/{trace_id}.json"):
            store = self._store_for_trace_path(trace_path)
            if store is None:
                continue
            if store.load_trace(trace_id) is not None:
                return store
        return None

    def close(self) -> None:
        with self._lock:
            entries = list(self._services_by_runtime.values())
            self._services_by_runtime.clear()
        for entry in entries:
            self._close_services(entry.services)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@asynccontextmanager
async def lifespan(app: FastAPI):
    service_manager = AppServiceManager(PROJECT_ROOT)
    service_manager.get_services()
    app.state.service_manager = service_manager
    try:
        yield
    finally:
        service_manager.close()


def create_app() -> FastAPI:
    app = FastAPI(title="PaperSearchAgent", version="0.1.0", lifespan=lifespan)
    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(search_router, prefix="/api", tags=["search"])
    app.include_router(papers_router, prefix="/api", tags=["papers"])
    app.include_router(projects_router, prefix="/api", tags=["projects"])
    app.include_router(chat_router, prefix="/api", tags=["chat"])
    app.include_router(traces_router, prefix="/api", tags=["traces"])
    return app


app = create_app()
