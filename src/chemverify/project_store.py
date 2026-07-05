from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .config import Settings
from .corpus_catalog import default_selected_corpora
from .models import ProjectPaperSessionRecord, ProjectRecord, ProjectSearchThreadRecord
from .utils import now_iso

ModelT = TypeVar("ModelT", bound=BaseModel)

_PROJECT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_SLUG_SANITIZE_PATTERN = re.compile(r"[^a-z0-9]+")


class ProjectStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root_dir = settings.data_dir / "projects"
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[ProjectRecord]:
        projects = [self._load_project(path) for path in self.root_dir.glob("*/project.json")]
        projects = [project for project in projects if project is not None]
        return sorted(projects, key=lambda item: (item.updated_at, item.created_at, item.project_id), reverse=True)

    def ensure_default_project(self) -> ProjectRecord:
        existing = self.get_project("scratch")
        if existing is not None:
            return existing
        return self._create_project(project_id="scratch", title="Scratch")

    def create_project(self, title: str) -> ProjectRecord:
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("Project title cannot be empty.")
        return self._create_project(
            project_id=self._generate_project_id(self._slugify(normalized_title)),
            title=normalized_title,
        )

    def update_project(
        self,
        project_id: str,
        *,
        title: str | None = None,
        selected_corpora: list[str] | None | object = None,
    ) -> ProjectRecord:
        project = self.require_project(project_id)
        updates: dict[str, object] = {"updated_at": now_iso()}
        if title is not None:
            normalized_title = title.strip()
            if not normalized_title:
                raise ValueError("Project title cannot be empty.")
            updates["title"] = normalized_title
        if selected_corpora is not None:
            updates["selected_corpora"] = [item.strip() for item in selected_corpora if item.strip()]
        updated = project.model_copy(update=updates)
        self._write_model(self._project_dir(project_id) / "project.json", updated)
        return updated

    def get_project(self, project_id: str) -> ProjectRecord | None:
        return self._load_project(self._project_dir(project_id) / "project.json")

    def require_project(self, project_id: str) -> ProjectRecord:
        project = self.get_project(project_id)
        if project is None:
            raise FileNotFoundError(project_id)
        return project

    def delete_project(self, project_id: str) -> None:
        project_dir = self._project_dir(project_id)
        if not project_dir.exists():
            raise FileNotFoundError(project_id)
        shutil.rmtree(project_dir)

    def clear_project(self, project_id: str) -> ProjectRecord:
        project = self.require_project(project_id)
        project_dir = self._project_dir(project_id)
        shutil.rmtree(project_dir / "search_threads", ignore_errors=True)
        shutil.rmtree(project_dir / "paper_sessions", ignore_errors=True)
        refreshed = project.model_copy(update={"updated_at": now_iso()})
        self._write_model(project_dir / "project.json", refreshed)
        return refreshed

    def list_threads(self, project_id: str) -> list[ProjectSearchThreadRecord]:
        self.require_project(project_id)
        thread_dir = self._project_dir(project_id) / "search_threads"
        threads = [self._read_model(path, ProjectSearchThreadRecord) for path in thread_dir.glob("*.json")]
        threads = [thread for thread in threads if thread is not None]
        return sorted(threads, key=lambda item: (item.updated_at, item.created_at, item.thread_id), reverse=True)

    def get_thread(self, project_id: str, thread_id: str) -> ProjectSearchThreadRecord | None:
        self.require_project(project_id)
        return self._read_model(
            self._project_dir(project_id) / "search_threads" / f"{thread_id}.json",
            ProjectSearchThreadRecord,
        )

    def save_thread(self, record: ProjectSearchThreadRecord) -> ProjectSearchThreadRecord:
        self.require_project(record.project_id)
        existing = self.get_thread(record.project_id, record.thread_id)
        created_at = existing.created_at if existing is not None else record.created_at
        updated = record.model_copy(update={"created_at": created_at, "updated_at": now_iso()})
        self._write_model(
            self._project_dir(record.project_id) / "search_threads" / f"{record.thread_id}.json",
            updated,
        )
        self._touch_project(record.project_id, updated.updated_at)
        return updated

    def list_paper_sessions(self, project_id: str) -> list[ProjectPaperSessionRecord]:
        self.require_project(project_id)
        session_dir = self._project_dir(project_id) / "paper_sessions"
        sessions = [self._read_model(path, ProjectPaperSessionRecord) for path in session_dir.glob("*.json")]
        sessions = [session for session in sessions if session is not None]
        return sorted(sessions, key=lambda item: (item.updated_at, item.created_at, item.paper_id), reverse=True)

    def get_paper_session(self, project_id: str, paper_id: str) -> ProjectPaperSessionRecord | None:
        self.require_project(project_id)
        return self._read_model(
            self._project_dir(project_id) / "paper_sessions" / f"{paper_id}.json",
            ProjectPaperSessionRecord,
        )

    def save_paper_session(self, record: ProjectPaperSessionRecord) -> ProjectPaperSessionRecord:
        self.require_project(record.project_id)
        existing = self.get_paper_session(record.project_id, record.paper_id)
        created_at = existing.created_at if existing is not None else record.created_at
        updated = record.model_copy(update={"created_at": created_at, "updated_at": now_iso()})
        self._write_model(
            self._project_dir(record.project_id) / "paper_sessions" / f"{record.paper_id}.json",
            updated,
        )
        self._touch_project(record.project_id, updated.updated_at)
        return updated

    def thread_count(self, project_id: str) -> int:
        return len(self.list_threads(project_id))

    def paper_session_count(self, project_id: str) -> int:
        return len(self.list_paper_sessions(project_id))

    def _create_project(self, *, project_id: str, title: str) -> ProjectRecord:
        project_id = self._validate_project_id(project_id)
        timestamp = now_iso()
        project = ProjectRecord(
            project_id=project_id,
            title=title,
            created_at=timestamp,
            updated_at=timestamp,
            selected_corpora=default_selected_corpora(self.settings),
        )
        project_dir = self._project_dir(project_id)
        (project_dir / "search_threads").mkdir(parents=True, exist_ok=True)
        (project_dir / "paper_sessions").mkdir(parents=True, exist_ok=True)
        self._write_model(project_dir / "project.json", project)
        return project

    def _load_project(self, path: Path) -> ProjectRecord | None:
        project = self._read_model(path, ProjectRecord)
        if project is None:
            return None
        if project.selected_corpora is None:
            project = project.model_copy(update={"selected_corpora": default_selected_corpora(self.settings)})
            self._write_model(path, project)
        return project

    def _touch_project(self, project_id: str, updated_at: str) -> None:
        project = self.require_project(project_id)
        self._write_model(
            self._project_dir(project_id) / "project.json",
            project.model_copy(update={"updated_at": updated_at}),
        )

    def _generate_project_id(self, base_project_id: str) -> str:
        normalized_base = self._validate_project_id(base_project_id)
        if not (self._project_dir(normalized_base) / "project.json").exists():
            return normalized_base
        suffix = 2
        while True:
            candidate = self._validate_project_id(f"{normalized_base}-{suffix}")
            if not (self._project_dir(candidate) / "project.json").exists():
                return candidate
            suffix += 1

    def _project_dir(self, project_id: str) -> Path:
        return self.root_dir / self._validate_project_id(project_id)

    @staticmethod
    def _slugify(value: str) -> str:
        lowered = value.strip().lower()
        normalized = _SLUG_SANITIZE_PATTERN.sub("-", lowered).strip("-")
        return normalized or "project"

    @staticmethod
    def _validate_project_id(project_id: str) -> str:
        normalized = project_id.strip().lower()
        if not _PROJECT_ID_PATTERN.fullmatch(normalized):
            raise ValueError(f"Invalid project_id={project_id!r}")
        return normalized

    @staticmethod
    def _write_model(path: Path, record: BaseModel) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump_json(indent=2)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            tmp_path = Path(handle.name)
        tmp_path.replace(path)

    @staticmethod
    def _read_model(path: Path, model: type[ModelT]) -> ModelT | None:
        if not path.exists():
            return None
        raw_text = path.read_text(encoding="utf-8")
        if not raw_text.strip():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return None
        try:
            return model.model_validate_json(raw_text)
        except Exception:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return None
