from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)


class IndexProgress:
    def __init__(self) -> None:
        self.console = Console(stderr=True, soft_wrap=True)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[status]}"),
            console=self.console,
            transient=False,
        )
        self._tasks: dict[str, TaskID] = {}

    def __enter__(self) -> "IndexProgress":
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.progress.stop()

    def mineru_start(self, *, total: int, batches: int, pages: int, device: str) -> None:
        status = f"{total} PDFs, {pages} pages, {batches} MinerU jobs, device={device}; first run may load models"
        self._ensure("mineru", "Parse PDFs", total=max(total, 1), status=status)

    def mineru_batch(
        self,
        *,
        completed: int,
        total: int,
        batch_index: int,
        batches: int,
        papers: int,
        pages: int,
        current: str,
    ) -> None:
        status = _mineru_status(
            batch_index=batch_index,
            batches=batches,
            papers=papers,
            pages=pages,
            current=current,
            elapsed_seconds=0,
            tick=0,
            cancel_requested=False,
        )
        self._update("mineru", completed=completed, total=max(total, 1), status=status)

    def mineru_heartbeat(
        self,
        *,
        completed: int,
        total: int,
        batch_index: int,
        batches: int,
        papers: int,
        pages: int,
        current: str,
        elapsed_seconds: int,
        tick: int,
        cancel_requested: bool,
    ) -> None:
        self._update(
            "mineru",
            completed=completed,
            total=max(total, 1),
            status=_mineru_status(
                batch_index=batch_index,
                batches=batches,
                papers=papers,
                pages=pages,
                current=current,
                elapsed_seconds=elapsed_seconds,
                tick=tick,
                cancel_requested=cancel_requested,
            ),
        )
        self.progress.refresh()

    def mineru_update(self, *, completed: int, total: int, failed: int) -> None:
        status = f"failed={failed}" if failed else "ok"
        self._ensure("mineru", "Parse PDFs", total=max(total, 1), status=status)
        self._update("mineru", completed=completed, total=max(total, 1), status=status)

    def mineru_skip(self, *, skipped_failed: int) -> None:
        status = f"no pending PDFs, skipped_failed={skipped_failed}"
        self._ensure("mineru", "Parse PDFs", total=1, status=status)
        self._update("mineru", completed=1, status=status)

    def index_parse_start(self, *, total: int) -> None:
        self._ensure("normalize", "Normalize papers", total=max(total, 1), status="reading MinerU artifacts")

    def index_parse_update(
        self,
        *,
        completed: int,
        total: int,
        indexed: int,
        failed: int,
        sections: int,
        chunks: int,
    ) -> None:
        status = f"indexed={indexed} failed={failed} sections={sections} chunks={chunks}"
        self._ensure("normalize", "Normalize papers", total=max(total, 1), status=status)
        self._update("normalize", completed=completed, total=max(total, 1), status=status)

    def prepare(self, status: str) -> None:
        self._ensure("prepare", "Prepare evidence", total=1, status=status)
        self._update("prepare", completed=1, status=status)

    def encode_start(self, *, name: str, total: int, backend: str) -> None:
        self._ensure("encode", "Encode vectors", total=max(total, 1), status=f"{name}, {backend}")
        self._update("encode", completed=0, total=max(total, 1), status=f"{name}, {backend}")

    def encode_update(self, *, name: str, completed: int, total: int) -> None:
        self._update("encode", completed=completed, total=max(total, 1), status=name)

    def encode_done(self, *, name: str, total: int) -> None:
        self._update("encode", completed=max(total, 1), total=max(total, 1), status=f"{name} done")

    def save_start(self) -> None:
        self._ensure("save", "Save index", total=1, status="writing files")
        self._update("save", completed=0, status="writing files")

    def save_done(self) -> None:
        self._ensure("save", "Save index", total=1, status="done")
        self._update("save", completed=1, status="done")

    def publish_start(self) -> None:
        self._ensure("publish", "Publish index", total=1, status="activating current index")
        self._update("publish", completed=0, status="activating current index")

    def publish_done(self) -> None:
        self._ensure("publish", "Publish index", total=1, status="done")
        self._update("publish", completed=1, status="done")

    def _ensure(self, key: str, description: str, *, total: int, status: str) -> TaskID:
        task_id = self._tasks.get(key)
        if task_id is None:
            task_id = self.progress.add_task(description, total=total, status=status)
            self._tasks[key] = task_id
        else:
            self.progress.update(task_id, description=description, total=total, status=status)
        return task_id

    def _update(
        self,
        key: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        status: str | None = None,
    ) -> None:
        task_id = self._tasks[key]
        kwargs: dict[str, object] = {}
        if completed is not None:
            kwargs["completed"] = completed
        if total is not None:
            kwargs["total"] = total
        if status is not None:
            kwargs["status"] = status
        self.progress.update(task_id, **kwargs)


def _format_elapsed(seconds: int) -> str:
    minutes, secs = divmod(max(seconds, 0), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _mineru_status(
    *,
    batch_index: int,
    batches: int,
    papers: int,
    pages: int,
    current: str,
    elapsed_seconds: int,
    tick: int,
    cancel_requested: bool,
) -> str:
    dots = "." * (tick % 4 or 4)
    if cancel_requested:
        return f"q received; finishing current PDF, then cleanup | current={current} | elapsed {_format_elapsed(elapsed_seconds)}"
    if papers == 1:
        return f"current={current} | {pages} pages | MinerU parsing {dots} | elapsed {_format_elapsed(elapsed_seconds)}"
    return (
        f"job {batch_index}/{batches} | {papers} PDFs, {pages} pages | "
        f"current={current} | MinerU parsing {dots} | elapsed {_format_elapsed(elapsed_seconds)}"
    )
