from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from mineru.cli.common import do_parse, read_fn
from mineru.utils.enum_class import MakeMode

from .cancel import CancelRequested
from .config import Settings
from .devices import resolve_mineru_device
from .models import PaperRecord, ParseFailureRecord
from .utils import now_iso

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BatchItem:
    paper: PaperRecord
    pdf_path: Path
    pages: int


class _TailCapture:
    encoding = "utf-8"
    errors = "replace"

    def __init__(self, max_chars: int = 4000) -> None:
        self.max_chars = max_chars
        self._value = ""

    def write(self, value: object) -> int:
        value = str(value)
        self._value = (self._value + value)[-self.max_chars :]
        return len(value)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def getvalue(self) -> str:
        return self._value


@dataclass(slots=True)
class MinerUPipelineConfig:
    max_pdfs_per_batch: int = 1
    max_pages_per_batch: int = 120
    min_batch_inference_size: int = 384
    render_threads: int = 1
    render_timeout: int = 600


def artifact_complete(output_dir: Path, paper_id: str) -> bool:
    root = output_dir / paper_id
    has_content_list = any(root.rglob(f"{paper_id}_content_list.json"))
    has_middle = any(root.rglob(f"{paper_id}_middle.json"))
    has_markdown = any(root.rglob(f"{paper_id}.md"))
    return has_content_list and has_middle and has_markdown


def remove_artifacts(output_dir: Path, paper_ids: list[str]) -> None:
    for paper_id in paper_ids:
        root = output_dir / paper_id
        if root.exists():
            shutil.rmtree(root)


def load_failure_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def save_failure_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")


def remove_failure_entries(path: Path, paper_ids: set[str]) -> None:
    entries = [entry for entry in load_failure_entries(path) if entry.get("paper_id") not in paper_ids]
    if entries:
        save_failure_entries(path, entries)
        return
    if path.exists():
        path.unlink()


def normalize_failure_entries(settings: Settings, papers: list[PaperRecord]) -> list[ParseFailureRecord]:
    paper_lookup = {paper.paper_id: paper for paper in papers}
    records: list[ParseFailureRecord] = []
    for entry in load_failure_entries(settings.mineru_failure_manifest_path):
        paper_id = str(entry.get("paper_id") or "")
        paper = paper_lookup.get(paper_id)
        if paper is None:
            continue
        if artifact_complete(settings.mineru_output_dir, paper_id):
            continue
        records.append(
            ParseFailureRecord(
                paper_id=paper.paper_id,
                venue=paper.venue,
                year=paper.year,
                track=paper.track,
                parser_backend="mineru_pipeline",
                stage=str(entry.get("stage") or "mineru_pipeline"),
                error_type=str(entry.get("error_type") or "mineru_pipeline_error"),
                error_message=str(entry.get("error_message") or "unknown mineru pipeline error"),
                local_pdf_path=paper.local_pdf_path,
                analysis=str(entry.get("analysis") or ""),
                suggestion=str(entry.get("suggestion") or ""),
                details={
                    "pdf_path": entry.get("pdf_path"),
                    "pages": entry.get("pages"),
                },
                occurred_at=str(entry.get("occurred_at") or now_iso()),
            )
        )
    return records


def run_mineru_pipeline(
    *,
    settings: Settings,
    papers: list[PaperRecord],
    controller: Any | None = None,
    cancel_check: Callable[[], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
    progress: Any | None = None,
    config: MinerUPipelineConfig | None = None,
) -> dict[str, int]:
    config = config or MinerUPipelineConfig()
    _configure_mineru_env(settings=settings, config=config)
    mineru_device = os.environ.get("MINERU_DEVICE_MODE", "auto")

    failed_entries = load_failure_entries(settings.mineru_failure_manifest_path)
    failed_ids = {
        str(entry.get("paper_id") or "")
        for entry in failed_entries
        if entry.get("paper_id") and not artifact_complete(settings.mineru_output_dir, str(entry["paper_id"]))
    }

    pending_items = [
        BatchItem(paper=paper, pdf_path=Path(str(paper.local_pdf_path)), pages=_pdf_page_count(Path(str(paper.local_pdf_path))))
        for paper in papers
        if paper.local_pdf_path
        and not artifact_complete(settings.mineru_output_dir, paper.paper_id)
        and paper.paper_id not in failed_ids
    ]
    if not pending_items:
        if progress is not None:
            progress.mineru_skip(skipped_failed=len(failed_ids))
        else:
            logger.info("[bold magenta]MinerU[/] | skipped corpus=%s reason=no_pending_pdfs", settings.corpus.key)
        return {"processed": 0, "failed": 0, "skipped_failed": len(failed_ids)}

    start_time = time.time()
    batches = _build_batches(
        pending_items,
        max_pdfs=config.max_pdfs_per_batch,
        max_pages=config.max_pages_per_batch,
    )
    processed = 0
    failed = 0
    total = len(pending_items)
    if progress is not None:
        progress.mineru_start(
            total=total,
            batches=len(batches),
            pages=sum(item.pages for item in pending_items),
            device=mineru_device,
        )

    if progress is None:
        logger.info(
            "[bold magenta]MinerU[/] | start corpus=%s pending=%s batch_limit=%s/%s",
            settings.corpus.key,
            total,
            config.max_pdfs_per_batch,
            config.max_pages_per_batch,
        )

    for batch_index, batch in enumerate(batches, start=1):
        _check_pause(controller)
        _check_cancel(cancel_check)
        batch_ids = [item.paper.paper_id for item in batch]
        batch_pages = sum(item.pages for item in batch)
        current = _batch_label(batch)
        if progress is not None:
            progress.mineru_batch(
                completed=processed + failed,
                total=total,
                batch_index=batch_index,
                batches=len(batches),
                papers=len(batch),
                pages=batch_pages,
                current=current,
            )
        else:
            logger.info(
                "[bold magenta]MinerU[/] | batch_start batch=%s/%s papers=%s pages=%s ids=%s",
                batch_index,
                len(batches),
                len(batch),
                batch_pages,
                " ".join(batch_ids),
            )
        try:
            with _progress_heartbeat(
                progress,
                completed=processed + failed,
                total=total,
                batch_index=batch_index,
                batches=len(batches),
                papers=len(batch),
                pages=batch_pages,
                current=current,
                cancel_requested=cancel_requested,
            ):
                _run_batch(
                    batch,
                    output_dir=settings.mineru_output_dir,
                    lang=settings.mineru_lang or "en",
                    parse_method="txt" if settings.mineru_method == "auto" else settings.mineru_method,
                    backend=settings.mineru_backend,
                    formula=settings.mineru_formula,
                    table=settings.mineru_table,
                    quiet_output=progress is not None,
                )
            _check_cancel(cancel_check)
            processed += len(batch)
            remove_failure_entries(settings.mineru_failure_manifest_path, set(batch_ids))
            if progress is not None:
                progress.mineru_update(completed=processed + failed, total=total, failed=failed)
            _emit_progress(
                controller=controller,
                phase="parse",
                message=f"MinerU completed batch {batch_index}/{len(batches)}",
                completed=processed + failed,
                total=total,
                unit="papers",
                started_at=start_time,
            )
        except CancelRequested:
            raise
        except Exception as exc:
            if progress is None:
                logger.warning("[bold magenta]MinerU[/] | batch_failed batch=%s/%s error=%r", batch_index, len(batches), exc)
            for item in batch:
                _check_pause(controller)
                _check_cancel(cancel_check)
                current = _batch_label([item])
                if progress is not None:
                    progress.mineru_batch(
                        completed=processed + failed,
                        total=total,
                        batch_index=batch_index,
                        batches=len(batches),
                        papers=1,
                        pages=item.pages,
                        current=current,
                    )
                try:
                    with _progress_heartbeat(
                        progress,
                        completed=processed + failed,
                        total=total,
                        batch_index=batch_index,
                        batches=len(batches),
                        papers=1,
                        pages=item.pages,
                        current=current,
                        cancel_requested=cancel_requested,
                    ):
                        _run_batch(
                            [item],
                            output_dir=settings.mineru_output_dir,
                            lang=settings.mineru_lang or "en",
                            parse_method="txt" if settings.mineru_method == "auto" else settings.mineru_method,
                            backend=settings.mineru_backend,
                            formula=settings.mineru_formula,
                            table=settings.mineru_table,
                            quiet_output=progress is not None,
                        )
                    _check_cancel(cancel_check)
                    processed += 1
                    remove_failure_entries(settings.mineru_failure_manifest_path, {item.paper.paper_id})
                    if progress is not None:
                        progress.mineru_update(completed=processed + failed, total=total, failed=failed)
                except CancelRequested:
                    raise
                except Exception as single_exc:
                    failed += 1
                    _append_failure_entry(
                        settings.mineru_failure_manifest_path,
                        paper=item.paper,
                        pages=item.pages,
                        stage="single_file_retry",
                        error_type=single_exc.__class__.__name__,
                        error_message=repr(single_exc),
                        analysis="MinerU still failed during the single-paper retry, so this parse result cannot enter indexing.",
                        suggestion="Inspect the PDF and MinerU artifacts. Fix the PDF if needed, then rerun this corpus with rebuild.",
                    )
                    logger.error("[bold magenta]MinerU[/] | single_paper_failed paper=%s error=%r", item.paper.paper_id, single_exc)
                    if progress is not None:
                        progress.mineru_update(completed=processed + failed, total=total, failed=failed)
                _emit_progress(
                    controller=controller,
                    phase="parse",
                    message=f"MinerU retrying failed paper {item.paper.paper_id}",
                    completed=processed + failed,
                    total=total,
                    unit="papers",
                    started_at=start_time,
                )

    if progress is None:
        logger.info(
            "[bold magenta]MinerU[/] | done corpus=%s success=%s failed=%s elapsed=%s",
            settings.corpus.key,
            processed,
            failed,
            _format_duration(time.time() - start_time),
        )
    return {"processed": processed, "failed": failed, "skipped_failed": len(failed_ids)}


@contextmanager
def _progress_heartbeat(
    progress: Any | None,
    *,
    completed: int = 0,
    total: int = 1,
    batch_index: int,
    batches: int,
    papers: int,
    pages: int,
    current: str = "current PDF",
    cancel_requested: Callable[[], bool] | None = None,
    interval: float = 1.0,
):
    if progress is None:
        yield
        return
    stop = threading.Event()
    started_at = time.monotonic()

    def beat() -> None:
        tick = 0
        while not stop.wait(interval):
            tick += 1
            progress.mineru_heartbeat(
                completed=completed,
                total=total,
                batch_index=batch_index,
                batches=batches,
                papers=papers,
                pages=pages,
                current=current,
                elapsed_seconds=int(time.monotonic() - started_at),
                tick=tick,
                cancel_requested=bool(cancel_requested and cancel_requested()),
            )

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def _run_batch(
    batch: list[BatchItem],
    *,
    output_dir: Path,
    lang: str,
    parse_method: str,
    backend: str,
    formula: bool,
    table: bool,
    quiet_output: bool = False,
) -> None:
    pdf_file_names = [item.paper.paper_id for item in batch]
    pdf_bytes_list = [read_fn(item.pdf_path) for item in batch]
    output = _TailCapture()
    try:
        if quiet_output:
            with redirect_stdout(output), redirect_stderr(output), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _do_parse(
                    output_dir=output_dir,
                    pdf_file_names=pdf_file_names,
                    pdf_bytes_list=pdf_bytes_list,
                    lang=lang,
                    backend=backend,
                    parse_method=parse_method,
                    formula=formula,
                    table=table,
                )
            return
        _do_parse(
            output_dir=output_dir,
            pdf_file_names=pdf_file_names,
            pdf_bytes_list=pdf_bytes_list,
            lang=lang,
            backend=backend,
            parse_method=parse_method,
            formula=formula,
            table=table,
        )
    except Exception as exc:
        captured = output.getvalue().strip()
        if captured:
            tail = captured[-2000:]
            raise RuntimeError(f"{exc!r}\nMinerU output tail:\n{tail}") from exc
        raise


def _do_parse(
    *,
    output_dir: Path,
    pdf_file_names: list[str],
    pdf_bytes_list: list[bytes],
    lang: str,
    backend: str,
    parse_method: str,
    formula: bool,
    table: bool,
) -> None:
    do_parse(
        output_dir=str(output_dir),
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        p_lang_list=[lang] * len(pdf_file_names),
        backend=backend,
        parse_method=parse_method,
        formula_enable=formula,
        table_enable=table,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
    )


def _configure_mineru_env(*, settings: Settings, config: MinerUPipelineConfig) -> None:
    device = resolve_mineru_device(settings.mineru_device, purpose="MinerU PDF parsing")
    os.environ["MINERU_DEVICE_MODE"] = device
    os.environ["MINERU_MIN_BATCH_INFERENCE_SIZE"] = str(config.min_batch_inference_size)
    os.environ["MINERU_PDF_RENDER_THREADS"] = str(config.render_threads)
    os.environ["MINERU_PDF_RENDER_TIMEOUT"] = str(config.render_timeout)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _build_batches(items: list[BatchItem], *, max_pdfs: int, max_pages: int) -> list[list[BatchItem]]:
    batches: list[list[BatchItem]] = []
    current: list[BatchItem] = []
    current_pages = 0
    for item in items:
        would_overflow_pdf_count = len(current) >= max_pdfs
        would_overflow_pages = current and current_pages + item.pages > max_pages
        if would_overflow_pdf_count or would_overflow_pages:
            batches.append(current)
            current = []
            current_pages = 0
        current.append(item)
        current_pages += item.pages
    if current:
        batches.append(current)
    return batches


def _batch_label(batch: list[BatchItem]) -> str:
    if len(batch) == 1:
        item = batch[0]
        return _short_label(item.pdf_path.name or item.paper.paper_id)
    first = _short_label(batch[0].pdf_path.name or batch[0].paper.paper_id)
    return f"{first} +{len(batch) - 1} more"


def _short_label(value: str, max_chars: int = 56) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[: max_chars - 1]}..."


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        document = pdfium.PdfDocument(str(pdf_path))
        count = len(document)
        document.close()
        return max(1, int(count))
    except Exception:
        return 16


def _append_failure_entry(
    path: Path,
    *,
    paper: PaperRecord,
    pages: int,
    stage: str,
    error_type: str,
    error_message: str,
    analysis: str,
    suggestion: str,
) -> None:
    entries = load_failure_entries(path)
    entries = [entry for entry in entries if entry.get("paper_id") != paper.paper_id]
    entries.append(
        {
            "paper_id": paper.paper_id,
            "pdf_path": paper.local_pdf_path,
            "pages": pages,
            "stage": stage,
            "error_type": error_type,
            "error_message": error_message,
            "analysis": analysis,
            "suggestion": suggestion,
            "occurred_at": now_iso(),
        }
    )
    save_failure_entries(path, entries)


def _emit_progress(
    *,
    controller: Any | None,
    phase: str,
    message: str,
    completed: int,
    total: int,
    unit: str,
    started_at: float,
) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    rate = completed / elapsed
    remaining = max(total - completed, 0)
    eta_seconds = remaining / rate if rate > 0 else None
    logger.info(
        "[bold magenta]MinerU[/] | progress completed=%s total=%s percent=%.2f rate_%s_per_min=%.2f eta=%s",
        completed,
        total,
        (completed / total) * 100 if total else 100.0,
        unit,
        rate * 60,
        _format_duration(eta_seconds),
    )
    if controller is not None:
        controller.update_progress(
            phase=phase,
            message=message,
            completed=completed,
            total=total,
            unit=unit,
            rate_per_min=rate * 60,
            eta_seconds=eta_seconds,
        )


def _check_pause(controller: Any | None) -> None:
    if controller is not None:
        controller.check_pause_requested()


def _check_cancel(cancel_check: Callable[[], None] | None) -> None:
    if cancel_check is not None:
        cancel_check()


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(seconds))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m{secs:02d}s"
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def eta_timestamp(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return (datetime.now() + timedelta(seconds=max(0, seconds))).isoformat(timespec="seconds")
