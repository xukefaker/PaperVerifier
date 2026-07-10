from __future__ import annotations

import json
import logging
import hashlib
import os
import re
import copy
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .acl_anthology import ACLAnthologyIngestor
from .branding import app_name, app_tagline
from .cancel import CancelRequested, ConsoleCancelWatcher
from .config import CorpusSpec, Settings
from .devices import resolve_mineru_device, resolve_torch_device, torch_cuda_report
from .index_progress import IndexProgress
from .indexer import IndexBuilder
from .models import PaperRecord
from .runtime import resolve_project_root
from .search_current import rebuild_search_current
from .storage import LocalStore
from .terminal_logging import configure_terminal_logging
from .utils import extract_keywords, now_iso

configure_terminal_logging()

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)
PROJECT_ROOT = resolve_project_root()

CHEM_DEMO_PAPERS = [
    {
        "paper_id": "chem-demo-01",
        "title": "Sunlight-driven simultaneous CO2 reduction and water oxidation using indium-organic framework heterostructures",
        "year": 2025,
        "url": "https://www.nature.com/articles/s41467-025-57742-5",
        "pdf_url": "https://www.nature.com/articles/s41467-025-57742-5.pdf",
        "abstract": "Open-access chemistry paper on artificial photosynthesis, CO2 reduction, water oxidation, and indium-organic framework heterostructures.",
    },
    {
        "paper_id": "chem-demo-02",
        "title": "Post-synthetic modification of covalent organic frameworks for CO2 electroreduction",
        "year": 2023,
        "url": "https://www.nature.com/articles/s41467-023-39544-9",
        "pdf_url": "https://www.nature.com/articles/s41467-023-39544-9.pdf",
        "abstract": "Open-access chemistry paper on covalent organic frameworks, catalytic sites, and CO2 electroreduction.",
    },
    {
        "paper_id": "chem-demo-03",
        "title": "Linkage-engineered donor-acceptor covalent organic frameworks for optimal photosynthesis of hydrogen peroxide from water and air",
        "year": 2023,
        "url": "https://www.nature.com/articles/s41929-023-01102-3",
        "pdf_url": "https://www.nature.com/articles/s41929-023-01102-3.pdf",
        "abstract": "Open-access chemistry paper on donor-acceptor covalent organic frameworks, charge transfer, mass transport, and photocatalytic hydrogen peroxide production.",
    },
    {
        "paper_id": "chem-demo-04",
        "title": "Linking oxidative and reductive clusters to prepare crystalline porous catalysts for photocatalytic CO2 reduction with H2O",
        "year": 2022,
        "url": "https://www.nature.com/articles/s41467-022-32449-z",
        "pdf_url": "https://www.nature.com/articles/s41467-022-32449-z.pdf",
        "abstract": "Open-access chemistry paper on crystalline porous catalysts, photocatalytic CO2 reduction, and water as the oxidation source.",
    },
    {
        "paper_id": "chem-demo-05",
        "title": "Efficient electron transmission in covalent organic framework nanosheets for highly active electrocatalytic carbon dioxide reduction",
        "year": 2019,
        "url": "https://www.nature.com/articles/s41467-019-14237-4",
        "pdf_url": "https://www.nature.com/articles/s41467-019-14237-4.pdf",
        "abstract": "Open-access chemistry paper on covalent organic framework nanosheets, electron transmission, and electrocatalytic carbon dioxide reduction.",
    },
    {
        "paper_id": "chem-demo-06",
        "title": "Designing covalent organic frameworks with Co-O4 atomic sites for efficient CO2 photoreduction",
        "year": 2023,
        "url": "https://www.nature.com/articles/s41467-023-36779-4",
        "pdf_url": "https://www.nature.com/articles/s41467-023-36779-4.pdf",
        "abstract": "Open-access chemistry paper on cobalt-coordinated covalent organic frameworks and CO2 photoreduction.",
    },
    {
        "paper_id": "chem-demo-07",
        "title": "Photocatalytic CO2 reduction to syngas using metallosalen covalent organic frameworks",
        "year": 2023,
        "url": "https://www.nature.com/articles/s41467-023-42757-7",
        "pdf_url": "https://www.nature.com/articles/s41467-023-42757-7.pdf",
        "abstract": "Open-access chemistry paper on metallosalen covalent organic frameworks and photocatalytic CO2 reduction to syngas.",
    },
    {
        "paper_id": "chem-demo-08",
        "title": "Oxygen-tolerant CO2 electroreduction over covalent organic frameworks via photoswitching control oxygen passivation strategy",
        "year": 2024,
        "url": "https://www.nature.com/articles/s41467-024-45959-9",
        "pdf_url": "https://www.nature.com/articles/s41467-024-45959-9.pdf",
        "abstract": "Open-access chemistry paper on oxygen-tolerant CO2 electroreduction over covalent organic frameworks.",
    },
]


def _project_root() -> Path:
    return Path(PROJECT_ROOT).expanduser().resolve()


def _local_node_bin(root: Path) -> Path | None:
    candidates = [
        root / ".local" / "node" / "current" / "bin",
        root / ".local" / "node" / "current",
    ]
    for candidate in candidates:
        node = shutil.which("node", path=str(candidate))
        npm = shutil.which("npm", path=str(candidate))
        if node and npm:
            return candidate
    return None


def _tool_path(root: Path, name: str) -> str | None:
    node_bin = _local_node_bin(root)
    search_path = os.environ.get("PATH", "")
    if node_bin is not None:
        search_path = f"{node_bin}{os.pathsep}{search_path}"
    return shutil.which(name, path=search_path)


def _env_with_local_node(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    node_bin = _local_node_bin(root)
    if node_bin is not None:
        env["PATH"] = f"{node_bin}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("npm_config_cache", str(root / ".local" / "npm-cache"))
    return env


def _ensure_port_free(host: str, port: int, label: str) -> None:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            typer.echo(f"{label} port {port} is already in use. Choose another port with --{label.lower()}-port.", err=True)
            raise typer.Exit(code=1)


def _components() -> tuple[Settings, LocalStore]:
    settings = Settings.from_env(root_dir=_project_root())
    store = LocalStore(settings)
    return settings, store


def _exit_with_error(message: str) -> None:
    Console(stderr=True).print(Panel(message, title=f"{app_name()} cannot continue", border_style="red"))
    raise typer.Exit(code=1)


def _online_components() -> tuple[Settings, LocalStore]:
    settings = Settings.from_env(root_dir=_project_root())
    store = LocalStore(settings, root_dir=settings.search_current_dir)
    return settings, store


def _corpus_track(track: list[str] | None = None, *, explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit
    normalized = [item.strip().lower() for item in (track or ["long"]) if item.strip()]
    if not normalized:
        return "long"
    if "all" in normalized or len(normalized) > 1:
        return "all"
    return normalized[0]


def _components_for_corpus(*, venue: str, year: int, track: str) -> tuple[Settings, LocalStore]:
    settings = Settings.from_env(root_dir=_project_root(), corpus=CorpusSpec.from_values(venue, year, track))
    store = LocalStore(settings)
    return settings, store


def _parse_corpus_ref(value: str) -> CorpusSpec:
    parts = [part.strip() for part in value.split("/") if part.strip()]
    if len(parts) != 3:
        raise typer.BadParameter(f"Invalid corpus reference {value!r}. Expected format: venue/year/track")
    venue, year_text, track = parts
    try:
        year = int(year_text)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid corpus year in {value!r}: {year_text!r}") from exc
    return CorpusSpec.from_values(venue, year, track)


def _write_search_current_scope(selected_corpora: list[CorpusSpec]) -> None:
    settings = Settings.from_env(root_dir=_project_root())
    if len(selected_corpora) == 1:
        payload = selected_corpora[0].to_dict()
        settings.active_corpus_path.parent.mkdir(parents=True, exist_ok=True)
        settings.active_corpus_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    settings.active_corpus_path.unlink(missing_ok=True)


def _select_source_papers_for_index(
    source_papers: list[PaperRecord],
    *,
    max_papers: int | None,
    paper_ids: list[str] | None,
) -> list[PaperRecord]:
    selected = source_papers
    if paper_ids is not None:
        paper_lookup = {paper.paper_id: paper for paper in source_papers}
        missing_ids: list[str] = []
        selected = []
        for paper_id in paper_ids:
            paper = paper_lookup.get(paper_id)
            if paper is None:
                missing_ids.append(paper_id)
                continue
            selected.append(paper)
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            suffix = " ..." if len(missing_ids) > 5 else ""
            raise RuntimeError(f"{len(missing_ids)} paper ids were not found in data/raw/papers.jsonl: {preview}{suffix}")
    if max_papers is not None:
        selected = selected[:max_papers]
    if not selected:
        raise RuntimeError("No source papers selected for index.")
    return selected


def _personal_corpus(year: int | None = None) -> CorpusSpec:
    return CorpusSpec.from_values("personal", year or datetime.now(timezone.utc).year, "library")


def _discover_pdf_files(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    pdfs: list[Path] = []
    for path in paths:
        if path.is_dir():
            candidates = sorted(path.rglob("*.pdf"))
        elif path.is_file() and path.suffix.lower() == ".pdf":
            candidates = [path]
        else:
            continue
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            pdfs.append(resolved)
    return pdfs


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug[:48] or "paper"


def _personal_paper_id(path: Path) -> str:
    digest_input = f"{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}".encode("utf-8")
    digest = hashlib.sha1(digest_input).hexdigest()[:12]
    return f"personal-{_safe_slug(path.stem)}-{digest}"


def _title_from_pdf_path(path: Path) -> str:
    title = re.sub(r"[_-]+", " ", path.stem)
    return re.sub(r"\s+", " ", title).strip() or path.name


def _copy_or_link_pdf(source: Path, destination: Path, *, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_files:
        shutil.copy2(source, destination)
        return
    destination.symlink_to(source)


def _write_completed_index_state(settings: Settings, summary: object) -> None:
    now = now_iso()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": f"index_{settings.corpus.venue}_{settings.corpus.year}_{settings.corpus.track}_{time.time_ns()}",
        "corpus": settings.corpus.to_dict(),
        "mode": "index",
        "started_at": getattr(summary, "built_at", now),
        "updated_at": now,
        "status": "completed",
        "phase": "completed",
        "message": "Index completed.",
        "build_summary": summary.model_dump() if hasattr(summary, "model_dump") else {},
    }
    settings.state_dir.joinpath("job_state.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_until_exit(processes: list[subprocess.Popen[bytes]]) -> int:
    try:
        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    return int(code)
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 130
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


@contextmanager
def _quiet_index_loggers():
    logger_names = ["chemsearch.indexer", "chemsearch.mineru_pipeline"]
    previous = {name: logging.getLogger(name).level for name in logger_names}
    try:
        for name in logger_names:
            logging.getLogger(name).setLevel(logging.WARNING)
        yield
    finally:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)


def _staged_index_settings(settings: Settings, run_id: str, *, stage_parse: bool) -> Settings:
    run_dir = settings.data_dir / ".runs" / run_id
    staged = copy.copy(settings)
    staged.current_release_path = run_dir / "release" / "current"
    staged.normalized_dir = staged.current_release_path / "normalized"
    staged.deep_chat_normalized_dir = staged.normalized_dir / "deep_chat"
    staged.index_dir = staged.current_release_path / "indexes" / "layout"
    staged.deep_chat_index_dir = staged.current_release_path / "indexes" / "deep_chat"
    if stage_parse:
        staged.mineru_output_dir = run_dir / "parsed" / "mineru"
        staged.mineru_failure_manifest_path = run_dir / "parsed" / "mineru_failures.jsonl"
    return staged


def _cleanup_run_dir(settings: Settings, run_id: str) -> None:
    run_dir = (settings.data_dir / ".runs" / run_id).resolve()
    runs_root = (settings.data_dir / ".runs").resolve()
    if run_dir == runs_root or runs_root not in run_dir.parents:
        raise RuntimeError(f"Refusing to clean unsafe run directory: {run_dir}")
    shutil.rmtree(run_dir, ignore_errors=True)


def _publish_current_release(staged_settings: Settings, final_settings: Settings) -> None:
    source = staged_settings.current_release_path
    if not source.exists():
        raise RuntimeError(f"Staged index release is missing: {source}")
    destination = final_settings.current_release_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.parent / f".previous-current-{time.time_ns()}"
    try:
        if destination.exists():
            destination.rename(backup)
        source.rename(destination)
    except Exception:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        if backup.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def _publish_mineru_artifacts(staged_settings: Settings, final_settings: Settings, papers: list[PaperRecord]) -> None:
    from .mineru_pipeline import load_failure_entries, save_failure_entries

    paper_ids = {paper.paper_id for paper in papers}
    final_settings.mineru_output_dir.mkdir(parents=True, exist_ok=True)
    for paper_id in sorted(paper_ids):
        source = staged_settings.mineru_output_dir / paper_id
        if not source.exists():
            continue
        destination = final_settings.mineru_output_dir / paper_id
        backup = destination.with_name(f".previous-{destination.name}-{time.time_ns()}")
        try:
            if destination.exists():
                destination.rename(backup)
            source.rename(destination)
        except Exception:
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            if backup.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)

    final_entries = [
        entry
        for entry in load_failure_entries(final_settings.mineru_failure_manifest_path)
        if entry.get("paper_id") not in paper_ids
    ]
    final_entries.extend(load_failure_entries(staged_settings.mineru_failure_manifest_path))
    if final_entries:
        save_failure_entries(final_settings.mineru_failure_manifest_path, final_entries)
    else:
        final_settings.mineru_failure_manifest_path.unlink(missing_ok=True)


@app.command("init")
def init_project(force_env: bool = typer.Option(False, "--force-env", help="Overwrite an existing .env file.")) -> None:
    root = _project_root()
    env_path = root / ".env"
    example_path = root / ".env.example"
    if env_path.exists() and not force_env:
        typer.echo(f"Kept existing {env_path}")
    elif example_path.exists():
        shutil.copy2(example_path, env_path)
        typer.echo(f"Wrote {env_path}")
    else:
        env_path.write_text(
            "# Required. Your OpenAI or OpenAI-compatible API key.\n"
            "OPENAI_API_KEY=\n\n"
            "# Keep this value for OpenAI. Change it only if you use a compatible proxy.\n"
            "OPENAI_BASE_URL=https://api.openai.com/v1\n\n"
            "# Default model for question answering.\n"
            "OPENAI_MODEL=gpt-4o-mini\n\n"
            "# Local storage for PDFs, parsed text, and indexes.\n"
            "CHEMSEARCH_DATA_DIR=./data\n\n"
            "# auto prefers CUDA or Apple MPS when PyTorch can use it, otherwise CPU.\n"
            "CHEMSEARCH_DEVICE=auto\n",
            encoding="utf-8",
        )
        typer.echo(f"Wrote {env_path}")
    if os.name != "nt":
        env_path.chmod(0o600)
    settings = Settings.from_env(root_dir=root, corpus=_personal_corpus())
    settings.ensure_dirs()
    typer.echo(f"Initialized {app_name()} at {root}")


@app.command("doctor")
def doctor() -> None:
    root = _project_root()
    settings = Settings.from_env(root_dir=root)
    settings.ensure_dirs()
    report = torch_cuda_report()
    requested_device = settings.mineru_device or "auto"
    mineru_device = resolve_mineru_device(settings.mineru_device, purpose="MinerU PDF parsing")
    dense_device = resolve_torch_device(settings.dense_device, purpose="Dense retrieval")
    reranker_device = resolve_torch_device(settings.reranker_device, purpose="Reranking")
    accelerated = bool(report["cuda_available"] or report.get("mps_available"))
    data_dir_writable = os.access(settings.data_dir, os.W_OK)
    node_path = _tool_path(root, "node")
    rows = [
        ("Project root", str(root), "ok"),
        ("Python", sys.version.split()[0], "ok"),
        (".venv", "found" if root.joinpath(".venv").exists() else "missing", "ok" if root.joinpath(".venv").exists() else "warning"),
        (".env", "found" if root.joinpath(".env").exists() else "missing", "ok" if root.joinpath(".env").exists() else "warning"),
        ("OPENAI_API_KEY", "set" if settings.openai_api_key else "missing", "ok" if settings.openai_api_key else "warning"),
        ("Node.js", node_path or "not found", "ok" if node_path else "warning"),
        ("MinerU command", settings.resolve_mineru_command(), "ok" if shutil.which(settings.resolve_mineru_command()) or Path(settings.resolve_mineru_command()).exists() else "warning"),
        ("Data directory", str(settings.data_dir), "ok" if data_dir_writable else "error"),
        ("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES") or "not set", "ok"),
        ("Requested device", requested_device, "ok"),
        ("MinerU device", mineru_device, "ok" if mineru_device != "cpu" or not accelerated else "warning"),
        ("Dense device", dense_device, "ok" if dense_device != "cpu" or not accelerated else "warning"),
        ("Reranker device", reranker_device, "ok" if reranker_device != "cpu" or not accelerated else "warning"),
        ("Torch", str(report["torch_version"]), "ok" if report["torch_imported"] else "error"),
        ("Torch CUDA", str(report["torch_cuda_version"] or "none"), "ok" if report["torch_cuda_version"] else "warning"),
        ("CUDA available", str(report["cuda_available"]), "ok" if report["cuda_available"] else "warning"),
        ("MPS available", str(report.get("mps_available", False)), "ok" if report.get("mps_available") else "warning"),
        ("GPU", str(report["gpu_name"] or "not visible to PyTorch"), "ok" if report["gpu_name"] else "warning"),
    ]

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Check")
    table.add_column("Value")
    table.add_column("Status")
    for name, value, status in rows:
        color = {"ok": "green", "warning": "yellow", "error": "red"}[status]
        table.add_row(name, value, f"[{color}]{status}[/]")
    console = Console()
    console.print(Panel(table, title=f"{app_name()} Doctor", border_style="cyan"))
    if not accelerated:
        console.print("[yellow]No accelerator detected. ChemSearch can run on CPU, but indexing will be slow.[/]")


@app.command("add-pdfs")
def add_pdfs(
    paths: list[Path] = typer.Argument(..., exists=True, readable=True, resolve_path=True),
    year: int | None = typer.Option(None, "--year", help="Personal library year. Defaults to the current year."),
    copy_files: bool = typer.Option(True, "--copy/--link", help="Copy PDFs into data/ or keep symlinks."),
) -> None:
    settings = Settings.from_env(root_dir=_project_root(), corpus=_personal_corpus(year))
    store = LocalStore(settings)
    pdfs = _discover_pdf_files(paths)
    if not pdfs:
        raise typer.BadParameter("No PDF files were found.")

    existing = {paper.paper_id: paper for paper in store.load_raw_papers()}
    for source in pdfs:
        paper_id = _personal_paper_id(source)
        destination = settings.pdf_dir / "personal" / f"{paper_id}.pdf"
        _copy_or_link_pdf(source, destination, copy_files=copy_files)
        stored_pdf_path = destination.absolute()
        existing[paper_id] = PaperRecord(
            paper_id=paper_id,
            title=_title_from_pdf_path(source),
            authors=[],
            venue=settings.corpus.venue,
            year=settings.corpus.year,
            track=settings.corpus.track,
            url=stored_pdf_path.as_uri(),
            pdf_url=stored_pdf_path.as_uri(),
            local_pdf_path=str(stored_pdf_path),
            source="personal_pdf",
            metadata={"original_path": str(source), "added_at": now_iso()},
        )

    store.save_raw_papers(sorted(existing.values(), key=lambda item: item.paper_id))
    _write_search_current_scope([settings.corpus])
    typer.echo(f"Added {len(pdfs)} PDF(s) to {settings.raw_dir / 'papers.jsonl'}")


@app.command("index")
def index_library(
    year: int | None = typer.Option(None, "--year", help="Personal library year. Defaults to the current year."),
    max_papers: int | None = typer.Option(None, "--max-papers", min=1),
    paper_id_file: Path | None = typer.Option(
        None,
        "--paper-id-file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    run_parse: bool = typer.Option(True, "--parse/--skip-parse", help="Run MinerU before building indexes."),
) -> None:
    settings = (
        Settings.from_env(root_dir=_project_root(), corpus=_personal_corpus(year))
        if year is not None
        else Settings.from_env(root_dir=_project_root())
    )
    run_id = f"index-{settings.corpus.venue}-{settings.corpus.year}-{settings.corpus.track}-{time.time_ns()}"
    try:
        store = LocalStore(settings)
        source_papers = store.load_source_papers()
        if not source_papers:
            corpus_key = f"{settings.corpus.venue}/{settings.corpus.year}/{settings.corpus.track}"
            raise RuntimeError(
                f"No PDFs are registered for {corpus_key}. Run `chemsearch add-pdfs ./your-pdfs` "
                "or `chemsearch demo-chem` first."
            )

        staged_settings = _staged_index_settings(settings, run_id, stage_parse=run_parse)
        staged_store = LocalStore(staged_settings)
        Console(stderr=True).print(
            "[bold cyan]Index[/] Press [bold]q[/] to cancel. "
            f"If MinerU is parsing a PDF, {app_name()} stops after that PDF and cleans this run."
        )
        with ConsoleCancelWatcher() as cancel, IndexProgress() as progress, _quiet_index_loggers():
            builder = IndexBuilder(staged_settings, staged_store, cancel_check=cancel.check, progress=progress)
            paper_ids = builder.load_paper_ids(paper_id_file) if paper_id_file is not None else None
            selected_papers = _select_source_papers_for_index(
                source_papers,
                max_papers=max_papers,
                paper_ids=paper_ids,
            )
            if run_parse:
                from .mineru_pipeline import run_mineru_pipeline

                run_mineru_pipeline(
                    settings=staged_settings,
                    papers=selected_papers,
                    cancel_check=cancel.check,
                    cancel_requested=lambda: cancel.requested,
                    progress=progress,
                )

            summary = builder.build(max_papers=max_papers, paper_ids=paper_ids)
            cancel.check()

            if summary.indexed_papers <= 0:
                raise RuntimeError("Index build finished with 0 indexed papers. Check MinerU/PDF parse failures before publishing.")

            progress.publish_start()
            if run_parse:
                _publish_mineru_artifacts(staged_settings, settings, selected_papers)
            _publish_current_release(staged_settings, settings)
            _write_completed_index_state(settings, summary)
            manifest = rebuild_search_current(_project_root(), corpora=[settings.corpus])
            _write_search_current_scope([settings.corpus])
            progress.publish_done()
        typer.echo(json.dumps({"build": summary.model_dump(), "search_current": manifest}, ensure_ascii=False, indent=2))
    except CancelRequested:
        _cleanup_run_dir(settings, run_id)
        _exit_with_error("Index canceled. Staged files from this run were removed.")
    except RuntimeError as exc:
        _cleanup_run_dir(settings, run_id)
        _exit_with_error(str(exc))
    except KeyboardInterrupt:
        _cleanup_run_dir(settings, run_id)
        _exit_with_error("Index interrupted. Staged files from this run were removed.")
    else:
        _cleanup_run_dir(settings, run_id)


@app.command("web")
def web(
    host: str = typer.Option("127.0.0.1", "--host"),
    web_port: int = typer.Option(4000, "--web-port"),
    api_port: int = typer.Option(4001, "--api-port"),
    install_deps: bool = typer.Option(True, "--install-deps/--no-install-deps"),
) -> None:
    root = _project_root()
    settings = Settings.from_env(root_dir=root)
    web_dir = root / "apps" / "web"
    if not web_dir.joinpath("package.json").exists():
        typer.echo(f"Web app not found: {web_dir}", err=True)
        raise typer.Exit(code=1)
    if not settings.search_current_manifest_path.exists():
        typer.echo(
            "No online index found. Run `./chemsearch index` first (Windows: `.\\chemsearch.cmd index`).",
            err=True,
        )
        raise typer.Exit(code=1)
    _ensure_port_free(host, web_port, "web")
    _ensure_port_free(host, api_port, "api")
    npm_path = _tool_path(root, "npm")
    if npm_path is None:
        typer.echo("npm is required to start the web app. Run `./scripts/install.sh` first.", err=True)
        raise typer.Exit(code=1)
    env = _env_with_local_node(root)
    if install_deps and not web_dir.joinpath("node_modules").exists():
        subprocess.run([npm_path, "--prefix", str(web_dir), "install"], cwd=root, env=env, check=True)

    env["CHEMSEARCH_ROOT"] = str(root)
    env["CHEMSEARCH_API_BASE_URL"] = f"http://127.0.0.1:{api_port}/api"
    env.setdefault("NEXT_PUBLIC_CHEMSEARCH_APP_NAME", app_name())
    env.setdefault("NEXT_PUBLIC_CHEMSEARCH_APP_TAGLINE", app_tagline())
    api_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "chemsearch.api.app:create_app",
            "--factory",
            "--host",
            host,
            "--port",
            str(api_port),
        ],
        cwd=root,
        env=env,
    )
    web_process = subprocess.Popen(
        [npm_path, "--prefix", str(web_dir), "run", "dev", "--", "--hostname", host, "--port", str(web_port)],
        cwd=root,
        env=env,
    )
    typer.echo(f"{app_name()} web: http://{host}:{web_port}")
    raise typer.Exit(code=_run_until_exit([api_process, web_process]))


def _download_chem_demo_pdf(settings: Settings, paper: dict[str, object]) -> tuple[Path, bool]:
    paper_id = str(paper["paper_id"])
    pdf_url = str(paper["pdf_url"])
    destination = settings.pdf_dir / "chemistry" / str(settings.corpus.year) / settings.corpus.track / f"{paper_id}.pdf"
    if destination.exists():
        return destination, False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(timeout=settings.request_timeout, follow_redirects=True) as client:
            response = client.get(pdf_url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower():
                raise RuntimeError(f"download did not return a PDF. content_type={content_type}")
            destination.write_bytes(response.content)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination, True


@app.command("demo-chem")
def demo_chem(
    max_papers: int = typer.Option(5, "--max-papers", min=1, help="Number of demo chemistry papers to download."),
) -> None:
    settings, store = _components_for_corpus(venue="chemistry", year=2026, track="demo")
    selected = CHEM_DEMO_PAPERS[: min(max_papers, len(CHEM_DEMO_PAPERS))]
    if max_papers > len(CHEM_DEMO_PAPERS):
        Console(stderr=True).print(
            f"[yellow]Only {len(CHEM_DEMO_PAPERS)} demo papers are bundled; downloading all of them.[/]"
        )

    existing = {paper.paper_id: paper for paper in store.load_raw_papers()}
    downloaded = 0
    cached = 0
    console = Console()
    with console.status("[bold cyan]Downloading chemistry demo PDFs...[/]"):
        for item in selected:
            try:
                local_pdf, changed = _download_chem_demo_pdf(settings, item)
            except Exception as exc:
                _exit_with_error(f"Could not download demo PDF {item['paper_id']}: {exc}")
            downloaded += int(changed)
            cached += int(not changed)
            text_for_keywords = f"{item['title']} {item['abstract']}"
            existing[str(item["paper_id"])] = PaperRecord(
                paper_id=str(item["paper_id"]),
                title=str(item["title"]),
                authors=[],
                venue="chemistry",
                year=int(item["year"]),
                track="demo",
                url=str(item["url"]),
                pdf_url=str(item["pdf_url"]),
                local_pdf_path=str(local_pdf.resolve()),
                source="chemsearch_demo",
                abstract=str(item["abstract"]),
                keywords=extract_keywords(text_for_keywords, limit=12),
                metadata={"demo_url": item["url"], "downloaded_at": now_iso()},
            )

    store.save_raw_papers(sorted(existing.values(), key=lambda paper: paper.paper_id))
    _write_search_current_scope([settings.corpus])
    typer.echo(
        json.dumps(
            {
                "corpus": settings.corpus.to_dict(),
                "selected_papers": len(selected),
                "downloaded_pdfs": downloaded,
                "cached_pdfs": cached,
                "pdf_dir": str(settings.pdf_dir / "chemistry" / str(settings.corpus.year) / settings.corpus.track),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    typer.echo("Next: run `./chemsearch index`, then `./chemsearch web` (Windows: `.\\chemsearch.cmd ...`).")


@app.command("demo-acl", hidden=True)
def demo_acl(
    max_papers: int = typer.Option(100, "--max-papers", min=1, help="Number of ACL papers to download."),
    year: int = typer.Option(2025, "--year", help="ACL event year."),
    track: str = typer.Option("long", "--track", help="ACL track to sample, for example long or short."),
) -> None:
    corpus_track = _corpus_track([track])
    settings, store = _components_for_corpus(venue="acl", year=year, track=corpus_track)
    try:
        summary = ACLAnthologyIngestor(settings, store).ingest_event(
            venue="acl",
            year=year,
            tracks=[track],
            max_papers=max_papers,
            download_pdfs=True,
        )
    except RuntimeError as exc:
        _exit_with_error(str(exc))
    _write_search_current_scope([settings.corpus])
    typer.echo(summary.model_dump_json(indent=2))
    typer.echo(f"Downloaded PDFs are under {settings.pdf_dir / 'acl' / str(year) / corpus_track}")
    typer.echo("Next: run `./chemsearch index`, then `./chemsearch web` (Windows: `.\\chemsearch.cmd ...`).")


@app.command("ingest-acl", hidden=True)
def ingest_acl(
    venue: str = typer.Option(..., "--venue"),
    year: int = typer.Option(..., "--year"),
    track: list[str] = typer.Option(["long"], "--track"),
    max_papers: int | None = typer.Option(None, "--max-papers"),
    download_pdfs: bool = typer.Option(True, "--download-pdfs/--no-download-pdfs"),
) -> None:
    settings, store = _components_for_corpus(venue=venue, year=year, track=_corpus_track(track))
    summary = ACLAnthologyIngestor(settings, store).ingest_event(
        venue=venue,
        year=year,
        tracks=track,
        max_papers=max_papers,
        download_pdfs=download_pdfs,
    )
    typer.echo(summary.model_dump_json(indent=2))


@app.command("build-index", hidden=True)
def build_index(
    max_papers: int | None = typer.Option(None, "--max-papers", min=1),
    paper_id_file: Path | None = typer.Option(
        None,
        "--paper-id-file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    settings, store = _components()
    builder = IndexBuilder(settings, store)
    paper_ids = builder.load_paper_ids(paper_id_file) if paper_id_file is not None else None
    summary = builder.build(max_papers=max_papers, paper_ids=paper_ids)
    typer.echo(summary.model_dump_json(indent=2))


@app.command("search")
def search(
    query: str = typer.Option(..., "--query"),
    top_k: int = typer.Option(10, "--top-k"),
) -> None:
    from .search import SearchEngine

    settings, store = _online_components()
    response = SearchEngine(settings, store).search(query, top_k=top_k)
    typer.echo(response.model_dump_json(indent=2))


@app.command("show-paper", hidden=True)
def show_paper(paper_id: str = typer.Argument(...)) -> None:
    _, store = _online_components()
    paper = store.get_paper(paper_id)
    if paper is None:
        raise typer.Exit(code=1)
    typer.echo(paper.model_dump_json(indent=2))


@app.command("inspect-trace", hidden=True)
def inspect_trace(trace_id: str = typer.Argument(...)) -> None:
    _, store = _online_components()
    trace = store.load_trace(trace_id)
    if trace is None:
        raise typer.Exit(code=1)
    typer.echo(trace.model_dump_json(indent=2))


@app.command("rebuild-search-current", hidden=True)
def rebuild_search_current_command(
    corpus: list[str] = typer.Option(
        [],
        "--corpus",
        help="Publish only the specified corpus set. Format: venue/year/track. Repeat --corpus to publish multiple corpora.",
    ),
) -> None:
    selected_corpora = [_parse_corpus_ref(item) for item in corpus]
    manifest = rebuild_search_current(_project_root(), corpora=selected_corpora or None)
    if selected_corpora:
        _write_search_current_scope(selected_corpora)
    typer.echo(json.dumps(manifest, ensure_ascii=False, indent=2))


@app.command("offline-run", hidden=True)
def offline_run(
    venue: str = typer.Option("acl", "--venue"),
    year: int = typer.Option(2025, "--year"),
    track: str = typer.Option("long", "--track"),
    mode: str = typer.Option("resume", "--mode"),
) -> None:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"resume", "rebuild"}:
        raise typer.BadParameter("--mode must be either 'resume' or 'rebuild'")
    from .offline import OfflineRunner

    settings = Settings.from_env(root_dir=_project_root(), corpus=CorpusSpec.from_values(venue, year, track))
    result = OfflineRunner(settings).run(mode=normalized_mode)
    typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


@app.command("offline-enrich", hidden=True)
def offline_enrich(
    venue: str = typer.Option("acl", "--venue"),
    year: int = typer.Option(2025, "--year"),
    track: str = typer.Option("long", "--track"),
    mode: str = typer.Option("resume", "--mode"),
) -> None:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"resume", "rebuild"}:
        raise typer.BadParameter("--mode must be either 'resume' or 'rebuild'")
    from .offline import OfflineEnrichmentRunner

    settings = Settings.from_env(root_dir=_project_root(), corpus=CorpusSpec.from_values(venue, year, track))
    result = OfflineEnrichmentRunner(settings).run(mode=normalized_mode)
    typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


@app.command("offline-pause", hidden=True)
def offline_pause() -> None:
    from .offline import request_pause

    settings = Settings.from_env(root_dir=_project_root())
    typer.echo(json.dumps(request_pause(settings), ensure_ascii=False, indent=2))


@app.command("offline-status", hidden=True)
def offline_status() -> None:
    from .offline import render_status

    settings = Settings.from_env(root_dir=_project_root())
    render_status(settings)
