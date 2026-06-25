from __future__ import annotations

import json
import logging
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import typer

from .acl_anthology import ACLAnthologyIngestor
from .config import CorpusSpec, Settings
from .indexer import IndexBuilder
from .models import PaperRecord
from .runtime import resolve_project_root
from .search_current import rebuild_search_current
from .storage import LocalStore
from .terminal_logging import configure_terminal_logging
from .utils import now_iso

configure_terminal_logging()

app = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)
PROJECT_ROOT = resolve_project_root()


def _project_root() -> Path:
    return Path(PROJECT_ROOT).expanduser().resolve()


def _components() -> tuple[Settings, LocalStore]:
    settings = Settings.from_env(root_dir=_project_root())
    store = LocalStore(settings)
    return settings, store


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
            "OPENAI_API_KEY=\nOPENAI_BASE_URL=https://api.openai.com/v1\nOPENAI_MODEL=\n"
            "PAPER_SEARCH_AGENT_DATA_DIR=./data\n",
            encoding="utf-8",
        )
        typer.echo(f"Wrote {env_path}")
    settings = Settings.from_env(root_dir=root, corpus=_personal_corpus())
    settings.ensure_dirs()
    typer.echo(f"Initialized PaperSearchAgent at {root}")


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
def index_personal_library(
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
    settings = Settings.from_env(root_dir=_project_root(), corpus=_personal_corpus(year))
    store = LocalStore(settings)
    source_papers = store.load_source_papers()
    if not source_papers:
        raise RuntimeError("No PDFs are registered. Run `paper-search-agent add-pdfs ./your-pdfs` first.")

    if run_parse:
        from .mineru_pipeline import run_mineru_pipeline

        run_mineru_pipeline(settings=settings, papers=source_papers)

    builder = IndexBuilder(settings, store)
    paper_ids = builder.load_paper_ids(paper_id_file) if paper_id_file is not None else None
    summary = builder.build(max_papers=max_papers, paper_ids=paper_ids)
    if summary.indexed_papers <= 0:
        raise RuntimeError("Index build finished with 0 indexed papers. Check MinerU/PDF parse failures before publishing.")

    _write_completed_index_state(settings, summary)
    manifest = rebuild_search_current(_project_root(), corpora=[settings.corpus])
    _write_search_current_scope([settings.corpus])
    typer.echo(json.dumps({"build": summary.model_dump(), "search_current": manifest}, ensure_ascii=False, indent=2))


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
        typer.echo("No online index found. Run `paper-search-agent index` before starting the web app.", err=True)
        raise typer.Exit(code=1)
    if shutil.which("npm") is None:
        typer.echo("npm is required to start the web app.", err=True)
        raise typer.Exit(code=1)
    if install_deps and not web_dir.joinpath("node_modules").exists():
        subprocess.run(["npm", "--prefix", str(web_dir), "install"], cwd=root, check=True)

    env = os.environ.copy()
    env["PAPER_SEARCH_AGENT_ROOT"] = str(root)
    env["PAPER_SEARCH_AGENT_API_BASE_URL"] = f"http://127.0.0.1:{api_port}/api"
    api_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "paper_search_agent.api.app:create_app",
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
        ["npm", "--prefix", str(web_dir), "run", "dev", "--", "--hostname", host, "--port", str(web_port)],
        cwd=root,
        env=env,
    )
    typer.echo(f"PaperSearchAgent web: http://{host}:{web_port}")
    raise typer.Exit(code=_run_until_exit([api_process, web_process]))


@app.command("demo-acl")
def demo_acl(
    max_papers: int = typer.Option(100, "--max-papers", min=1, help="Number of ACL papers to download."),
    year: int = typer.Option(2025, "--year", help="ACL event year."),
    track: str = typer.Option("long", "--track", help="ACL track to sample, for example long or short."),
) -> None:
    corpus_track = _corpus_track([track])
    settings, store = _components_for_corpus(venue="acl", year=year, track=corpus_track)
    summary = ACLAnthologyIngestor(settings, store).ingest_event(
        venue="acl",
        year=year,
        tracks=[track],
        max_papers=max_papers,
        download_pdfs=True,
    )
    _write_search_current_scope([settings.corpus])
    typer.echo(summary.model_dump_json(indent=2))
    typer.echo(f"Downloaded PDFs are under {settings.pdf_dir / 'acl' / str(year) / corpus_track}")
    typer.echo("Next: run `paper-search-agent index`, then `paper-search-agent web`.")


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
