#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/ChemVerify"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
BENCHMARK_ROOT="$PROJECT_ROOT/data/benchmarks/chemqa500_simple/package"
ANNOTATION_FILE="$PROJECT_ROOT/data/benchmarks/chemqa500_simple/ablation/input/chemqa500_search_benchmark_for_runner.jsonl"
RUN_ROOT="$PROJECT_ROOT/data/benchmarks/chemqa500_simple/ablation/runs"
LOG_ROOT="$PROJECT_ROOT/data/benchmarks/chemqa500_simple/ablation/logs"
ABlATION_SCRIPT="$PROJECT_ROOT/scripts/benchmarks/chemqa40/search_ablation_replay.py"
CORPUS_KEY="chemqa500_simple/2026/all"

cd "$PROJECT_ROOT"
mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$(dirname "$ANNOTATION_FILE")"
export CHEMVERIFY_VENUE="chemqa500_simple"
export CHEMVERIFY_YEAR="2026"
export CHEMVERIFY_TRACK="all"
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED="1"

printf "[%s] prepare/build ChemQA500-simple corpus\n" "$(date -Is)"
"$PYTHON_BIN" - <<PY
from __future__ import annotations

import json
from pathlib import Path

from chemverify.config import CorpusSpec, Settings
from chemverify.indexer import IndexBuilder
from chemverify.mineru_pipeline import artifact_complete, normalize_failure_entries, run_mineru_pipeline
from chemverify.models import PaperRecord
from chemverify.search_current import rebuild_search_current
from chemverify.storage import LocalStore
from chemverify.utils import now_iso

project_root = Path("$PROJECT_ROOT")
benchmark_root = Path("$BENCHMARK_ROOT")
annotation_file = Path("$ANNOTATION_FILE")
settings = Settings.from_env(
    root_dir=project_root,
    corpus=CorpusSpec.from_values("chemqa500_simple", 2026, "all"),
)
store = LocalStore(settings)

manifest_path = benchmark_root / "corpus_manifest.jsonl"
queries_path = benchmark_root / "queries.jsonl"
manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
manifest_rows.sort(key=lambda row: int(str(row["paper_id"])))
if len(manifest_rows) != 500:
    raise RuntimeError(f"Expected 500 corpus rows, found {len(manifest_rows)}")

papers: list[PaperRecord] = []
for row in manifest_rows:
    paper_id = str(row["paper_id"])
    pdf_path = (benchmark_root / row["pdf_path"]).resolve()
    if not pdf_path.exists():
        raise RuntimeError(f"Missing PDF for paper {paper_id}: {pdf_path}")
    papers.append(
        PaperRecord(
            paper_id=paper_id,
            anthology_id=None,
            title=str(row["title"]),
            authors=[],
            venue="chemqa500_simple",
            year=2026,
            track="all",
            volume_id=None,
            abstract="",
            doi=None,
            url=f"benchmark://chemqa500_simple/{paper_id}",
            pdf_url=None,
            local_pdf_path=str(pdf_path),
            source="chemqa500_simple",
            metadata={
                "benchmark": "chemqa500_simple",
                "is_target": bool(row.get("is_target")),
                "benchmark_pdf_path": str(row["pdf_path"]),
            },
        )
    )
store.save_raw_papers(papers)

queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line.strip()]
with annotation_file.open("w", encoding="utf-8") as handle:
    for query in queries:
        handle.write(json.dumps({
            "query_id": str(query["query_id"]),
            "original_qa_id": str(query["query_id"]),
            "target_paper_id": str(query["target_paper_id"]),
            "topic": "all",
            "query_text": str(query["query_text"]),
        }, ensure_ascii=False))
        handle.write("\n")

parse_success = [paper for paper in papers if artifact_complete(settings.mineru_output_dir, paper.paper_id)]
parse_failures = normalize_failure_entries(settings, papers)
print(json.dumps({
    "stage": "before_mineru",
    "papers": len(papers),
    "already_parsed": len(parse_success),
    "known_failures": len(parse_failures),
    "annotation_file": str(annotation_file),
}, ensure_ascii=False), flush=True)

if len(parse_success) + len(parse_failures) < len(papers):
    result = run_mineru_pipeline(settings=settings, papers=papers)
    print(json.dumps({"stage": "mineru_result", **result}, ensure_ascii=False), flush=True)

parse_success = [paper for paper in papers if artifact_complete(settings.mineru_output_dir, paper.paper_id)]
parse_failures = normalize_failure_entries(settings, papers)
if not parse_success:
    raise RuntimeError("No ChemQA500 paper parsed successfully; cannot build index.")
print(json.dumps({
    "stage": "before_index",
    "parse_success": len(parse_success),
    "parse_failures": len(parse_failures),
}, ensure_ascii=False), flush=True)

summary = IndexBuilder(settings, store).build()
now = now_iso()
settings.state_dir.mkdir(parents=True, exist_ok=True)
settings.active_corpus_path.parent.mkdir(parents=True, exist_ok=True)
settings.active_corpus_path.write_text(json.dumps(settings.corpus.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
(settings.state_dir / "job_state.json").write_text(json.dumps({
    "status": "completed",
    "phase": "completed",
    "corpus": settings.corpus.key,
    "message": "ChemQA500-simple local benchmark corpus built.",
    "started_at": now,
    "updated_at": now,
    "completed_at": now,
    "build_summary": summary.model_dump(),
}, ensure_ascii=False, indent=2), encoding="utf-8")
manifest = rebuild_search_current(settings.root_dir, corpora=[settings.corpus], allow_uncompleted_selected=True)
print(json.dumps({
    "stage": "search_current_ready",
    "build_summary": summary.model_dump(),
    "search_current_manifest": manifest,
}, ensure_ascii=False, indent=2), flush=True)
PY

run_ablation() {
  local profile="$1"
  local run_name="$2"
  printf "[%s] START %s (%s)\n" "$(date -Is)" "$run_name" "$profile"
  "$PYTHON_BIN" "$ABlATION_SCRIPT" \
    --annotation-file "$ANNOTATION_FILE" \
    --run-dir "$RUN_ROOT/$run_name" \
    --profile "$profile" \
    --mode full \
    --samples-per-topic 1 \
    --top-k 10 \
    --display-k 10 \
    --corpus-key "$CORPUS_KEY"
  printf "[%s] DONE %s\n" "$(date -Is)" "$run_name"
}

run_ablation "full" "20260423_chemqa500_full"
run_ablation "no_planner" "20260423_chemqa500_no_planner"
run_ablation "no_content_refinement" "20260423_chemqa500_no_content_refinement"
run_ablation "no_verifier" "20260423_chemqa500_no_verifier"

printf "[%s] ChemQA500 queue completed\n" "$(date -Is)"
