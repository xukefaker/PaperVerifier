#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/workspace/ChemVerify"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-20260424_chemqa500_remaining_baselines}"
FORCE="${FORCE:-0}"
BASE="data/benchmarks/chemqa500_simple"
INPUTS="$BASE/baseline_inputs"
RUNS="$BASE/baselines/runs"
CACHES="$BASE/baselines/caches"
INDEXES="$BASE/baselines/indexes"
REPORTS="$BASE/baselines/reports"
LOG_ROOT="$BASE/baselines/logs/$RUN_TAG"

PY=".venv/bin/python"

mkdir -p "$LOG_ROOT" "$REPORTS" "$RUNS" "$CACHES" "$INDEXES"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

collect_report() {
  "$PY" - "$RUN_TAG" "$REPORTS" <<'PY'
import json
import sys
from pathlib import Path

run_tag = sys.argv[1]
reports = Path(sys.argv[2])

runs = [
    ("BM25 (full text)", Path("data/benchmarks/chemqa500_simple/baselines/runs/bm25_full_text/20260424_chemqa500_default/summary.json")),
    ("BM25 (title + abstract)", Path("data/benchmarks/chemqa500_simple/baselines/runs/bm25_title_abstract/20260424_chemqa500_default/summary.json")),
    ("SPECTER2 dense retrieval", Path("data/benchmarks/chemqa500_simple/baselines/runs/specter2/20260424_chemqa500_cpu_default/summary.json")),
    ("Hybrid BM25 + SPECTER2 (RRF)", Path("data/benchmarks/chemqa500_simple/baselines/runs/hybrid_rrf/20260424_chemqa500_bm25_specter2_rrf/summary.json")),
    ("CSQE-adapted", Path("data/benchmarks/chemqa500_simple/baselines/runs/csqe/20260424_chemqa500_adapted_default/summary.json")),
    ("SemRank-adapted", Path(f"data/benchmarks/chemqa500_simple/baselines/runs/semrank_adapted/{run_tag}/summary.json")),
    ("ColBERTv2", Path(f"data/benchmarks/chemqa500_simple/baselines/runs/colbertv2/{run_tag}/summary.json")),
    ("SPLADE++", Path(f"data/benchmarks/chemqa500_simple/baselines/runs/splade_pp/{run_tag}/summary.json")),
    ("Hybrid + MonoT5 reranker", Path(f"data/benchmarks/chemqa500_simple/baselines/runs/hybrid_monot5/{run_tag}/summary.json")),
    ("Hybrid + RankT5 reranker", Path(f"data/benchmarks/chemqa500_simple/baselines/runs/hybrid_rankt5/{run_tag}/summary.json")),
    ("Hybrid + GPT-5.4-mini reranker", Path(f"data/benchmarks/chemqa500_simple/baselines/runs/hybrid_gpt54mini/{run_tag}/summary.json")),
]

rows = []
for method, path in runs:
    row = {"method": method, "summary_path": str(path), "status": "pending"}
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        row.update(
            {
                "status": "done",
                "query_count": summary.get("query_count"),
                "hit_at_1": summary.get("hit_at_1"),
                "hit_at_3": summary.get("hit_at_3"),
                "hit_at_5": summary.get("hit_at_5"),
                "hit_at_10": summary.get("hit_at_10"),
                "mrr": summary.get("mrr"),
            }
        )
    rows.append(row)

reports.mkdir(parents=True, exist_ok=True)
(reports / f"{run_tag}_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

lines = [
    f"# ChemQA500 Baseline Queue Summary ({run_tag})",
    "",
    "| Method | Status | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    if row["status"] == "done":
        lines.append(
            "| {method} | done | {hit_at_1:.3f} | {hit_at_3:.3f} | {hit_at_5:.3f} | {hit_at_10:.3f} | {mrr:.3f} |".format(**row)
        )
    else:
        lines.append(f"| {row['method']} | pending | -- | -- | -- | -- | -- |")
lines.append("")
(reports / f"{run_tag}_summary.md").write_text("\n".join(lines), encoding="utf-8")
print(reports / f"{run_tag}_summary.md")
PY
}

run_method() {
  local method="$1"
  local summary_path="$2"
  shift 2
  local log_path="$LOG_ROOT/${method}.log"

  echo "[$(timestamp)] START $method"
  echo "[$(timestamp)] log=$log_path"

  if [[ -f "$summary_path" && "$FORCE" != "1" ]]; then
    echo "[$(timestamp)] SKIP $method summary already exists: $summary_path"
    collect_report
    return 0
  fi

  mkdir -p "$(dirname "$summary_path")"
  {
    echo "[$(timestamp)] command: $*"
    "$@"
  } >"$log_path" 2>&1

  echo "[$(timestamp)] DONE $method"
  collect_report
}

collect_report

run_method \
  "semrank_adapted" \
  "$RUNS/semrank_adapted/$RUN_TAG/summary.json" \
  "$PY" baselines/chemqa40/methods/semrank_adapted/scripts/run_semrank_adapted.py \
    --paper-abstracts "$INPUTS/views/paper_abstracts.jsonl" \
    --queries "$INPUTS/queries/queries.jsonl" \
    --specter2-results "$RUNS/specter2/20260424_chemqa500_cpu_default/raw_results.jsonl" \
    --run-root "$RUNS/semrank_adapted" \
    --run-name "$RUN_TAG"

run_method \
  "colbertv2" \
  "$RUNS/colbertv2/$RUN_TAG/summary.json" \
  "$PY" baselines/chemqa40/methods/colbertv2/scripts/run_colbertv2.py \
    --passages-jsonl "$INPUTS/views/paper_passage_view.jsonl" \
    --queries-jsonl "$INPUTS/queries/queries.jsonl" \
    --run-root "$RUNS/colbertv2" \
    --index-root "$INDEXES/colbertv2/$RUN_TAG" \
    --cache-root "$CACHES/colbertv2/$RUN_TAG" \
    --run-name "$RUN_TAG"

run_method \
  "splade_pp" \
  "$RUNS/splade_pp/$RUN_TAG/summary.json" \
  "$PY" baselines/chemqa40/methods/splade_pp/scripts/run_splade_pp.py \
    --paper-text-view "$INPUTS/views/paper_text_view.jsonl" \
    --queries-jsonl "$INPUTS/queries/queries.jsonl" \
    --run-root "$RUNS/splade_pp" \
    --cache-root "$CACHES/splade_pp/$RUN_TAG" \
    --run-name "$RUN_TAG"

run_method \
  "hybrid_monot5" \
  "$RUNS/hybrid_monot5/$RUN_TAG/summary.json" \
  "$PY" baselines/chemqa40/methods/hybrid_monot5/scripts/run_hybrid_monot5.py \
    --paper-text-view "$INPUTS/views/paper_text_view.jsonl" \
    --hybrid-results "$RUNS/hybrid_rrf/20260424_chemqa500_bm25_specter2_rrf/raw_results.jsonl" \
    --run-root "$RUNS/hybrid_monot5" \
    --cache-root "$CACHES/hybrid_monot5/$RUN_TAG" \
    --run-name "$RUN_TAG" \
    --candidate-topk 20

run_method \
  "hybrid_rankt5" \
  "$RUNS/hybrid_rankt5/$RUN_TAG/summary.json" \
  "$PY" baselines/chemqa40/methods/hybrid_rankt5/scripts/run_hybrid_rankt5.py \
    --paper-text-view "$INPUTS/views/paper_text_view.jsonl" \
    --hybrid-results "$RUNS/hybrid_rrf/20260424_chemqa500_bm25_specter2_rrf/raw_results.jsonl" \
    --run-root "$RUNS/hybrid_rankt5" \
    --cache-root "$CACHES/hybrid_rankt5/$RUN_TAG" \
    --run-name "$RUN_TAG" \
    --candidate-topk 20

run_method \
  "hybrid_gpt54mini" \
  "$RUNS/hybrid_gpt54mini/$RUN_TAG/summary.json" \
  "$PY" baselines/chemqa40/methods/hybrid_gpt54mini/scripts/run_hybrid_gpt54mini.py \
    --paper-text-view "$INPUTS/views/paper_text_view.jsonl" \
    --hybrid-results "$RUNS/hybrid_rrf/20260424_chemqa500_bm25_specter2_rrf/raw_results.jsonl" \
    --run-root "$RUNS/hybrid_gpt54mini" \
    --run-name "$RUN_TAG" \
    --candidate-topk 20 \
    --max-workers 8

echo "[$(timestamp)] ChemQA500 remaining baseline queue completed"
collect_report
