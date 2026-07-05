#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/workspace/ChemVerify"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
SCRIPT_PATH="$PROJECT_ROOT/scripts/benchmarks/chemqa40/search_ablation_replay.py"
ANNOTATION_FILE="$PROJECT_ROOT/data/benchmarks/chemqa40/ablation/input/chemqa40_search_benchmark_final.jsonl"
RUN_ROOT="$PROJECT_ROOT/data/benchmarks/chemqa40/ablation/runs"
CORPUS_KEY="chemqa40/2026/all"

run_ablation() {
  local profile="$1"
  local mode="$2"
  local run_name="$3"

  echo "==== START ${run_name} (${profile}, ${mode}) ===="
  "$PYTHON_BIN" "$SCRIPT_PATH" \
    --annotation-file "$ANNOTATION_FILE" \
    --run-dir "$RUN_ROOT/$run_name" \
    --profile "$profile" \
    --mode "$mode" \
    --samples-per-topic 1 \
    --top-k 10 \
    --display-k 10 \
    --corpus-key "$CORPUS_KEY"
  echo "==== DONE ${run_name} ===="
}

run_ablation "no_planner" "smoke" "20260422_no_planner_top20_smoke"
run_ablation "no_content_refinement" "smoke" "20260422_no_content_refinement_top20_smoke"
run_ablation "no_verifier" "smoke" "20260422_no_verifier_top20_smoke"

run_ablation "full" "full" "20260422_full_top20_full"
run_ablation "no_planner" "full" "20260422_no_planner_top20_full"
run_ablation "no_content_refinement" "full" "20260422_no_content_refinement_top20_full"
run_ablation "no_verifier" "full" "20260422_no_verifier_top20_full"
