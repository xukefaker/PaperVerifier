#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

PROJECT_ROOT = Path("/workspace/ChemVerify")
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

CACHE_ROOT = Path(os.environ.get("CHEMVERIFY_CACHE_ROOT", "/workspace/caches"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(os.environ["HF_HOME"]) / "transformers"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
os.environ.setdefault("CHEMVERIFY_DENSE_DEVICE", "cpu")
os.environ.setdefault("CHEMVERIFY_RERANKER_DEVICE", "cuda:0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from chemverify.config import Settings
from chemverify.models import (  # noqa: E402
    EvidenceChunk,
    PaperResult,
    QueryPlan,
    ScopeConstraints,
    SearchResponse,
    StructuredRationale,
    TokenUsage,
    VerifierRubric,
)
from chemverify.planner import PlannerResult  # noqa: E402
from chemverify.search import (  # noqa: E402
    SearchEngine,
    _aggregate_top_local_scores,
    _normalize_scores,
)
from chemverify.storage import LocalStore  # noqa: E402
from chemverify.utils import cosine_similarity_matrix, truncate_text  # noqa: E402


@dataclass(slots=True)
class QueryItem:
    query_id: str
    original_qa_id: str
    target_paper_id: str
    topic: str
    query_text: str


def load_queries(path: Path, mode: str, per_topic: int) -> list[QueryItem]:
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
    if mode == "full":
        return [
            QueryItem(
                query_id=str(row["query_id"]),
                original_qa_id=row["original_qa_id"],
                target_paper_id=row["target_paper_id"],
                topic=row["topic"],
                query_text=row["query_text"],
            )
            for row in rows
        ]

    selected: list[QueryItem] = []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        topic = row["topic"]
        if counts[topic] >= per_topic:
            continue
        counts[topic] += 1
        selected.append(
            QueryItem(
                query_id=str(row["query_id"]),
                original_qa_id=row["original_qa_id"],
                target_paper_id=row["target_paper_id"],
                topic=topic,
                query_text=row["query_text"],
            )
        )
    return selected


def make_minimal_plan(query: str) -> QueryPlan:
    return QueryPlan(
        mode="identity_minimal",
        user_query=query,
        global_query=query,
        scope_constraints=ScopeConstraints(),
        entity_terms=[],
        exact_phrases=[],
        aspect_queries=[],
        verifier_rubric=VerifierRubric(
            must_satisfy=[],
            should_satisfy=[],
            rejection_rules=[],
        ),
        evidence_buckets=[],
    )


def install_no_planner_patch(engine: SearchEngine) -> None:
    def minimal_plan(_planner_self: object, query: str) -> PlannerResult:
        return PlannerResult(plan=make_minimal_plan(query), usage=TokenUsage())

    engine.planner.plan = MethodType(minimal_plan, engine.planner)


def install_no_verifier_patch(engine: SearchEngine) -> None:
    original_shortlist = engine._shortlist_for_verifier

    def _shortlist_for_verifier_with_cache(
        self: SearchEngine,
        *,
        query_plan: QueryPlan,
        candidate_pool: list[tuple[str, float]],
        coarse_scores: dict[str, float],
        narrowed_sections: dict[str, list[dict[str, Any]]],
        evidence_packs: dict[str, dict[str, list[Any]]],
    ) -> tuple[list[tuple[str, float]], list[dict[str, object]]]:
        shortlisted_pool, shortlist_summary = original_shortlist(
            query_plan=query_plan,
            candidate_pool=candidate_pool,
            coarse_scores=coarse_scores,
            narrowed_sections=narrowed_sections,
            evidence_packs=evidence_packs,
        )
        self._ablation_shortlist_score_map = {
            str(row["paper_id"]): float(row["pre_verifier_score"]) for row in shortlist_summary
        }
        return shortlisted_pool, shortlist_summary

    def _verify_candidates_no_verifier(
        self: SearchEngine,
        runtime: Any,
        query_plan: QueryPlan,
        candidate_pool: list[tuple[str, float]],
        coarse_scores: dict[str, float],
        evidence_packs: dict[str, dict[str, list[Any]]],
        narrowed_sections: dict[str, list[dict[str, Any]]],
        top_k: int,
        progress_callback: Any = None,
    ) -> tuple[dict[str, list[PaperResult]], TokenUsage]:
        total_candidates = len(candidate_pool)
        if total_candidates == 0:
            return {"satisfied": [], "partial": [], "rejected": []}, TokenUsage()

        ordered_ids = [paper_id for paper_id, _ in candidate_pool]
        coarse_values = np.array([float(coarse_scores.get(paper_id, 0.0)) for paper_id in ordered_ids], dtype=float)
        shortlist_score_map = getattr(self, "_ablation_shortlist_score_map", {})

        grouped_results: dict[str, list[PaperResult]] = {"satisfied": [], "partial": [], "rejected": []}
        self._emit_progress(
            progress_callback,
            "final_verifier",
            "Scoring candidate papers with pre-verifier aggregation.",
            stage_progress=0.05,
            completed_items=0,
            total_items=total_candidates,
        )
        for index, paper_id in enumerate(ordered_ids, start=1):
            paper = runtime.paper_lookup[paper_id]
            rank_score = float(shortlist_score_map.get(paper_id, 0.0))
            matched_sections = [row.get("section_title", "") for row in narrowed_sections.get(paper_id, []) if row.get("section_title")]
            grouped_results["satisfied"].append(
                PaperResult(
                    paper_id=paper.paper_id,
                    title=paper.title,
                    score=float(rank_score),
                    coarse_score=float(coarse_values[index - 1]),
                    verifier_score=0.0,
                    venue=paper.venue,
                    year=paper.year,
                    track=paper.track,
                    verdict="satisfied",
                    confidence=float(rank_score),
                    rationale="Final verifier disabled; ranked by normalized pre-verifier aggregation.",
                    rationale_structured=StructuredRationale(
                        main_reason="Final verifier disabled; ranked by normalized coarse, section, and evidence scores."
                    ),
                    matched_sections=matched_sections,
                    matched_sections_summary={},
                    evidence_chunks=evidence_packs.get(paper_id, {}),
                    main_image_url=None,
                    abstract=paper.abstract or None,
                    authors=[],
                    affiliations=[],
                    authors_structured=[],
                    structured_summary=None,
                    enriched_metadata=None,
                )
            )
            self._emit_progress(
                progress_callback,
                "final_verifier",
                f"Ranked {index}/{total_candidates} candidate papers without verifier.",
                completed_items=index,
                total_items=total_candidates,
            )

        grouped_results["satisfied"].sort(key=lambda item: item.score, reverse=True)
        grouped_results["satisfied"] = grouped_results["satisfied"][:top_k]
        return grouped_results, TokenUsage()

    engine._shortlist_for_verifier = MethodType(_shortlist_for_verifier_with_cache, engine)
    engine._verify_candidates = MethodType(_verify_candidates_no_verifier, engine)


def install_no_content_refinement_patch(engine: SearchEngine) -> None:
    def _section_narrowing_no_content_refinement(
        self: SearchEngine,
        runtime: Any,
        query_plan: QueryPlan,
        candidate_pool: list[tuple[str, float]],
        progress_callback: Any = None,
    ) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
        total_candidates = len(candidate_pool)
        narrowed = {paper_id: [] for paper_id, _ in candidate_pool}
        summary = {paper_id: [] for paper_id, _ in candidate_pool}
        for processed_count, _ in enumerate(candidate_pool, start=1):
            self._emit_progress(
                progress_callback,
                "section_narrowing",
                f"Skipped fine-grained section narrowing for {processed_count}/{total_candidates} candidate papers.",
                completed_items=processed_count,
                total_items=total_candidates,
            )
        return narrowed, summary

    def _assemble_evidence_no_content_refinement(
        self: SearchEngine,
        runtime: Any,
        query_plan: QueryPlan,
        candidate_pool: list[tuple[str, float]],
        narrowed_sections: dict[str, list[dict]],
        progress_callback: Any = None,
    ) -> dict[str, dict[str, list[EvidenceChunk]]]:
        query_cache = self._build_query_cache(
            runtime.chunk_bm25,
            runtime.chunk_encoder,
            runtime.chunk_vectors,
            runtime.chunk_ids,
            [query_plan.global_query],
        )
        cached = query_cache[query_plan.global_query]
        total_candidates = len(candidate_pool)
        evidence_packs: dict[str, dict[str, list[EvidenceChunk]]] = {}
        total_required_chunks = max(1, sum(max(1, bucket.target_chunks) for bucket in query_plan.evidence_buckets))

        for processed_count, (paper_id, _) in enumerate(candidate_pool, start=1):
            chunk_rows = runtime.chunks_by_paper.get(paper_id, [])
            if not chunk_rows:
                evidence_packs[paper_id] = {bucket.bucket_id: [] for bucket in query_plan.evidence_buckets}
                self._emit_progress(
                    progress_callback,
                    "evidence_assembly",
                    f"Assembled simplified evidence for {processed_count}/{total_candidates} candidate papers.",
                    completed_items=processed_count,
                    total_items=total_candidates,
                )
                continue

            chunk_indices = np.array([row["index"] for row in chunk_rows], dtype=int)
            local_sparse = _normalize_scores(cached["sparse"][chunk_indices])
            local_dense = _normalize_scores(cosine_similarity_matrix(cached["dense_vector"], runtime.chunk_vectors[chunk_indices]))
            base_scores = self.settings.evidence_sparse_weight * local_sparse + self.settings.evidence_dense_weight * local_dense
            ranked_rows = [
                (row_index, float(score))
                for row_index, score in sorted(enumerate(base_scores.tolist()), key=lambda item: item[1], reverse=True)
                if score > 0
            ][:total_required_chunks]

            selected_chunks = [
                EvidenceChunk(
                    paper_id=paper_id,
                    bucket_id="global_top_chunks",
                    chunk_id=chunk_rows[row_index]["chunk_id"],
                    chunk_type=chunk_rows[row_index]["chunk_type"],
                    score=float(score),
                    source_query=query_plan.global_query,
                    heading=chunk_rows[row_index]["heading"],
                    section_path=chunk_rows[row_index]["section_path"],
                    page_start=chunk_rows[row_index]["page_start"],
                    page_end=chunk_rows[row_index]["page_end"],
                    text=truncate_text(chunk_rows[row_index]["text"], limit=self.settings.evidence_chunk_text_limit),
                )
                for row_index, score in ranked_rows
            ]

            bucket_chunks: dict[str, list[EvidenceChunk]] = {}
            cursor = 0
            for bucket in query_plan.evidence_buckets:
                need = max(1, bucket.target_chunks)
                assigned = []
                for chunk in selected_chunks[cursor : cursor + need]:
                    assigned.append(chunk.model_copy(update={"bucket_id": bucket.bucket_id}))
                bucket_chunks[bucket.bucket_id] = assigned
                cursor += need
            evidence_packs[paper_id] = bucket_chunks
            self._emit_progress(
                progress_callback,
                "evidence_assembly",
                f"Assembled simplified evidence for {processed_count}/{total_candidates} candidate papers.",
                completed_items=processed_count,
                total_items=total_candidates,
            )
        return evidence_packs

    def _shortlist_for_verifier_no_content_refinement(
        self: SearchEngine,
        *,
        query_plan: QueryPlan,
        candidate_pool: list[tuple[str, float]],
        coarse_scores: dict[str, float],
        narrowed_sections: dict[str, list[dict]],
        evidence_packs: dict[str, dict[str, list[EvidenceChunk]]],
    ) -> tuple[list[tuple[str, float]], list[dict[str, object]]]:
        if not candidate_pool:
            return [], []

        ordered_ids = [paper_id for paper_id, _ in candidate_pool]
        coarse_values = np.array([float(coarse_scores.get(paper_id, 0.0)) for paper_id in ordered_ids], dtype=float)
        coarse_norm = _normalize_scores(coarse_values)

        scored_rows: list[dict[str, object]] = []
        for index, paper_id in enumerate(ordered_ids):
            scored_rows.append(
                {
                    "paper_id": paper_id,
                    "coarse_score": float(coarse_values[index]),
                    "section_score": 0.0,
                    "evidence_score": 0.0,
                    "pre_verifier_score": float(coarse_norm[index]),
                }
            )

        scored_rows.sort(key=lambda row: (-float(row["pre_verifier_score"]), row["paper_id"]))
        shortlist_limit = min(self.settings.verifier_candidate_limit, len(scored_rows))
        shortlist_ids = {row["paper_id"] for row in scored_rows[:shortlist_limit]}
        shortlisted_pool = [
            (paper_id, coarse_score)
            for paper_id, coarse_score in candidate_pool
            if paper_id in shortlist_ids
        ]
        shortlist_summary = [
            {
                **row,
                "shortlisted": row["paper_id"] in shortlist_ids,
            }
            for row in scored_rows
        ]
        return shortlisted_pool, shortlist_summary

    engine._section_narrowing = MethodType(_section_narrowing_no_content_refinement, engine)
    engine._assemble_evidence = MethodType(_assemble_evidence_no_content_refinement, engine)
    engine._shortlist_for_verifier = MethodType(_shortlist_for_verifier_no_content_refinement, engine)


def select_display_results(result: SearchResponse, display_k: int) -> list[PaperResult]:
    if display_k <= 0:
        return []
    display_results = list(result.satisfied[:display_k])
    if len(display_results) < display_k:
        display_results.extend(result.partial[: display_k - len(display_results)])
    return display_results[:display_k]


def find_bucket(result: SearchResponse, paper_id: str) -> tuple[str, PaperResult | None]:
    for bucket_name, items in (
        ("satisfied", result.satisfied),
        ("partial", result.partial),
        ("rejected", result.rejected),
    ):
        for item in items:
            if item.paper_id == paper_id:
                return bucket_name, item
    return "not_returned", None


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [r for r in records if r["job_status"] == "completed"]
    failed = [r for r in records if r["job_status"] != "completed"]
    if completed:
        hit1 = sum(r["target_hit_at_1"] for r in completed) / len(completed)
        hit3 = sum(r["target_hit_at_3"] for r in completed) / len(completed)
        hit5 = sum(r["target_hit_at_5"] for r in completed) / len(completed)
        hit10 = sum(r["target_hit_at_10"] for r in completed) / len(completed)
        mrr = sum(r["mrr"] for r in completed) / len(completed)
        satisfied = sum(r["target_bucket"] == "satisfied" for r in completed) / len(completed)
        partial_or_better = sum(r["target_bucket"] in {"satisfied", "partial"} for r in completed) / len(completed)
        avg_latency = statistics.mean(r["total_latency_sec"] for r in completed if r["total_latency_sec"] is not None)
    else:
        hit1 = hit3 = hit5 = hit10 = mrr = satisfied = partial_or_better = avg_latency = 0.0

    by_topic: dict[str, dict[str, Any]] = {}
    topic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in completed:
        topic_groups[record["topic"]].append(record)
    for topic, group in sorted(topic_groups.items()):
        by_topic[topic] = {
            "count": len(group),
            "hit_at_1": sum(r["target_hit_at_1"] for r in group) / len(group),
            "hit_at_3": sum(r["target_hit_at_3"] for r in group) / len(group),
            "hit_at_5": sum(r["target_hit_at_5"] for r in group) / len(group),
            "hit_at_10": sum(r["target_hit_at_10"] for r in group) / len(group),
            "mrr": sum(r["mrr"] for r in group) / len(group),
            "target_in_satisfied": sum(r["target_bucket"] == "satisfied" for r in group) / len(group),
            "target_in_partial_or_better": sum(r["target_bucket"] in {"satisfied", "partial"} for r in group)
            / len(group),
        }

    return {
        "query_count": len(records),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_5": hit5,
        "hit_at_10": hit10,
        "mrr": mrr,
        "target_in_satisfied": satisfied,
        "target_in_partial_or_better": partial_or_better,
        "avg_latency_sec": avg_latency,
        "by_topic": by_topic,
    }


def write_summary_md(path: Path, summary: dict[str, Any], run_name: str, profile: str, corpus_key: str) -> None:
    lines = [
        f"# ChemQA40 Search Ablation Summary: {run_name}",
        "",
        f"- profile: `{profile}`",
        f"- corpus: `{corpus_key}`",
        f"- query_count: `{summary['query_count']}`",
        f"- completed_count: `{summary['completed_count']}`",
        f"- failed_count: `{summary['failed_count']}`",
        f"- Hit@1: `{summary['hit_at_1']:.3f}`",
        f"- Hit@3: `{summary['hit_at_3']:.3f}`",
        f"- Hit@5: `{summary['hit_at_5']:.3f}`",
        f"- Hit@10: `{summary['hit_at_10']:.3f}`",
        f"- MRR: `{summary['mrr']:.3f}`",
        f"- target_in_satisfied: `{summary['target_in_satisfied']:.3f}`",
        f"- target_in_partial_or_better: `{summary['target_in_partial_or_better']:.3f}`",
        f"- avg_latency_sec: `{summary['avg_latency_sec']:.2f}`",
        "",
        "## By Topic",
        "",
    ]
    for topic, metrics in summary["by_topic"].items():
        lines.extend(
            [
                f"### {topic}",
                f"- count: `{metrics['count']}`",
                f"- Hit@1: `{metrics['hit_at_1']:.3f}`",
                f"- Hit@3: `{metrics['hit_at_3']:.3f}`",
                f"- Hit@5: `{metrics['hit_at_5']:.3f}`",
                f"- Hit@10: `{metrics['hit_at_10']:.3f}`",
                f"- MRR: `{metrics['mrr']:.3f}`",
                f"- target_in_satisfied: `{metrics['target_in_satisfied']:.3f}`",
                f"- target_in_partial_or_better: `{metrics['target_in_partial_or_better']:.3f}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def configure_engine(profile: str, run_dir: Path) -> SearchEngine:
    settings = Settings.from_env(PROJECT_ROOT)
    store = LocalStore(settings, root_dir=settings.search_current_dir)
    store.trace_dir = run_dir / "traces"
    store.trace_dir.mkdir(parents=True, exist_ok=True)
    engine = SearchEngine(settings, store)

    if profile == "no_planner":
        install_no_planner_patch(engine)
    elif profile == "no_verifier":
        install_no_verifier_patch(engine)
    elif profile == "no_content_refinement":
        install_no_content_refinement_patch(engine)
    elif profile != "full":
        raise RuntimeError(f"Unsupported profile: {profile}")

    return engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--profile", choices=["full", "no_planner", "no_verifier", "no_content_refinement"], required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--samples-per-topic", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--display-k", type=int, default=10)
    parser.add_argument("--corpus-key", default="chemqa40/2026/all")
    args = parser.parse_args()

    annotation_path = Path(args.annotation_file)
    run_dir = Path(args.run_dir)
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(annotation_path, args.mode, args.samples_per_topic)
    engine = configure_engine(args.profile, run_dir)
    raw_results_path = run_dir / "raw_results.jsonl"
    existing_records = load_existing_records(raw_results_path)
    completed_query_ids = {str(record["query_id"]) for record in existing_records}
    pending_queries = [item for item in queries if item.query_id not in completed_query_ids]

    run_config = {
        "annotation_file": str(annotation_path),
        "profile": args.profile,
        "mode": args.mode,
        "samples_per_topic": args.samples_per_topic,
        "top_k": args.top_k,
        "display_k": args.display_k,
        "corpus_key": args.corpus_key,
        "query_count": len(queries),
        "existing_record_count": len(existing_records),
        "pending_query_count": len(pending_queries),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    records: list[dict[str, Any]] = list(existing_records)
    if existing_records:
        print(
            f"Resuming existing run: found {len(existing_records)} existing records, {len(pending_queries)} pending queries.",
            flush=True,
        )
    else:
        print(f"Starting fresh run with {len(pending_queries)} queries.", flush=True)

    with raw_results_path.open("a", encoding="utf-8") as raw_file:
        for offset, item in enumerate(pending_queries, start=1):
            absolute_index = len(existing_records) + offset
            print(f"[{absolute_index}/{len(queries)}] {item.query_id} :: {item.query_text}", flush=True)
            try:
                response = engine.search(
                    item.query_text,
                    top_k=args.top_k,
                    workspace_scope=[args.corpus_key],
                )
                display_results = select_display_results(response, args.display_k)
                display_ids = [paper.paper_id for paper in display_results]
                target_bucket, target_item = find_bucket(response, item.target_paper_id)
                target_rank = display_ids.index(item.target_paper_id) + 1 if item.target_paper_id in display_ids else None
                trace = engine.store.load_trace(response.trace_id)
                record = {
                    "query_id": item.query_id,
                    "original_qa_id": item.original_qa_id,
                    "paper_id": item.target_paper_id,
                    "topic": item.topic,
                    "query_text": item.query_text,
                    "job_status": "completed",
                    "trace_id": response.trace_id,
                    "total_latency_sec": round(sum((trace.timings_ms or {}).values()) / 1000.0, 3) if trace else None,
                    "target_rank": target_rank,
                    "target_hit_at_1": 1 if target_rank == 1 else 0,
                    "target_hit_at_3": 1 if target_rank is not None and target_rank <= 3 else 0,
                    "target_hit_at_5": 1 if target_rank is not None and target_rank <= 5 else 0,
                    "target_hit_at_10": 1 if target_rank is not None and target_rank <= 10 else 0,
                    "mrr": (1.0 / target_rank) if target_rank is not None else 0.0,
                    "target_bucket": target_bucket,
                    "target_verdict": target_item.verdict if target_item is not None else None,
                    "target_score": target_item.score if target_item is not None else None,
                    "target_coarse_score": target_item.coarse_score if target_item is not None else None,
                    "target_verifier_score": target_item.verifier_score if target_item is not None else None,
                    "top10_paper_ids": display_ids,
                    "counts": {
                        "satisfied": len(response.satisfied),
                        "partial": len(response.partial),
                        "rejected": len(response.rejected),
                    },
                    "trace_timings_ms": trace.timings_ms if trace else {},
                }
            except Exception as exc:  # noqa: BLE001
                record = {
                    "query_id": item.query_id,
                    "original_qa_id": item.original_qa_id,
                    "paper_id": item.target_paper_id,
                    "topic": item.topic,
                    "query_text": item.query_text,
                    "job_status": "failed",
                    "error": str(exc),
                    "trace_id": None,
                    "total_latency_sec": None,
                    "target_rank": None,
                    "target_hit_at_1": 0,
                    "target_hit_at_3": 0,
                    "target_hit_at_5": 0,
                    "target_hit_at_10": 0,
                    "mrr": 0.0,
                    "target_bucket": "not_returned",
                    "target_verdict": None,
                    "target_score": None,
                    "target_coarse_score": None,
                    "target_verifier_score": None,
                    "top10_paper_ids": [],
                    "counts": {},
                    "trace_timings_ms": {},
                }
            records.append(record)
            raw_file.write(json.dumps(record, ensure_ascii=False))
            raw_file.write("\n")
            raw_file.flush()

    summary = compute_metrics(records)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(run_dir / "summary.md", summary, run_dir.name, args.profile, args.corpus_key)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
