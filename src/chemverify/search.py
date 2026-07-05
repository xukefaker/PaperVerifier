from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Callable

import httpx
import numpy as np
from rank_bm25 import BM25Okapi

from .config import Settings
from .encoders import EncoderConfig, SentenceTransformerEncoder
from .models import (
    EvidenceBucket,
    EvidenceChunk,
    ObjectRecord,
    PaperRecord,
    PaperResult,
    QueryPlan,
    RecallItem,
    ScopeConstraints,
    SearchResponse,
    SearchTrace,
    TokenUsage,
)
from .planner import QueryPlanner
from .presentation import (
    build_matched_sections_summary,
    load_cached_paper_authorship,
    load_cached_paper_enrichment,
    select_main_image_url,
    structure_rationale_text,
)
from .provider import require_openai_model
from .reranker import CrossEncoderReranker, RerankerConfig
from .storage import LocalStore
from .utils import cosine_similarity_matrix, make_trace_id, now_iso, tokenize, top_k_from_scores, truncate_text, weighted_rrf


@dataclass(slots=True)
class _Runtime:
    papers: list[PaperRecord]
    paper_lookup: dict[str, PaperRecord]
    paper_ids_by_corpus: dict[str, set[str]]
    paper_bm25: BM25Okapi
    paper_ids: list[str]
    paper_vectors: np.ndarray
    paper_encoder: SentenceTransformerEncoder
    objects_by_paper: dict[str, list[ObjectRecord]]
    section_lookup: dict[str, dict]
    sections_by_paper: dict[str, list[dict]]
    section_bm25: BM25Okapi
    section_ids: list[str]
    section_vectors: np.ndarray
    chunk_lookup: dict[str, dict]
    chunks_by_paper: dict[str, list[dict]]
    chunks_by_section: dict[str, list[dict]]
    chunk_bm25: BM25Okapi
    chunk_ids: list[str]
    chunk_vectors: np.ndarray
    chunk_encoder: SentenceTransformerEncoder
    local_reranker: CrossEncoderReranker


_SEARCH_PROGRESS_STAGES = (
    "loading_index",
    "planning_query",
    "candidate_generation",
    "section_narrowing",
    "evidence_assembly",
    "final_verifier",
    "saving_trace",
)

_SEARCH_PROGRESS_STAGE_INDEX = {
    stage_name: index
    for index, stage_name in enumerate(_SEARCH_PROGRESS_STAGES, start=1)
}


@dataclass(slots=True, frozen=True)
class SearchProgressUpdate:
    stage: str
    message: str
    stage_index: int
    stage_total: int
    stage_progress: float
    overall_progress: float
    completed_items: int | None = None
    total_items: int | None = None


@dataclass(slots=True)
class _BucketSelectionState:
    paper_id: str
    bucket: EvidenceBucket
    chunk_rows: list[dict]
    preselected_rows: list[int]
    best_sparse: np.ndarray
    best_dense: np.ndarray
    best_query: list[str]
    reranker_query: str
    reranker_offset: int
    reranker_count: int


class SearchEngine:
    def __init__(self, settings: Settings, store: LocalStore | None = None) -> None:
        self.settings = settings
        self.store = store or LocalStore(settings)
        self.planner = QueryPlanner(settings)
        self.runtime: _Runtime | None = None
        self._load_lock = Lock()

    def load(self) -> None:
        if self.runtime is not None:
            return
        with self._load_lock:
            if self.runtime is not None:
                return
            papers = self.store.load_papers()
            objects = self.store.load_objects()
            sections = self.store.load_sections()
            chunks = self.store.load_chunks()
            paper_meta = self.store.load_index_meta("paper")
            section_meta = self.store.load_index_meta("section")
            chunk_meta = self.store.load_index_meta("chunk")
            paper_ids, paper_vectors = self.store.load_vectors("paper")
            section_ids, section_vectors = self.store.load_vectors("section")
            chunk_ids, chunk_vectors = self.store.load_vectors("chunk")

            if not papers or not sections or not chunks or not paper_ids or not section_ids or not chunk_ids:
                raise RuntimeError("Layout V2 index is missing. Run build-index first.")

            paper_tokens = paper_meta.get("tokens", [])
            section_tokens = section_meta.get("tokens", [])
            chunk_tokens = chunk_meta.get("tokens", [])
            if not paper_tokens or not section_tokens or not chunk_tokens:
                raise RuntimeError("Index metadata is incomplete. Run build-index again.")
            _assert_index_alignment("paper", paper_meta, paper_ids, paper_vectors)
            _assert_index_alignment("section", section_meta, section_ids, section_vectors)
            _assert_index_alignment("chunk", chunk_meta, chunk_ids, chunk_vectors)

            paper_lookup = {paper.paper_id: paper for paper in papers}
            paper_ids_by_corpus: dict[str, set[str]] = defaultdict(set)
            for paper in papers:
                paper_ids_by_corpus[_paper_corpus_key(paper)].add(paper.paper_id)
            objects_by_paper: dict[str, list[ObjectRecord]] = defaultdict(list)
            for obj in objects:
                objects_by_paper[obj.paper_id].append(obj)

            section_lookup: dict[str, dict] = {}
            sections_by_paper: dict[str, list[dict]] = defaultdict(list)
            for index, section in enumerate(sections):
                row = {
                    "section_id": section.section_id,
                    "paper_id": section.paper_id,
                    "section_title": section.section_title,
                    "section_path": section.section_path,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "text": section.text,
                    "index": index,
                }
                section_lookup[section.section_id] = row
                sections_by_paper[section.paper_id].append(row)

            chunk_lookup: dict[str, dict] = {}
            chunks_by_paper: dict[str, list[dict]] = defaultdict(list)
            chunks_by_section: dict[str, list[dict]] = defaultdict(list)
            for index, chunk in enumerate(chunks):
                row = {
                    "chunk_id": chunk.chunk_id,
                    "chunk_type": chunk.chunk_type,
                    "paper_id": chunk.paper_id,
                    "section_id": chunk.section_id,
                    "heading": chunk.heading,
                    "section_path": chunk.section_path,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "text": chunk.text,
                    "index": index,
                }
                chunk_lookup[chunk.chunk_id] = row
                chunks_by_paper[chunk.paper_id].append(row)
                chunks_by_section[chunk.section_id].append(row)

            paper_encoder = SentenceTransformerEncoder(
                EncoderConfig(
                    self.settings.paper_dense_model,
                    device=self.settings.dense_device,
                    batch_size=self.settings.dense_batch_size,
                )
            )
            chunk_encoder = SentenceTransformerEncoder(
                EncoderConfig(
                    self.settings.chunk_dense_model,
                    device=self.settings.dense_device,
                    batch_size=self.settings.dense_batch_size,
                )
            )
            local_reranker = CrossEncoderReranker(
                RerankerConfig(
                    model_name=self.settings.reranker_model,
                    device=self.settings.reranker_device or self.settings.dense_device,
                    batch_size=self.settings.reranker_batch_size,
                )
            )
            _assert_encoder_compatibility("paper", paper_meta, paper_vectors, paper_encoder)
            _assert_encoder_compatibility("section", section_meta, section_vectors, chunk_encoder)
            _assert_encoder_compatibility("chunk", chunk_meta, chunk_vectors, chunk_encoder)

            self.runtime = _Runtime(
                papers=papers,
                paper_lookup=paper_lookup,
                paper_ids_by_corpus=dict(paper_ids_by_corpus),
                paper_bm25=BM25Okapi(paper_tokens),
                paper_ids=paper_ids,
                paper_vectors=paper_vectors,
                paper_encoder=paper_encoder,
                objects_by_paper=objects_by_paper,
                section_lookup=section_lookup,
                sections_by_paper=sections_by_paper,
                section_bm25=BM25Okapi(section_tokens),
                section_ids=section_ids,
                section_vectors=section_vectors,
                chunk_lookup=chunk_lookup,
                chunks_by_paper=chunks_by_paper,
                chunks_by_section=chunks_by_section,
                chunk_bm25=BM25Okapi(chunk_tokens),
                chunk_ids=chunk_ids,
                chunk_vectors=chunk_vectors,
                chunk_encoder=chunk_encoder,
                local_reranker=local_reranker,
            )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        workspace_scope: list[str] | None = None,
        progress_callback: Callable[[SearchProgressUpdate], None] | None = None,
    ) -> SearchResponse:
        if self.runtime is None:
            self._emit_progress(
                progress_callback,
                "loading_index",
                "Loading layout paper, section, and chunk indexes.",
                stage_progress=0.0,
            )
            self.load()
            self._emit_progress(
                progress_callback,
                "loading_index",
                "Layout paper, section, and chunk indexes loaded.",
                stage_progress=1.0,
            )
        runtime = self.runtime
        assert runtime is not None

        overall_start = perf_counter()

        self._emit_progress(
            progress_callback,
            "planning_query",
            "Parsing the natural-language query.",
            stage_progress=0.0,
        )
        plan_start = perf_counter()
        planner_result = self.planner.plan(query)
        query_plan = planner_result.plan
        timings = {"planner": (perf_counter() - plan_start) * 1000}
        resolved_workspace_scope = self._resolve_workspace_scope(runtime, workspace_scope)
        if not resolved_workspace_scope:
            raise RuntimeError("The current workspace has no available corpora selected.")
        effective_scope = self._resolve_effective_scope(runtime, resolved_workspace_scope, query_plan.scope_constraints)
        if not effective_scope:
            raise RuntimeError("The query scope does not overlap with the current workspace corpus selection.")
        self._emit_progress(
            progress_callback,
            "planning_query",
            "Query plan created.",
            stage_progress=1.0,
        )

        self._emit_progress(
            progress_callback,
            "candidate_generation",
            "Generating a wide candidate paper pool.",
            stage_progress=0.0,
        )
        candidate_start = perf_counter()
        candidate_pool, coarse_scores, recall_items, filter_summary = self._candidate_generation(
            runtime,
            query_plan,
            effective_scope,
        )
        timings["candidate_generation"] = (perf_counter() - candidate_start) * 1000
        self._emit_progress(
            progress_callback,
            "candidate_generation",
            f"Candidate paper pool ready with {len(candidate_pool)} papers.",
            stage_progress=1.0,
        )

        self._emit_progress(
            progress_callback,
            "section_narrowing",
            "Narrowing each candidate paper to relevant sections.",
            completed_items=0,
            total_items=len(candidate_pool),
        )
        section_start = perf_counter()
        narrowed_sections, section_summary = self._section_narrowing(
            runtime,
            query_plan,
            candidate_pool,
            progress_callback=progress_callback,
        )
        timings["section_narrowing"] = (perf_counter() - section_start) * 1000
        self._emit_progress(
            progress_callback,
            "section_narrowing",
            f"Section narrowing completed for {len(candidate_pool)} candidate papers.",
            stage_progress=1.0,
            completed_items=len(candidate_pool),
            total_items=len(candidate_pool),
        )

        self._emit_progress(
            progress_callback,
            "evidence_assembly",
            "Collecting object-aware evidence chunks from narrowed sections.",
            completed_items=0,
            total_items=len(candidate_pool),
        )
        evidence_start = perf_counter()
        evidence_packs = self._assemble_evidence(
            runtime,
            query_plan,
            candidate_pool,
            narrowed_sections,
            progress_callback=progress_callback,
        )
        timings["evidence_assembly"] = (perf_counter() - evidence_start) * 1000
        self._emit_progress(
            progress_callback,
            "evidence_assembly",
            f"Evidence assembly completed for {len(candidate_pool)} candidate papers.",
            stage_progress=1.0,
            completed_items=len(candidate_pool),
            total_items=len(candidate_pool),
        )

        self._emit_progress(
            progress_callback,
            "final_verifier",
            "Selecting the top candidate papers for final verification.",
            stage_progress=0.0,
        )
        shortlist_start = perf_counter()
        verifier_candidate_pool, shortlist_summary = self._shortlist_for_verifier(
            query_plan=query_plan,
            candidate_pool=candidate_pool,
            coarse_scores=coarse_scores,
            narrowed_sections=narrowed_sections,
            evidence_packs=evidence_packs,
        )
        timings["pre_verifier_shortlist"] = (perf_counter() - shortlist_start) * 1000
        self._emit_progress(
            progress_callback,
            "final_verifier",
            f"Shortlisted {len(verifier_candidate_pool)} papers for final verification.",
            stage_progress=0.1,
            completed_items=0,
            total_items=len(verifier_candidate_pool),
        )
        verifier_start = perf_counter()
        grouped_results, verifier_usage = self._verify_candidates(
            runtime=runtime,
            query_plan=query_plan,
            candidate_pool=verifier_candidate_pool,
            coarse_scores=coarse_scores,
            evidence_packs=evidence_packs,
            narrowed_sections=narrowed_sections,
            top_k=top_k or self.settings.default_top_k,
            progress_callback=progress_callback,
        )
        timings["final_verifier"] = (perf_counter() - verifier_start) * 1000
        self._emit_progress(
            progress_callback,
            "final_verifier",
            f"Final verifier completed for {len(verifier_candidate_pool)} shortlisted papers.",
            stage_progress=1.0,
            completed_items=len(verifier_candidate_pool),
            total_items=len(verifier_candidate_pool),
        )
        timings["total"] = (perf_counter() - overall_start) * 1000

        self._emit_progress(
            progress_callback,
            "saving_trace",
            "Saving the structured search trace.",
            stage_progress=0.0,
        )
        trace_id = make_trace_id()
        trace = SearchTrace(
            trace_id=trace_id,
            created_at=now_iso(),
            mode="layout_llm_verifier",
            user_query=query,
            workspace_scope=resolved_workspace_scope,
            effective_scope=effective_scope,
            query_plan=query_plan,
            paper_recall=recall_items,
            evidence_packs=evidence_packs,
            filter_summary={
                **filter_summary,
                "section_narrowing": section_summary,
                "verifier_shortlist": shortlist_summary,
            },
            verifier_summary={
                "candidate_pool_count": len(candidate_pool),
                "verifier_candidate_limit": self.settings.verifier_candidate_limit,
                "verifier_shortlist_count": len(verifier_candidate_pool),
                "satisfied_count": len(grouped_results["satisfied"]),
                "partial_count": len(grouped_results["partial"]),
                "rejected_count": len(grouped_results["rejected"]),
                "reranker_backend": runtime.local_reranker.backend_name,
            },
            final_results=grouped_results,
            timings_ms=timings,
            token_usage=TokenUsage(
                prompt_tokens=planner_result.usage.prompt_tokens + verifier_usage.prompt_tokens,
                completion_tokens=planner_result.usage.completion_tokens + verifier_usage.completion_tokens,
                total_tokens=planner_result.usage.total_tokens + verifier_usage.total_tokens,
                cost_estimate_usd=_sum_costs(planner_result.usage.cost_estimate_usd, verifier_usage.cost_estimate_usd),
            ),
        )
        self.store.save_trace(trace)
        self._emit_progress(
            progress_callback,
            "saving_trace",
            "Structured search trace saved.",
            stage_progress=1.0,
        )
        self._emit_progress(progress_callback, "completed", "Search completed.", stage_progress=1.0)
        return SearchResponse(
            trace_id=trace_id,
            mode="layout_llm_verifier",
            workspace_scope=resolved_workspace_scope,
            query_scope=query_plan.scope_constraints,
            effective_scope=effective_scope,
            **grouped_results,
        )

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[SearchProgressUpdate], None] | None,
        stage: str,
        message: str,
        *,
        stage_progress: float | None = None,
        completed_items: int | None = None,
        total_items: int | None = None,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            _make_progress_update(
                stage,
                message,
                stage_progress=stage_progress,
                completed_items=completed_items,
                total_items=total_items,
            )
        )

    def _candidate_generation(
        self,
        runtime: _Runtime,
        query_plan: QueryPlan,
        effective_scope: list[str],
    ) -> tuple[list[tuple[str, float]], dict[str, float], list[RecallItem], dict[str, object]]:
        allowed_paper_ids = self._allowed_paper_ids(runtime, effective_scope)
        source_limit = self.settings.candidate_source_limit

        source_rankings: dict[str, list[str]] = {}
        source_weights = {
            "paper_sparse": self.settings.paper_sparse_rrf_weight,
            "paper_dense": self.settings.paper_dense_rrf_weight,
            "section_aggregated": self.settings.chunk_aggregated_rrf_weight,
            "literal_entity": self.settings.literal_entity_rrf_weight,
            "exact_phrase": self.settings.exact_phrase_rrf_weight,
        }
        aspect_hits: dict[str, set[str]] = defaultdict(set)
        source_hits: dict[str, set[str]] = defaultdict(set)

        paper_sparse_rank, sparse_aspects = self._rank_paper_source(runtime, query_plan, allowed_paper_ids, source_limit, dense=False)
        paper_dense_rank, dense_aspects = self._rank_paper_source(runtime, query_plan, allowed_paper_ids, source_limit, dense=True)
        section_aggregated_rank, section_aspects = self._rank_section_aggregated_source(runtime, query_plan, allowed_paper_ids, source_limit)
        literal_entity_rank = self._rank_literal_entity_source(runtime, query_plan, allowed_paper_ids, source_limit)
        exact_phrase_rank = self._rank_exact_phrase_source(runtime, query_plan, allowed_paper_ids, source_limit)

        source_rankings["paper_sparse"] = paper_sparse_rank
        source_rankings["paper_dense"] = paper_dense_rank
        source_rankings["section_aggregated"] = section_aggregated_rank
        source_rankings["literal_entity"] = literal_entity_rank
        source_rankings["exact_phrase"] = exact_phrase_rank

        for mapping in (sparse_aspects, dense_aspects, section_aspects):
            for paper_id, aspect_ids in mapping.items():
                aspect_hits[paper_id].update(aspect_ids)

        for source_name, ranking in source_rankings.items():
            for paper_id in ranking:
                source_hits[paper_id].add(source_name)

        coarse_scores = weighted_rrf(source_rankings, source_weights)
        total_aspects = max(1, len(query_plan.aspect_queries))
        source_count = max(1, len([items for items in source_rankings.values() if items]))
        literal_entity_set = set(literal_entity_rank)
        exact_phrase_set = set(exact_phrase_rank)

        for paper_id in set().union(*[set(items) for items in source_rankings.values() if items]):
            aspect_coverage = len(aspect_hits.get(paper_id, set())) / total_aspects
            diversity = len(source_hits.get(paper_id, set())) / source_count
            coarse_scores[paper_id] = coarse_scores.get(paper_id, 0.0)
            coarse_scores[paper_id] += self.settings.aspect_coverage_bonus * aspect_coverage
            coarse_scores[paper_id] += self.settings.source_diversity_bonus * diversity
            coarse_scores[paper_id] += self.settings.literal_entity_bonus if paper_id in literal_entity_set else 0.0
            coarse_scores[paper_id] += self.settings.exact_phrase_bonus if paper_id in exact_phrase_set else 0.0

        candidate_pool = top_k_from_scores(coarse_scores, self.settings.candidate_pool_size)
        recall_items: list[RecallItem] = []
        for source_name, ranking in source_rankings.items():
            for rank, paper_id in enumerate(ranking[:50], start=1):
                recall_items.append(
                    RecallItem(
                        item_id=paper_id,
                        source=source_name,
                        score=float(coarse_scores.get(paper_id, 0.0)),
                        rank=rank,
                    )
                )

        filter_summary: dict[str, object] = {
            "allowed_paper_ids": len(allowed_paper_ids),
            "effective_scope": effective_scope,
            "candidate_pool_count": len(candidate_pool),
            "candidate_pool_ids": [paper_id for paper_id, _ in candidate_pool],
            "source_sizes": {source_name: len(ranking) for source_name, ranking in source_rankings.items()},
            "entity_terms": query_plan.entity_terms,
            "exact_phrases": query_plan.exact_phrases,
        }
        return candidate_pool, coarse_scores, recall_items, filter_summary

    def _section_narrowing(
        self,
        runtime: _Runtime,
        query_plan: QueryPlan,
        candidate_pool: list[tuple[str, float]],
        progress_callback: Callable[[SearchProgressUpdate], None] | None = None,
    ) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
        narrowed: dict[str, list[dict]] = {}
        summary: dict[str, list[dict]] = {}
        unique_queries = [query_plan.global_query, *[aspect.query for aspect in query_plan.aspect_queries]]
        for bucket in query_plan.evidence_buckets:
            unique_queries.extend(bucket.queries)
        cache = self._build_query_cache(runtime.section_bm25, runtime.chunk_encoder, runtime.section_vectors, runtime.section_ids, unique_queries)

        weighted_queries: list[tuple[str, float]] = [(query_plan.global_query, 1.0)]
        weighted_queries.extend((aspect.query, aspect.weight) for aspect in query_plan.aspect_queries)
        for bucket in query_plan.evidence_buckets:
            bucket_queries = [query for query in bucket.queries if query.strip()]
            if not bucket_queries:
                continue
            bucket_query_weight = 0.35 / len(bucket_queries)
            weighted_queries.extend((query, bucket_query_weight) for query in bucket_queries)
        normalized_entity_terms = [_normalized_match_text(item) for item in query_plan.entity_terms if _normalized_match_text(item)]
        normalized_exact_phrases = [_normalized_match_text(item) for item in query_plan.exact_phrases if _normalized_match_text(item)]

        total_candidates = len(candidate_pool)
        for processed_count, (paper_id, _) in enumerate(candidate_pool, start=1):
            section_rows = runtime.sections_by_paper.get(paper_id, [])
            if not section_rows:
                narrowed[paper_id] = []
                summary[paper_id] = []
                self._emit_progress(
                    progress_callback,
                    "section_narrowing",
                    f"Narrowed relevant sections for {processed_count}/{total_candidates} candidate papers.",
                    completed_items=processed_count,
                    total_items=total_candidates,
                )
                continue

            section_indices = np.array([row["index"] for row in section_rows], dtype=int)
            aggregate = np.zeros(len(section_rows), dtype=float)
            for query, weight in weighted_queries:
                if query not in cache:
                    continue
                cached = cache[query]
                local_sparse = _normalize_scores(cached["sparse"][section_indices])
                local_dense = _normalize_scores(cosine_similarity_matrix(cached["dense_vector"], runtime.section_vectors[section_indices]))
                aggregate += weight * (0.55 * local_sparse + 0.45 * local_dense)

            for idx, row in enumerate(section_rows):
                bonus = 0.0
                normalized = _normalized_match_text(f"{row['section_title']} {' '.join(row['section_path'])} {row['text']}")
                if any(_contains_normalized_phrase(normalized, term) for term in normalized_entity_terms):
                    bonus += 0.08
                if any(_contains_normalized_phrase(normalized, phrase) for phrase in normalized_exact_phrases):
                    bonus += 0.05
                aggregate[idx] += bonus

            top_rows = sorted(zip(section_rows, aggregate.tolist()), key=lambda item: item[1], reverse=True)[:4]
            selected = []
            for row, score in top_rows:
                if score <= 0:
                    continue
                selected.append({**row, "score": float(score)})
            narrowed[paper_id] = selected
            summary[paper_id] = [
                {
                    "section_id": row["section_id"],
                    "section_title": row["section_title"],
                    "section_path": row["section_path"],
                    "pages": [row["page_start"], row["page_end"]],
                    "score": row["score"],
                }
                for row in selected
            ]
            self._emit_progress(
                progress_callback,
                "section_narrowing",
                f"Narrowed relevant sections for {processed_count}/{total_candidates} candidate papers.",
                completed_items=processed_count,
                total_items=total_candidates,
            )
        return narrowed, summary

    def _assemble_evidence(
        self,
        runtime: _Runtime,
        query_plan: QueryPlan,
        candidate_pool: list[tuple[str, float]],
        narrowed_sections: dict[str, list[dict]],
        progress_callback: Callable[[SearchProgressUpdate], None] | None = None,
    ) -> dict[str, dict[str, list[EvidenceChunk]]]:
        unique_queries = [query_plan.global_query]
        for bucket in query_plan.evidence_buckets:
            unique_queries.extend(bucket.queries)
        query_cache = self._build_query_cache(runtime.chunk_bm25, runtime.chunk_encoder, runtime.chunk_vectors, runtime.chunk_ids, unique_queries)

        evidence_packs: dict[str, dict[str, list[EvidenceChunk]]] = {}
        selection_states: dict[tuple[str, str], _BucketSelectionState | None] = {}
        reranker_pairs: list[tuple[str, str]] = []
        total_candidates = len(candidate_pool)
        for paper_id, _ in candidate_pool:
            section_rows = narrowed_sections.get(paper_id, [])
            selected_section_ids = {row["section_id"] for row in section_rows}
            chunk_rows = []
            for section_id in selected_section_ids:
                chunk_rows.extend(runtime.chunks_by_section.get(section_id, []))
            if not chunk_rows:
                for bucket in query_plan.evidence_buckets:
                    selection_states[(paper_id, bucket.bucket_id)] = None
                continue

            chunk_indices = np.array([row["index"] for row in chunk_rows], dtype=int)
            for bucket in query_plan.evidence_buckets:
                selection_state = self._prepare_bucket_chunk_selection(
                    runtime=runtime,
                    bucket=bucket,
                    paper_id=paper_id,
                    chunk_rows=chunk_rows,
                    chunk_indices=chunk_indices,
                    query_cache=query_cache,
                )
                selection_states[(paper_id, bucket.bucket_id)] = selection_state
                if selection_state is None:
                    continue
                selection_state.reranker_offset = len(reranker_pairs)
                selection_state.reranker_count = len(selection_state.preselected_rows)
                reranker_pairs.extend(
                    (
                        selection_state.reranker_query,
                        f"{selection_state.chunk_rows[row_index]['heading']} {selection_state.chunk_rows[row_index]['text']}",
                    )
                    for row_index in selection_state.preselected_rows
                )

        reranker_scores = np.zeros(0, dtype=np.float32)
        if reranker_pairs:
            self._emit_progress(
                progress_callback,
                "evidence_assembly",
                "Scoring evidence candidates across all narrowed papers.",
                stage_progress=0.45,
            )
            reranker_scores = runtime.local_reranker.score_pairs(reranker_pairs)

        for processed_count, (paper_id, _) in enumerate(candidate_pool, start=1):
            bucket_chunks: dict[str, list[EvidenceChunk]] = {}
            for bucket in query_plan.evidence_buckets:
                selection_state = selection_states.get((paper_id, bucket.bucket_id))
                if selection_state is None:
                    bucket_chunks[bucket.bucket_id] = []
                    continue
                start = selection_state.reranker_offset
                end = start + selection_state.reranker_count
                bucket_chunks[bucket.bucket_id] = self._finalize_bucket_chunks(
                    selection_state,
                    reranker_scores[start:end],
                )
            evidence_packs[paper_id] = bucket_chunks
            self._emit_progress(
                progress_callback,
                "evidence_assembly",
                f"Assembled evidence for {processed_count}/{total_candidates} candidate papers.",
                completed_items=processed_count,
                total_items=total_candidates,
            )
        return evidence_packs

    def _shortlist_for_verifier(
        self,
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

        section_values: list[float] = []
        evidence_values: list[float] = []
        total_buckets = max(1, len(query_plan.evidence_buckets))

        for paper_id in ordered_ids:
            section_scores = [float(row.get("score", 0.0)) for row in narrowed_sections.get(paper_id, [])]
            section_values.append(_aggregate_top_local_scores(section_scores))

            bucket_best_scores: list[float] = []
            nonempty_bucket_count = 0
            for bucket in query_plan.evidence_buckets:
                chunks = evidence_packs.get(paper_id, {}).get(bucket.bucket_id, [])
                if not chunks:
                    continue
                nonempty_bucket_count += 1
                bucket_best_scores.append(max(float(chunk.score) for chunk in chunks))

            evidence_strength = _aggregate_top_local_scores(bucket_best_scores)
            evidence_coverage = nonempty_bucket_count / total_buckets
            evidence_values.append(0.7 * evidence_strength + 0.3 * evidence_coverage)

        coarse_norm = _normalize_scores(coarse_values)
        section_norm = _normalize_scores(np.array(section_values, dtype=float))
        evidence_norm = _normalize_scores(np.array(evidence_values, dtype=float))

        scored_rows: list[dict[str, object]] = []
        for index, paper_id in enumerate(ordered_ids):
            pre_verifier_score = (
                0.35 * float(coarse_norm[index])
                + 0.25 * float(section_norm[index])
                + 0.40 * float(evidence_norm[index])
            )
            scored_rows.append(
                {
                    "paper_id": paper_id,
                    "coarse_score": float(coarse_values[index]),
                    "section_score": float(section_values[index]),
                    "evidence_score": float(evidence_values[index]),
                    "pre_verifier_score": float(pre_verifier_score),
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

    def _verify_candidates(
        self,
        runtime: _Runtime,
        query_plan: QueryPlan,
        candidate_pool: list[tuple[str, float]],
        coarse_scores: dict[str, float],
        evidence_packs: dict[str, dict[str, list[EvidenceChunk]]],
        narrowed_sections: dict[str, list[dict]],
        top_k: int,
        progress_callback: Callable[[SearchProgressUpdate], None] | None = None,
    ) -> tuple[dict[str, list[PaperResult]], TokenUsage]:
        if not self.settings.openai_enabled or not self.settings.openai_api_key:
            raise RuntimeError("Final verifier requires an enabled API model.")
        require_openai_model(self.settings)

        usage = TokenUsage()
        grouped_results: dict[str, list[PaperResult]] = {"satisfied": [], "partial": [], "rejected": []}
        total_candidates = len(candidate_pool)
        max_workers = min(max(1, self.settings.verifier_max_workers), max(1, len(candidate_pool)))
        client_limits = httpx.Limits(max_connections=max_workers, max_keepalive_connections=max_workers)
        with httpx.Client(timeout=self.settings.request_timeout, limits=client_limits) as client:
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="paper-verifier") as executor:
                future_map = {
                    executor.submit(
                        self._verify_with_openai,
                        query_plan,
                        runtime.paper_lookup[paper_id],
                        evidence_packs.get(paper_id, {}),
                        client,
                    ): (paper_id, coarse_score)
                    for paper_id, coarse_score in candidate_pool
                }
                for completed_count, future in enumerate(as_completed(future_map), start=1):
                    paper_id, coarse_score = future_map[future]
                    paper = runtime.paper_lookup[paper_id]
                    evidence_pack = evidence_packs.get(paper_id, {})
                    verifier_payload, one_usage = future.result()
                    usage.prompt_tokens += one_usage.prompt_tokens
                    usage.completion_tokens += one_usage.completion_tokens
                    usage.total_tokens += one_usage.total_tokens
                    usage.cost_estimate_usd = _sum_costs(usage.cost_estimate_usd, one_usage.cost_estimate_usd)

                    verdict = str(verifier_payload["verdict"]).strip().lower()
                    if verdict not in grouped_results:
                        raise RuntimeError(f"Verifier returned unsupported verdict: {verdict}")
                    confidence = float(verifier_payload["confidence"])
                    verifier_score = max(0.0, min(confidence, 1.0))
                    final_score = verifier_score + 0.05 * float(coarse_score)
                    main_image_url = select_main_image_url(
                        self.settings,
                        paper.paper_id,
                        runtime.objects_by_paper.get(paper.paper_id, []),
                    )
                    structured_summary, enriched_metadata = load_cached_paper_enrichment(self.settings, paper.paper_id)
                    authors, affiliations, authors_structured = load_cached_paper_authorship(self.settings, paper)
                    rationale = str(verifier_payload["rationale"])
                    matched_sections = [row["section_title"] for row in narrowed_sections.get(paper_id, [])]

                    grouped_results[verdict].append(
                        PaperResult(
                            paper_id=paper.paper_id,
                            title=paper.title,
                            score=float(final_score),
                            coarse_score=float(coarse_score),
                            verifier_score=float(verifier_score),
                            venue=paper.venue,
                            year=paper.year,
                            track=paper.track,
                            verdict=verdict,
                            entity_role=verifier_payload.get("entity_role"),
                            satisfied_constraints=[str(item) for item in verifier_payload.get("satisfied_constraints", [])],
                            missing_constraints=[str(item) for item in verifier_payload.get("missing_constraints", [])],
                            confidence=float(verifier_score),
                            rationale=rationale,
                            rationale_structured=structure_rationale_text(rationale),
                            matched_sections=matched_sections,
                            matched_sections_summary=build_matched_sections_summary(narrowed_sections.get(paper_id, [])),
                            evidence_chunks=evidence_pack,
                            main_image_url=main_image_url,
                            abstract=paper.abstract or None,
                            authors=authors,
                            affiliations=affiliations,
                            authors_structured=authors_structured,
                            structured_summary=structured_summary,
                            enriched_metadata=enriched_metadata,
                        )
                    )
                    self._emit_progress(
                        progress_callback,
                        "final_verifier",
                        f"Verified {completed_count}/{total_candidates} candidate papers.",
                        completed_items=completed_count,
                        total_items=total_candidates,
                    )

        for verdict, items in grouped_results.items():
            items.sort(key=lambda item: item.score, reverse=True)
            grouped_results[verdict] = items[:top_k]
        return grouped_results, usage

    def _rank_paper_source(
        self,
        runtime: _Runtime,
        query_plan: QueryPlan,
        allowed_paper_ids: set[str],
        limit: int,
        *,
        dense: bool,
    ) -> tuple[list[str], dict[str, set[str]]]:
        rankings: dict[str, list[str]] = {}
        weights: dict[str, float] = {}
        aspect_hits: dict[str, set[str]] = defaultdict(set)

        rankings["global"] = self._filter_ranked_ids(
            self._rank_dense(runtime.paper_encoder, runtime.paper_vectors, runtime.paper_ids, query_plan.global_query, limit=limit)
            if dense
            else self._rank_sparse(runtime.paper_bm25, runtime.paper_ids, query_plan.global_query, limit=limit),
            allowed_paper_ids,
        )
        weights["global"] = 1.0

        for aspect in query_plan.aspect_queries:
            rankings[aspect.aspect_id] = self._filter_ranked_ids(
                self._rank_dense(runtime.paper_encoder, runtime.paper_vectors, runtime.paper_ids, aspect.query, limit=limit)
                if dense
                else self._rank_sparse(runtime.paper_bm25, runtime.paper_ids, aspect.query, limit=limit),
                allowed_paper_ids,
            )
            weights[aspect.aspect_id] = aspect.weight
            for paper_id in rankings[aspect.aspect_id]:
                aspect_hits[paper_id].add(aspect.aspect_id)

        fused = weighted_rrf(rankings, weights)
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [paper_id for paper_id, _ in ranked], aspect_hits

    def _rank_section_aggregated_source(
        self,
        runtime: _Runtime,
        query_plan: QueryPlan,
        allowed_paper_ids: set[str],
        limit: int,
    ) -> tuple[list[str], dict[str, set[str]]]:
        query_scores: list[tuple[str, float, str | None]] = [(query_plan.global_query, 1.0, None)]
        query_scores.extend((aspect.query, aspect.weight, aspect.aspect_id) for aspect in query_plan.aspect_queries)
        section_limit = max(limit * 12, 300)

        paper_scores: dict[str, float] = defaultdict(float)
        aspect_hits: dict[str, set[str]] = defaultdict(set)

        for query, weight, aspect_id in query_scores:
            sparse_scores = self._score_sparse(runtime.section_bm25, runtime.section_ids, query, limit=section_limit)
            dense_scores = self._score_dense(runtime.chunk_encoder, runtime.section_vectors, runtime.section_ids, query, limit=section_limit)
            merged: dict[str, dict[str, float]] = defaultdict(lambda: {"sparse": 0.0, "dense": 0.0})
            for section_id, score in sparse_scores:
                merged[section_id]["sparse"] = score
            for section_id, score in dense_scores:
                merged[section_id]["dense"] = score

            per_paper_section_scores: dict[str, list[float]] = defaultdict(list)
            for section_id, parts in merged.items():
                row = runtime.section_lookup.get(section_id)
                if row is None or row["paper_id"] not in allowed_paper_ids:
                    continue
                combined_score = 0.55 * parts["sparse"] + 0.45 * parts["dense"]
                if combined_score <= 0:
                    continue
                per_paper_section_scores[row["paper_id"]].append(float(combined_score))

            for paper_id, values in per_paper_section_scores.items():
                paper_scores[paper_id] += weight * _aggregate_top_local_scores(values)
                if aspect_id is not None:
                    aspect_hits[paper_id].add(aspect_id)

        ranked = sorted(paper_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [paper_id for paper_id, _ in ranked], aspect_hits

    def _rank_literal_entity_source(
        self,
        runtime: _Runtime,
        query_plan: QueryPlan,
        allowed_paper_ids: set[str],
        limit: int,
    ) -> list[str]:
        if not query_plan.entity_terms:
            return []
        normalized_terms = [_normalized_match_text(term) for term in query_plan.entity_terms if _normalized_match_text(term)]
        paper_scores: dict[str, float] = {}
        for paper_id in allowed_paper_ids:
            chunk_rows = runtime.chunks_by_paper.get(paper_id, [])
            unique_hits = set()
            matching_chunks = 0
            for row in chunk_rows:
                normalized_text = _normalized_match_text(f"{row['heading']} {row['text']}")
                row_hit = False
                for term in normalized_terms:
                    if _contains_normalized_phrase(normalized_text, term):
                        unique_hits.add(term)
                        row_hit = True
                if row_hit:
                    matching_chunks += 1
            if unique_hits:
                paper_scores[paper_id] = len(unique_hits) + 0.05 * matching_chunks
        ranked = sorted(paper_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [paper_id for paper_id, _ in ranked]

    def _rank_exact_phrase_source(
        self,
        runtime: _Runtime,
        query_plan: QueryPlan,
        allowed_paper_ids: set[str],
        limit: int,
    ) -> list[str]:
        if not query_plan.exact_phrases:
            return []
        normalized_phrases = [_normalized_match_text(phrase) for phrase in query_plan.exact_phrases if _normalized_match_text(phrase)]
        paper_scores: dict[str, float] = {}
        for paper_id in allowed_paper_ids:
            chunk_rows = runtime.chunks_by_paper.get(paper_id, [])
            unique_hits = set()
            matching_chunks = 0
            for row in chunk_rows:
                normalized_text = _normalized_match_text(f"{row['heading']} {row['text']}")
                row_hit = False
                for phrase in normalized_phrases:
                    if _contains_normalized_phrase(normalized_text, phrase):
                        unique_hits.add(phrase)
                        row_hit = True
                if row_hit:
                    matching_chunks += 1
            if unique_hits:
                paper_scores[paper_id] = len(unique_hits) + 0.05 * matching_chunks
        ranked = sorted(paper_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [paper_id for paper_id, _ in ranked]

    def _build_query_cache(
        self,
        bm25: BM25Okapi,
        encoder: SentenceTransformerEncoder,
        vectors: np.ndarray,
        item_ids: list[str],
        queries: list[str],
    ) -> dict[str, dict[str, np.ndarray]]:
        cache: dict[str, dict[str, np.ndarray]] = {}
        unique_queries = []
        seen = set()
        for query in queries:
            normalized = query.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_queries.append(normalized)

        for query in unique_queries:
            tokens = tokenize(query)
            sparse_scores = np.asarray(bm25.get_scores(tokens), dtype=float) if tokens else np.zeros(len(item_ids), dtype=float)
            dense_vector = encoder.encode([query])[0]
            cache[query] = {"sparse": sparse_scores, "dense_vector": dense_vector}
        return cache

    def _prepare_bucket_chunk_selection(
        self,
        runtime: _Runtime,
        bucket: EvidenceBucket,
        paper_id: str,
        chunk_rows: list[dict],
        chunk_indices: np.ndarray,
        query_cache: dict[str, dict[str, np.ndarray]],
    ) -> _BucketSelectionState | None:
        if not chunk_rows:
            return None

        best_sparse = np.zeros(len(chunk_rows), dtype=float)
        best_dense = np.zeros(len(chunk_rows), dtype=float)
        best_query = [""] * len(chunk_rows)
        best_base = np.zeros(len(chunk_rows), dtype=float)

        valid_queries = [query for query in bucket.queries if query.strip() and query in query_cache]
        if not valid_queries:
            return None

        for query in valid_queries:
            cached = query_cache[query]
            local_sparse = _normalize_scores(cached["sparse"][chunk_indices])
            local_dense = _normalize_scores(cosine_similarity_matrix(cached["dense_vector"], runtime.chunk_vectors[chunk_indices]))
            base_scores = self.settings.evidence_sparse_weight * local_sparse + self.settings.evidence_dense_weight * local_dense
            for row_index, row in enumerate(chunk_rows):
                typed_bonus = 0.0
                if row["chunk_type"] in {"table_chunk", "figure_chunk"} and any(
                    token in bucket.description.lower() for token in ("result", "evaluation", "metric", "comparison")
                ):
                    typed_bonus = 0.04
                score = base_scores[row_index] + typed_bonus
                if score <= best_base[row_index]:
                    continue
                best_base[row_index] = score
                best_sparse[row_index] = local_sparse[row_index]
                best_dense[row_index] = local_dense[row_index]
                best_query[row_index] = query

        reranker_limit = min(len(chunk_rows), max(bucket.target_chunks * 4, self.settings.evidence_reranker_candidate_chunks))
        preselected = [
            item
            for item in sorted(enumerate(best_base.tolist()), key=lambda item: item[1], reverse=True)
            if item[1] > 0
        ][:reranker_limit]
        if not preselected:
            return None

        return _BucketSelectionState(
            paper_id=paper_id,
            bucket=bucket,
            chunk_rows=chunk_rows,
            preselected_rows=[row_index for row_index, _ in preselected],
            best_sparse=best_sparse,
            best_dense=best_dense,
            best_query=best_query,
            reranker_query=f"{bucket.description} :: {' ; '.join(valid_queries)}",
            reranker_offset=0,
            reranker_count=0,
        )

    def _finalize_bucket_chunks(
        self,
        selection_state: _BucketSelectionState,
        reranker_scores: np.ndarray,
    ) -> list[EvidenceChunk]:
        if not selection_state.preselected_rows:
            return []

        normalized_reranker_scores = _normalize_scores(np.asarray(reranker_scores, dtype=float))
        selected_chunks: list[EvidenceChunk] = []
        for row_index, reranker_score in sorted(
            zip(selection_state.preselected_rows, normalized_reranker_scores.tolist(), strict=False),
            key=lambda item: (
                self.settings.evidence_sparse_weight * selection_state.best_sparse[item[0]]
                + self.settings.evidence_dense_weight * selection_state.best_dense[item[0]]
                + self.settings.evidence_reranker_weight * item[1]
            ),
            reverse=True,
        ):
            row = selection_state.chunk_rows[row_index]
            final_score = (
                self.settings.evidence_sparse_weight * selection_state.best_sparse[row_index]
                + self.settings.evidence_dense_weight * selection_state.best_dense[row_index]
                + self.settings.evidence_reranker_weight * reranker_score
            )
            selected_chunks.append(
                EvidenceChunk(
                    paper_id=selection_state.paper_id,
                    bucket_id=selection_state.bucket.bucket_id,
                    chunk_id=row["chunk_id"],
                    chunk_type=row["chunk_type"],
                    score=float(final_score),
                    source_query=selection_state.best_query[row_index],
                    heading=row["heading"],
                    section_path=row["section_path"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    text=truncate_text(row["text"], limit=self.settings.evidence_chunk_text_limit),
                )
            )
            if len(selected_chunks) >= selection_state.bucket.target_chunks:
                break
        return selected_chunks

    def _verify_with_openai(
        self,
        query_plan: QueryPlan,
        paper: PaperRecord,
        evidence_pack: dict[str, list[EvidenceChunk]],
        client: httpx.Client,
    ) -> tuple[dict, TokenUsage]:
        model = require_openai_model(self.settings)

        messages = [
            {
                "role": "system",
                "content": (
                    "You verify whether a scientific paper satisfies a detailed scholarly retrieval query. "
                    "Return strict JSON with keys verdict, entity_role, satisfied_constraints, missing_constraints, confidence, rationale. "
                    "verdict must be one of: satisfied, partial, rejected. "
                    "entity_role must be one of: dataset_or_benchmark, method_or_system, task_or_setting, ambiguous_or_other. "
                    "confidence must be a number between 0 and 1. "
                    "Use only the supplied metadata and evidence chunks. "
                    "The paper metadata already gives the venue and year scope; do not reject due to missing scope if metadata matches. "
                    "For ambiguous entity strings, distinguish whether the entity is the intended benchmark/dataset, or actually the paper's own method/system name."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query_plan.user_query,
                        "global_query": query_plan.global_query,
                        "scope_constraints": query_plan.scope_constraints.model_dump(),
                        "entity_terms": query_plan.entity_terms,
                        "exact_phrases": query_plan.exact_phrases,
                        "verifier_rubric": query_plan.verifier_rubric.model_dump(),
                        "paper": {
                            "paper_id": paper.paper_id,
                            "title": paper.title,
                            "venue": paper.venue,
                            "year": paper.year,
                            "track": paper.track,
                            "abstract": truncate_text(paper.abstract, 600),
                            "intro_summary": truncate_text(paper.intro_summary, 500),
                        },
                        "evidence_buckets": self._verifier_evidence_payload(query_plan, evidence_pack),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        usage = TokenUsage()
        last_content: dict | None = None
        last_error: str | None = None

        for attempt in range(2):
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            response = client.post(f"{self.settings.openai_base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            raw = response.json()
            attempt_usage = _usage_from_openai_payload(self.settings, raw)
            usage.prompt_tokens += attempt_usage.prompt_tokens
            usage.completion_tokens += attempt_usage.completion_tokens
            usage.total_tokens += attempt_usage.total_tokens
            usage.cost_estimate_usd = _sum_costs(usage.cost_estimate_usd, attempt_usage.cost_estimate_usd)

            content = json.loads(raw["choices"][0]["message"]["content"])
            last_content = content
            verdict = str(content.get("verdict", "")).strip().lower()
            entity_role = str(content.get("entity_role", "")).strip().lower()
            invalid_fields = []
            if verdict not in {"satisfied", "partial", "rejected"}:
                invalid_fields.append(
                    f"verdict={content.get('verdict', '')!r} is invalid; allowed values are satisfied, partial, rejected"
                )
            if entity_role not in {"dataset_or_benchmark", "method_or_system", "task_or_setting", "ambiguous_or_other"}:
                invalid_fields.append(
                    "entity_role="
                    f"{content.get('entity_role', '')!r} is invalid; allowed values are "
                    "dataset_or_benchmark, method_or_system, task_or_setting, ambiguous_or_other"
                )
            if not invalid_fields:
                return {
                    "verdict": verdict,
                    "entity_role": entity_role,
                    "satisfied_constraints": content.get("satisfied_constraints", []),
                    "missing_constraints": content.get("missing_constraints", []),
                    "confidence": float(content.get("confidence", 0.0)),
                    "rationale": str(content.get("rationale", "")).strip(),
                }, usage

            last_error = "; ".join(invalid_fields)
            messages.extend(
                [
                    {"role": "assistant", "content": raw["choices"][0]["message"]["content"]},
                    {
                        "role": "user",
                        "content": (
                            "Your previous JSON did not satisfy the required schema. "
                            f"Problems: {last_error}. "
                            "Return corrected JSON only. Keep your semantic judgment the same whenever possible, "
                            "but use only allowed enum labels."
                        ),
                    },
                ]
            )

        raise RuntimeError(
            "Final verifier returned invalid structured output after repair attempt: "
            f"{last_error or last_content}"
        )

    def _verifier_evidence_payload(
        self,
        query_plan: QueryPlan,
        evidence_pack: dict[str, list[EvidenceChunk]],
    ) -> dict[str, dict[str, object]]:
        payload: dict[str, dict[str, object]] = {}
        for bucket in query_plan.evidence_buckets:
            payload[bucket.bucket_id] = {
                "description": bucket.description,
                "queries": bucket.queries,
                "target_chunks": bucket.target_chunks,
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "chunk_type": chunk.chunk_type,
                        "heading": chunk.heading,
                        "section_path": chunk.section_path,
                        "pages": [chunk.page_start, chunk.page_end],
                        "source_query": chunk.source_query,
                        "text": chunk.text,
                    }
                    for chunk in evidence_pack.get(bucket.bucket_id, [])
                ],
            }
        return payload

    def _allowed_paper_ids(self, runtime: _Runtime, effective_scope: list[str]) -> set[str]:
        allowed: set[str] = set()
        for corpus_key in effective_scope:
            allowed.update(runtime.paper_ids_by_corpus.get(corpus_key, set()))
        return allowed

    def _resolve_workspace_scope(self, runtime: _Runtime, workspace_scope: list[str] | None) -> list[str]:
        available_corpora = set(runtime.paper_ids_by_corpus)
        if workspace_scope is None:
            return sorted(available_corpora)
        return sorted({corpus for corpus in workspace_scope if corpus in available_corpora})

    def _resolve_effective_scope(
        self,
        runtime: _Runtime,
        workspace_scope: list[str],
        query_scope: ScopeConstraints,
    ) -> list[str]:
        if not (query_scope.venues or query_scope.years or query_scope.tracks):
            return workspace_scope
        query_scope_corpora = {
            corpus_key
            for corpus_key in runtime.paper_ids_by_corpus
            if _matches_corpus_constraints(corpus_key, query_scope)
        }
        return sorted(query_scope_corpora.intersection(workspace_scope))

    @staticmethod
    def _filter_ranked_ids(item_ids: list[str], allowed_paper_ids: set[str]) -> list[str]:
        return [item_id for item_id in item_ids if item_id in allowed_paper_ids]

    def _rank_sparse(self, bm25: BM25Okapi, item_ids: list[str], query: str, limit: int) -> list[str]:
        return [item_id for item_id, _ in self._score_sparse(bm25, item_ids, query, limit)]

    def _score_sparse(self, bm25: BM25Okapi, item_ids: list[str], query: str, limit: int) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        ranked = sorted(zip(item_ids, scores), key=lambda item: item[1], reverse=True)
        ranked = [(item_id, float(score)) for item_id, score in ranked[:limit] if score > 0]
        if not ranked:
            return []
        normalized = _normalize_scores(np.array([score for _, score in ranked], dtype=float))
        return [(item_id, float(score)) for (item_id, _), score in zip(ranked, normalized.tolist(), strict=False)]

    def _rank_dense(
        self,
        encoder: SentenceTransformerEncoder,
        vectors: np.ndarray,
        item_ids: list[str],
        query: str,
        limit: int,
    ) -> list[str]:
        return [item_id for item_id, _ in self._score_dense(encoder, vectors, item_ids, query, limit)]

    def _score_dense(
        self,
        encoder: SentenceTransformerEncoder,
        vectors: np.ndarray,
        item_ids: list[str],
        query: str,
        limit: int,
    ) -> list[tuple[str, float]]:
        if vectors.size == 0:
            return []
        query_vector = encoder.encode([query])[0]
        scores = cosine_similarity_matrix(query_vector, vectors)
        ranked = sorted(zip(item_ids, scores.tolist()), key=lambda item: item[1], reverse=True)
        ranked = [(item_id, float(score)) for item_id, score in ranked[:limit]]
        if not ranked:
            return []
        normalized = _normalize_scores(np.array([score for _, score in ranked], dtype=float))
        return [(item_id, float(score)) for (item_id, _), score in zip(ranked, normalized.tolist(), strict=False)]


def _paper_corpus_key(paper: PaperRecord) -> str:
    track = (paper.track or "unknown").strip().lower()
    return f"{paper.venue.lower()}/{paper.year}/{track}"


def _matches_corpus_constraints(corpus_key: str, constraints: ScopeConstraints) -> bool:
    venue, year_text, track = corpus_key.split("/", 2)
    if constraints.venues and venue not in constraints.venues:
        return False
    if constraints.years and int(year_text) not in constraints.years:
        return False
    if constraints.tracks and track not in constraints.tracks:
        return False
    return True


def _aggregate_top_local_scores(scores: list[float]) -> float:
    ordered = sorted(scores, reverse=True)[:3]
    if not ordered:
        return 0.0
    weights = [0.7, 0.2, 0.1]
    return sum(weight * score for weight, score in zip(weights, ordered, strict=False))


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=float)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-9:
        if maximum > 0:
            return np.ones_like(values, dtype=float)
        return np.zeros_like(values, dtype=float)
    return (values - minimum) / (maximum - minimum)


def _make_progress_update(
    stage: str,
    message: str,
    *,
    stage_progress: float | None = None,
    completed_items: int | None = None,
    total_items: int | None = None,
) -> SearchProgressUpdate:
    stage_total = len(_SEARCH_PROGRESS_STAGES)
    if stage == "completed":
        return SearchProgressUpdate(
            stage=stage,
            message=message,
            stage_index=stage_total,
            stage_total=stage_total,
            stage_progress=1.0,
            overall_progress=1.0,
            completed_items=completed_items,
            total_items=total_items,
        )

    stage_index = _SEARCH_PROGRESS_STAGE_INDEX.get(stage, 0)
    if stage_progress is None:
        if total_items is not None and total_items > 0 and completed_items is not None:
            stage_progress = completed_items / total_items
        else:
            stage_progress = 0.0
    clamped_stage_progress = max(0.0, min(float(stage_progress), 1.0))
    overall_progress = 0.0
    if stage_index > 0 and stage_total > 0:
        overall_progress = ((stage_index - 1) + clamped_stage_progress) / stage_total

    return SearchProgressUpdate(
        stage=stage,
        message=message,
        stage_index=stage_index,
        stage_total=stage_total,
        stage_progress=clamped_stage_progress,
        overall_progress=max(0.0, min(float(overall_progress), 1.0)),
        completed_items=completed_items,
        total_items=total_items,
    )


def _normalized_match_text(text: str) -> str:
    return " ".join(tokenize(text.replace("-", " ")))


def _contains_normalized_phrase(haystack: str, needle: str) -> bool:
    if not haystack or not needle:
        return False
    return f" {needle} " in f" {haystack} "


def _usage_from_openai_payload(settings: Settings, payload: dict) -> TokenUsage:
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
    cost = None
    if settings.input_price_per_1m is not None and settings.output_price_per_1m is not None:
        cost = (
            prompt_tokens / 1_000_000 * settings.input_price_per_1m
            + completion_tokens / 1_000_000 * settings.output_price_per_1m
        )
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_estimate_usd=cost,
    )


def _sum_costs(left: float | None, right: float | None) -> float | None:
    if left is None and right is None:
        return None
    return float(left or 0.0) + float(right or 0.0)


def _assert_index_alignment(name: str, meta: dict, vector_ids: list[str], vectors: np.ndarray) -> None:
    meta_ids = [str(item) for item in meta.get("ids", [])]
    if meta_ids != vector_ids:
        raise RuntimeError(f"{name} index metadata ids do not align with vector ids.")
    if vectors.shape[0] != len(vector_ids):
        raise RuntimeError(f"{name} vector row count does not match index ids.")


def _assert_encoder_compatibility(
    name: str,
    meta: dict,
    vectors: np.ndarray,
    encoder: SentenceTransformerEncoder,
) -> None:
    backend = str(meta.get("encoder_backend", "")).strip()
    model = str(meta.get("encoder_model", "")).strip()
    vector_dim = int(meta.get("vector_dim", 0) or 0)
    if not backend or not model or vector_dim <= 0:
        raise RuntimeError(f"{name} index metadata is missing encoder provenance. Rebuild indexes.")
    if backend != encoder.backend_name:
        raise RuntimeError(
            f"{name} index was built with backend {backend!r}, but runtime requires {encoder.backend_name!r}."
        )
    if model != encoder.model_name:
        raise RuntimeError(f"{name} index was built with model {model!r}, but runtime requires {encoder.model_name!r}.")
    if vectors.ndim != 2 or vectors.shape[1] != vector_dim:
        raise RuntimeError(
            f"{name} index vector_dim={vector_dim} does not match stored matrix shape {tuple(vectors.shape)}."
        )
