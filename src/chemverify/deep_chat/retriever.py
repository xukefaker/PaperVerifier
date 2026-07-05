from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import numpy as np
from rank_bm25 import BM25Okapi

from ..config import Settings
from ..encoders import SentenceTransformerEncoder
from ..reranker import CrossEncoderReranker
from ..utils import cosine_similarity_matrix, tokenize, top_k_from_scores, truncate_text, weighted_rrf
from .evidence import build_evidence_search_text
from .models import (
    EvidencePack,
    EvidenceUnit,
    HypothesisCompetitionSignal,
    HypothesisEvidencePack,
    InterpretationHypothesis,
    RetrievedEvidence,
)


@dataclass(slots=True)
class EvidenceRuntime:
    evidence_lookup: dict[str, EvidenceUnit]
    rows_by_paper: dict[str, list[dict[str, object]]]
    evidence_ids: list[str]
    evidence_vectors: np.ndarray


@dataclass(slots=True)
class _BucketRuntime:
    rows: list[dict[str, object]]
    row_by_id: dict[str, dict[str, object]]
    bm25: BM25Okapi


@dataclass(slots=True)
class _BucketSelection:
    query: str
    preferred_types: set[str]
    top_k: int
    preselected: list[tuple[str, float]]
    row_by_id: dict[str, dict[str, object]]


class DeepChatRetriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._bucket_cache: dict[tuple[str, tuple[str, ...]], _BucketRuntime] = {}
        self._cache_lock = Lock()

    def retrieve_for_hypotheses(
        self,
        *,
        paper_id: str,
        hypotheses: list[InterpretationHypothesis],
        runtime: EvidenceRuntime,
        query_encoder: SentenceTransformerEncoder,
        reranker: CrossEncoderReranker,
    ) -> list[HypothesisEvidencePack]:
        rows = runtime.rows_by_paper.get(paper_id, [])
        if not rows:
            return []

        row_lookup = {str(row["evidence_id"]): row for row in rows}
        retrieval_queries = [self._build_query_text(hypothesis) for hypothesis in hypotheses]
        query_vectors = query_encoder.encode(retrieval_queries) if retrieval_queries else np.zeros((0, 0), dtype=np.float32)

        intermediate: list[dict[str, object]] = []
        rerank_jobs: list[_BucketSelection] = []
        for hypothesis, query_text, query_vector in zip(hypotheses, retrieval_queries, query_vectors, strict=False):
            query_tokens = tokenize(query_text)
            global_allowed, global_preferred, local_allowed, local_preferred = self._relation_type_preferences(
                relation=hypothesis.target_relation,
                preferred_types=set(hypothesis.preferred_evidence_types),
            )
            global_selection = self._prepare_bucket_selection(
                paper_id=paper_id,
                all_rows=rows,
                query=query_text,
                query_tokens=query_tokens,
                query_vector=query_vector,
                allowed_types=global_allowed,
                preferred_types=global_preferred,
                vectors=runtime.evidence_vectors,
                top_k=self.settings.deep_chat_global_evidence_k,
            )
            excluded_ids = {evidence_id for evidence_id, _ in global_selection.preselected}
            local_selection = self._prepare_bucket_selection(
                paper_id=paper_id,
                all_rows=rows,
                query=query_text,
                query_tokens=query_tokens,
                query_vector=query_vector,
                allowed_types=local_allowed,
                preferred_types=local_preferred,
                vectors=runtime.evidence_vectors,
                top_k=self.settings.deep_chat_local_evidence_k,
                excluded_ids=excluded_ids,
            )
            rerank_jobs.extend([global_selection, local_selection])
            intermediate.append(
                {
                    "hypothesis": hypothesis,
                    "query_text": query_text,
                    "global_selection": global_selection,
                    "local_selection": local_selection,
                }
            )

        rerank_scores = self._score_rerank_jobs(rerank_jobs, reranker)
        output: list[HypothesisEvidencePack] = []
        finalized: list[dict[str, object]] = []
        for item in intermediate:
            global_selection = item["global_selection"]
            local_selection = item["local_selection"]
            assert isinstance(global_selection, _BucketSelection)
            assert isinstance(local_selection, _BucketSelection)
            global_evidence, global_scores = self._finalize_bucket(
                selection=global_selection,
                reranker_scores=rerank_scores.get(id(global_selection), []),
            )
            local_evidence, local_scores = self._finalize_bucket(
                selection=local_selection,
                reranker_scores=rerank_scores.get(id(local_selection), []),
            )
            combined_scores = dict(global_scores)
            for evidence_id, score in local_scores.items():
                combined_scores[evidence_id] = max(combined_scores.get(evidence_id, 0.0), score)
            finalized.append(
                {
                    "hypothesis": item["hypothesis"],
                    "query_text": item["query_text"],
                    "global_evidence": global_evidence,
                    "local_evidence": local_evidence,
                    "combined_scores": combined_scores,
                }
            )

        competition_k = max(2, min(3, self.settings.deep_chat_local_evidence_k))
        for item in finalized:
            hypothesis = item["hypothesis"]
            assert isinstance(hypothesis, InterpretationHypothesis)
            query_text = str(item["query_text"])
            global_evidence = item["global_evidence"]
            local_evidence = item["local_evidence"]
            combined_scores = item["combined_scores"]
            assert isinstance(combined_scores, dict)
            support_ids = {evidence.evidence_id for evidence in [*global_evidence, *local_evidence]}
            competitor_scores = {
                other_item["hypothesis"].hypothesis_id: other_item["combined_scores"]
                for other_item in finalized
                if other_item["hypothesis"].hypothesis_id != hypothesis.hypothesis_id
            }
            discriminative_evidence = self._select_competition_evidence(
                row_lookup=row_lookup,
                query=query_text,
                current_scores=combined_scores,
                competitor_scores=competitor_scores,
                mode="discriminative",
                top_k=competition_k,
                excluded_ids=support_ids,
            )
            conflicting_evidence = self._select_competition_evidence(
                row_lookup=row_lookup,
                query=query_text,
                current_scores=combined_scores,
                competitor_scores=competitor_scores,
                mode="conflicting",
                top_k=competition_k,
                excluded_ids=support_ids.union({evidence.evidence_id for evidence in discriminative_evidence}),
            )
            competition_signal = self._build_competition_signal(
                support_scores=combined_scores,
                competitor_scores=competitor_scores,
                discriminative_evidence=discriminative_evidence,
                conflicting_evidence=conflicting_evidence,
            )
            output.append(
                HypothesisEvidencePack(
                    hypothesis_id=hypothesis.hypothesis_id,
                    normalized_question=hypothesis.normalized_question,
                    target_relation=hypothesis.target_relation,
                    target_objects=list(hypothesis.target_objects),
                    required_constraints=list(hypothesis.required_constraints),
                    disambiguation_focus=list(hypothesis.disambiguation_focus),
                    evidence_pack=EvidencePack(
                        global_evidence=global_evidence,
                        local_evidence=local_evidence,
                    ),
                    discriminative_evidence=discriminative_evidence,
                    conflicting_evidence=conflicting_evidence,
                    competition_signal=competition_signal,
                )
            )
        return output

    def _prepare_bucket_selection(
        self,
        *,
        paper_id: str,
        all_rows: list[dict[str, object]],
        query: str,
        query_tokens: list[str],
        query_vector: np.ndarray,
        allowed_types: set[str],
        preferred_types: set[str],
        vectors: np.ndarray,
        top_k: int,
        excluded_ids: set[str] | None = None,
    ) -> _BucketSelection:
        bucket = self._get_bucket_runtime(
            paper_id=paper_id,
            all_rows=all_rows,
            allowed_types=allowed_types,
        )
        if not bucket.rows:
            return _BucketSelection(
                query=query,
                preferred_types=preferred_types,
                top_k=top_k,
                preselected=[],
                row_by_id={},
            )

        filtered_rows = [
            row
            for row in bucket.rows
            if str(row["evidence_id"]) not in (excluded_ids or set())
        ]
        if not filtered_rows:
            return _BucketSelection(
                query=query,
                preferred_types=preferred_types,
                top_k=top_k,
                preselected=[],
                row_by_id={},
            )

        bm25 = bucket.bm25 if not excluded_ids and len(filtered_rows) == len(bucket.rows) else BM25Okapi(
            [list(row["search_tokens"]) for row in filtered_rows]
        )
        sparse_scores = np.array(bm25.get_scores(query_tokens), dtype=float)
        row_indices = [int(row["index"]) for row in filtered_rows]
        doc_vectors = vectors[row_indices]
        dense_scores = cosine_similarity_matrix(query_vector, doc_vectors)
        sparse_rank = [
            str(filtered_rows[index]["evidence_id"])
            for index in np.argsort(-sparse_scores)[: self.settings.deep_chat_retrieval_candidate_k]
        ]
        dense_rank = [
            str(filtered_rows[index]["evidence_id"])
            for index in np.argsort(-dense_scores)[: self.settings.deep_chat_retrieval_candidate_k]
        ]
        fused = weighted_rrf(
            {"sparse": sparse_rank, "dense": dense_rank},
            {"sparse": 0.5, "dense": 0.5},
        )
        preselected = top_k_from_scores(
            fused,
            min(self.settings.deep_chat_retrieval_candidate_k, len(filtered_rows)),
        )
        return _BucketSelection(
            query=query,
            preferred_types=preferred_types,
            top_k=top_k,
            preselected=preselected,
            row_by_id={str(row["evidence_id"]): row for row in filtered_rows},
        )

    def _score_rerank_jobs(
        self,
        jobs: list[_BucketSelection],
        reranker: CrossEncoderReranker,
    ) -> dict[int, list[float]]:
        pairs: list[tuple[str, str]] = []
        counts: list[int] = []
        for job in jobs:
            job_pairs = [
                (
                    job.query,
                    f"{job.row_by_id[evidence_id]['heading']}\n{job.row_by_id[evidence_id]['search_text']}",
                )
                for evidence_id, _ in job.preselected
            ]
            pairs.extend(job_pairs)
            counts.append(len(job_pairs))
        if not pairs:
            return {id(job): [] for job in jobs}
        scores = reranker.score_pairs(pairs).tolist()
        result: dict[int, list[float]] = {}
        offset = 0
        for job, count in zip(jobs, counts, strict=False):
            result[id(job)] = scores[offset : offset + count]
            offset += count
        return result

    def _finalize_bucket(
        self,
        *,
        selection: _BucketSelection,
        reranker_scores: list[float],
    ) -> tuple[list[RetrievedEvidence], dict[str, float]]:
        if not selection.preselected:
            return [], {}
        scored_items = sorted(
            zip(selection.preselected, reranker_scores, strict=False),
            key=lambda item: self._final_score(
                fused_score=item[0][1],
                reranker_score=float(item[1]),
                evidence_type=str(selection.row_by_id[item[0][0]]["evidence_type"]),
                preferred_types=selection.preferred_types,
            ),
            reverse=True,
        )
        results: list[RetrievedEvidence] = []
        final_scores_by_id: dict[str, float] = {}
        for (evidence_id, fused_score), reranker_score in scored_items:
            row = selection.row_by_id[evidence_id]
            final_score = self._final_score(
                fused_score=fused_score,
                reranker_score=float(reranker_score),
                evidence_type=str(row["evidence_type"]),
                preferred_types=selection.preferred_types,
            )
            final_scores_by_id[evidence_id] = final_score
            if len(results) < selection.top_k:
                results.append(
                    self._row_to_retrieved_evidence(
                        row=row,
                        query=selection.query,
                        score=final_score,
                    )
                )
        return results, final_scores_by_id

    def _get_bucket_runtime(
        self,
        *,
        paper_id: str,
        all_rows: list[dict[str, object]],
        allowed_types: set[str],
    ) -> _BucketRuntime:
        cache_key = (paper_id, tuple(sorted(allowed_types)))
        cached = self._bucket_cache.get(cache_key)
        if cached is not None:
            return cached
        with self._cache_lock:
            cached = self._bucket_cache.get(cache_key)
            if cached is not None:
                return cached
            rows = [
                row
                for row in all_rows
                if str(row["evidence_type"]) in allowed_types
            ]
            bucket = _BucketRuntime(
                rows=rows,
                row_by_id={str(row["evidence_id"]): row for row in rows},
                bm25=BM25Okapi([list(row["search_tokens"]) for row in rows]) if rows else BM25Okapi([[]]),
            )
            self._bucket_cache[cache_key] = bucket
            return bucket

    @staticmethod
    def _final_score(
        *,
        fused_score: float,
        reranker_score: float,
        evidence_type: str,
        preferred_types: set[str],
    ) -> float:
        type_bonus = 0.08 if evidence_type in preferred_types else 0.0
        return float(fused_score + 0.25 * reranker_score + type_bonus)

    def _select_competition_evidence(
        self,
        *,
        row_lookup: dict[str, dict[str, object]],
        query: str,
        current_scores: dict[str, float],
        competitor_scores: dict[str, dict[str, float]],
        mode: str,
        top_k: int,
        excluded_ids: set[str],
    ) -> list[RetrievedEvidence]:
        if not competitor_scores:
            return []
        candidates: list[tuple[str, float, str | None, float]] = []
        for evidence_id, row in row_lookup.items():
            if evidence_id in excluded_ids:
                continue
            current_score = float(current_scores.get(evidence_id, 0.0))
            strongest_competitor_id: str | None = None
            strongest_competitor_score = 0.0
            for competitor_id, competitor_score_map in competitor_scores.items():
                score = float(competitor_score_map.get(evidence_id, 0.0))
                if score > strongest_competitor_score:
                    strongest_competitor_score = score
                    strongest_competitor_id = competitor_id
            if mode == "discriminative":
                margin = current_score - strongest_competitor_score
                if current_score <= 0.0 or margin <= 0.02:
                    continue
                score = margin
            else:
                margin = strongest_competitor_score - current_score
                if strongest_competitor_score <= 0.0 or margin <= 0.02:
                    continue
                score = margin
            candidates.append((evidence_id, score, strongest_competitor_id, strongest_competitor_score))

        results: list[RetrievedEvidence] = []
        for evidence_id, score, competitor_id, competitor_score in sorted(candidates, key=lambda item: item[1], reverse=True)[:top_k]:
            row = dict(row_lookup[evidence_id])
            metadata = dict(row["metadata"])
            metadata.update(
                {
                    "retrieval_role": mode,
                    "competition_against": competitor_id,
                    "competition_margin": score,
                    "competition_other_score": competitor_score,
                }
            )
            row["metadata"] = metadata
            results.append(
                self._row_to_retrieved_evidence(
                    row=row,
                    query=query,
                    score=score,
                )
            )
        return results

    @staticmethod
    def _build_competition_signal(
        *,
        support_scores: dict[str, float],
        competitor_scores: dict[str, dict[str, float]],
        discriminative_evidence: list[RetrievedEvidence],
        conflicting_evidence: list[RetrievedEvidence],
    ) -> HypothesisCompetitionSignal:
        ranked_support = sorted(support_scores.values(), reverse=True)
        support_score = float(np.mean(ranked_support[: min(3, len(ranked_support))])) if ranked_support else 0.0
        strongest_competitor_id: str | None = None
        strongest_competitor_support = 0.0
        for competitor_id, competitor_score_map in competitor_scores.items():
            ranked_competitor = sorted(competitor_score_map.values(), reverse=True)
            competitor_support = float(np.mean(ranked_competitor[: min(3, len(ranked_competitor))])) if ranked_competitor else 0.0
            if competitor_support > strongest_competitor_support:
                strongest_competitor_support = competitor_support
                strongest_competitor_id = competitor_id
        discriminative_score = float(np.mean([item.score for item in discriminative_evidence])) if discriminative_evidence else 0.0
        conflicting_score = float(np.mean([item.score for item in conflicting_evidence])) if conflicting_evidence else 0.0
        return HypothesisCompetitionSignal(
            support_score=support_score,
            discriminative_score=discriminative_score,
            conflicting_score=conflicting_score,
            margin_vs_next=support_score - strongest_competitor_support,
            strongest_competitor_id=strongest_competitor_id,
        )

    def _row_to_retrieved_evidence(
        self,
        *,
        row: dict[str, object],
        query: str,
        score: float,
    ) -> RetrievedEvidence:
        return RetrievedEvidence(
            evidence_id=str(row["evidence_id"]),
            evidence_type=str(row["evidence_type"]),
            score=score,
            source_query=query,
            heading=str(row["heading"]),
            section_path=list(row["section_path"]),
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            text=truncate_text(str(row["text"]), limit=self.settings.deep_chat_max_evidence_text_chars),
            html=truncate_text(str(row["html"]), limit=self.settings.deep_chat_max_evidence_text_chars),
            metadata=dict(row["metadata"]),
        )

    @staticmethod
    def _build_query_text(hypothesis: InterpretationHypothesis) -> str:
        parts = [hypothesis.normalized_question.strip()]
        if hypothesis.target_objects:
            parts.append(f"Target objects: {', '.join(hypothesis.target_objects)}")
        if hypothesis.required_constraints:
            parts.append(f"Required constraints: {', '.join(hypothesis.required_constraints)}")
        if hypothesis.disambiguation_focus:
            parts.append(f"Disambiguation focus: {', '.join(hypothesis.disambiguation_focus)}")
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _relation_type_preferences(
        *,
        relation: str,
        preferred_types: set[str],
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        normalized_relation = relation.strip().lower()
        global_allowed = {"section_unit", "paragraph_unit"}
        global_preferred = {"section_unit"} if normalized_relation in {"summary", "definition", "location"} else {"paragraph_unit"}
        local_allowed = {"paragraph_unit", "table_unit", "list_unit", "list_item_unit", "chunk_unit"}
        local_preferred = set(preferred_types)

        if normalized_relation in {"result_check", "numeric_lookup", "comparison"}:
            local_preferred.update({"table_unit", "list_item_unit", "paragraph_unit"})
        elif normalized_relation in {"usage_check", "location"}:
            local_preferred.update({"paragraph_unit", "chunk_unit", "list_item_unit"})
        elif normalized_relation in {"summary", "definition"}:
            local_preferred.update({"paragraph_unit", "chunk_unit"})
        else:
            local_preferred.update({"paragraph_unit", "chunk_unit"})

        return global_allowed, global_preferred, local_allowed, local_preferred


def build_runtime_rows(units: list[EvidenceUnit]) -> dict[str, list[dict[str, object]]]:
    rows_by_paper: dict[str, list[dict[str, object]]] = {}
    for index, unit in enumerate(units):
        search_text = build_evidence_search_text(unit)
        rows_by_paper.setdefault(unit.paper_id, []).append(
            {
                "index": index,
                "evidence_id": unit.evidence_id,
                "evidence_type": unit.evidence_type,
                "heading": unit.heading,
                "section_path": unit.section_path,
                "page_start": unit.page_start,
                "page_end": unit.page_end,
                "text": unit.text,
                "html": unit.html,
                "metadata": unit.metadata,
                "search_text": search_text,
                "search_tokens": tokenize(search_text),
            }
        )
    return rows_by_paper
