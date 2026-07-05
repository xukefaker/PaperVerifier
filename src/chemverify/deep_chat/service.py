from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..models import PaperRecord
from ..provider import require_openai_model
from ..search import SearchEngine
from ..storage import LocalStore
from ..utils import truncate_text
from .fact_extractor import DeepChatFactExtractor
from .generator import DeepChatAnswerGenerator
from .models import (
    DeepChatResponsePayload,
    EvidenceUnit,
    FactExtractionBatch,
    GeneratedAnswer,
    HypothesisEvidencePack,
    InterpretationPlan,
    VerificationResult,
)
from .planner import InterpretationPlanner
from .retriever import DeepChatRetriever, EvidenceRuntime, build_runtime_rows
from .store import DeepChatStore
from .verifier import DeepChatVerifier

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(slots=True)
class _Runtime:
    evidence_units: list[EvidenceUnit]
    evidence_lookup: dict[str, EvidenceUnit]
    evidence_runtime: EvidenceRuntime


class DeepChatService:
    def __init__(
        self,
        settings: Settings,
        store: LocalStore,
        engine: SearchEngine,
        *,
        root_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.engine = engine
        self.deep_chat_store = DeepChatStore(settings, root_dir=root_dir)
        self.planner = InterpretationPlanner()
        self.retriever = DeepChatRetriever(settings)
        self.fact_extractor = DeepChatFactExtractor()
        self.generator = DeepChatAnswerGenerator()
        self.verifier = DeepChatVerifier()
        self.runtime: _Runtime | None = None
        self._load_lock = Lock()
        self._client_lock = Lock()
        self._http_client: httpx.Client | None = None

    def load(self) -> None:
        if self.runtime is not None:
            return
        with self._load_lock:
            if self.runtime is not None:
                return
            if self.engine.runtime is None:
                self.engine.load()
            evidence_units = self.deep_chat_store.load_evidence_units()
            index_meta = self.deep_chat_store.load_index_meta()
            evidence_ids, evidence_vectors = self.deep_chat_store.load_vectors()
            if not evidence_units or not evidence_ids:
                raise RuntimeError("Deep chat evidence assets are missing. Run build-index first.")
            unit_ids = [unit.evidence_id for unit in evidence_units]
            self._assert_runtime_alignment(
                index_meta=index_meta,
                unit_ids=unit_ids,
                vector_ids=evidence_ids,
                vectors=evidence_vectors,
            )
            self._assert_encoder_compatibility(
                index_meta=index_meta,
                vectors=evidence_vectors,
            )
            evidence_lookup = {unit.evidence_id: unit for unit in evidence_units}
            rows_by_paper = build_runtime_rows(evidence_units)
            self.runtime = _Runtime(
                evidence_units=evidence_units,
                evidence_lookup=evidence_lookup,
                evidence_runtime=EvidenceRuntime(
                    evidence_lookup=evidence_lookup,
                    rows_by_paper=rows_by_paper,
                    evidence_ids=evidence_ids,
                    evidence_vectors=evidence_vectors,
                ),
            )

    def answer(
        self,
        paper_id: str,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if self.runtime is None:
            self.load()
        runtime = self.runtime
        assert runtime is not None
        assert self.engine.runtime is not None

        paper = self.engine.runtime.paper_lookup.get(paper_id) or self.store.get_paper(paper_id)
        if paper is None:
            raise KeyError(paper_id)

        history_window = (history or [])[-self.settings.deep_chat_history_turn_limit :]
        plan = self._call_model(
            system_prompt=self.planner.build_system_prompt(),
            user_payload=self.planner.build_user_payload(paper=paper, query=normalized_query, history=history_window),
            response_model=InterpretationPlan,
        )
        if not plan.hypotheses:
            raise RuntimeError("Deep chat planner returned no interpretation hypotheses.")
        plan = self._limit_plan_hypotheses(plan, max_hypotheses=2)

        hypothesis_packs = self.retriever.retrieve_for_hypotheses(
            paper_id=paper_id,
            hypotheses=plan.hypotheses,
            runtime=runtime.evidence_runtime,
            query_encoder=self.engine.runtime.chunk_encoder,
            reranker=self.engine.runtime.local_reranker,
        )
        if not any(pack.all_evidence() for pack in hypothesis_packs):
            raise RuntimeError("No indexed deep chat evidence is available for this paper.")
        fact_batch = self._call_model(
            system_prompt=self.fact_extractor.build_system_prompt(),
            user_payload=self.fact_extractor.build_user_payload(
                paper=paper,
                plan=plan,
                hypothesis_packs=hypothesis_packs,
            ),
            response_model=FactExtractionBatch,
        )
        self._assert_fact_batch_alignment(
            plan=plan,
            fact_batch=fact_batch,
            hypothesis_packs=hypothesis_packs,
        )

        generated = self._call_model(
            system_prompt=self.generator.build_system_prompt(),
            user_payload=self.generator.build_user_payload(
                paper=paper,
                query=normalized_query,
                plan=plan,
                history=history_window,
                fact_batch=fact_batch,
                hypothesis_packs=hypothesis_packs,
            ),
            response_model=GeneratedAnswer,
        )

        hypothesis_lookup = {item.hypothesis_id: item for item in plan.hypotheses}
        if generated.decision == "answer":
            if not generated.winning_hypothesis_id:
                raise RuntimeError("Deep chat answer did not choose a winning interpretation hypothesis.")
            if generated.winning_hypothesis_id not in hypothesis_lookup:
                raise RuntimeError("Deep chat answer selected an unknown interpretation hypothesis id.")
            if not generated.used_evidence_ids:
                raise RuntimeError("Deep chat answer returned no cited evidence ids.")
            self._assert_hypothesis_grounding(
                hypothesis_packs=hypothesis_packs,
                hypothesis_id=generated.winning_hypothesis_id,
                evidence_ids=generated.used_evidence_ids,
                source="answer",
            )

        verifier_payload: VerificationResult | None = None
        final_decision = generated.decision
        final_answer = generated.answer.strip()
        uncertainty_note = generated.uncertainty_note
        if generated.decision == "answer":
            verifier_plan, verifier_fact_batch, verifier_hypothesis_packs = self._prepare_verifier_context(
                plan=plan,
                fact_batch=fact_batch,
                hypothesis_packs=hypothesis_packs,
                generated=generated,
            )
            verifier_payload = self._call_model(
                system_prompt=self.verifier.build_system_prompt(),
                user_payload=self.verifier.build_user_payload(
                    paper=paper,
                    query=normalized_query,
                    history=history_window,
                    plan=verifier_plan,
                    fact_batch=verifier_fact_batch,
                    generated=generated,
                    hypothesis_packs=verifier_hypothesis_packs,
                ),
                response_model=VerificationResult,
            )
            if not verifier_payload.verified_evidence_ids:
                raise RuntimeError("Deep chat verifier returned no validated evidence ids.")
            self._assert_hypothesis_grounding(
                hypothesis_packs=hypothesis_packs,
                hypothesis_id=generated.winning_hypothesis_id,
                evidence_ids=verifier_payload.verified_evidence_ids,
                source="verifier",
            )
            if (
                verifier_payload.interpretation_verdict in {"ambiguous", "incorrect"}
                or verifier_payload.competition_verdict == "no_clear_winner"
            ):
                final_decision = "ask_clarification"
                uncertainty_note = verifier_payload.failure_reason or generated.answer.strip() or uncertainty_note
                final_answer = "I need clarification to answer this precisely from the indexed evidence in this paper."
            elif (
                verifier_payload.support_verdict == "unsupported"
                or verifier_payload.alignment_verdict == "misaligned"
            ):
                final_decision = "unsupported"
                uncertainty_note = verifier_payload.failure_reason or generated.answer.strip() or uncertainty_note
                final_answer = "I cannot answer this reliably from the indexed evidence in this paper."
            elif (
                verifier_payload.support_verdict == "partially_supported"
                or verifier_payload.alignment_verdict == "partially_aligned"
            ):
                note = verifier_payload.failure_reason or "The available evidence only partially supports the answer."
                uncertainty_note = note if not uncertainty_note else f"{uncertainty_note} {note}".strip()
            elif verifier_payload.competition_verdict == "weak_winner":
                note = verifier_payload.failure_reason or "The selected interpretation is only weakly separated from competing readings."
                uncertainty_note = note if not uncertainty_note else f"{uncertainty_note} {note}".strip()

        used_evidence_ids = verifier_payload.verified_evidence_ids if verifier_payload else generated.used_evidence_ids
        response_evidence = self._select_response_evidence(
            hypothesis_packs=hypothesis_packs,
            hypothesis_id=generated.winning_hypothesis_id,
            used_evidence_ids=used_evidence_ids,
        )
        citations = self._build_citations(response_evidence)
        payload = DeepChatResponsePayload(
            paper_id=paper.paper_id,
            decision=final_decision,
            answer=final_answer,
            rewritten_query=self._display_query(plan=plan, generated=generated),
            uncertainty_note=uncertainty_note,
            citations=citations,
            evidence=[
                {
                    "evidence_id": item.evidence_id,
                    "evidence_type": item.evidence_type,
                    "heading": item.heading,
                    "section_path": list(item.section_path),
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "snippet": truncate_text(item.text, limit=360),
                    "html": truncate_text(item.html, limit=1200) or None,
                }
                for item in response_evidence
            ],
            verifier=verifier_payload.model_dump(mode="json") if verifier_payload else None,
        )
        return payload.model_dump(mode="json")

    def close(self) -> None:
        with self._client_lock:
            if self._http_client is None:
                return
            close = getattr(self._http_client, "close", None)
            if callable(close):
                close()
            self._http_client = None

    def _call_model(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, object],
        response_model: type[ModelT],
    ) -> ModelT:
        if not self.settings.openai_enabled or not self.settings.openai_api_key:
            raise RuntimeError("Deep chat requires an enabled API model.")
        model = require_openai_model(self.settings)

        validation_error: str | None = None
        for attempt in range(2):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ]
            if validation_error is not None:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Your previous response was invalid. "
                            f"Validation error: {validation_error}. "
                            "Return a corrected JSON object only, with all labels inside the allowed schema."
                        ),
                    }
                )
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            try:
                client = self._get_http_client()
                response = client.post(
                    f"{self.settings.openai_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                raw = response.json()
                content = json.loads(raw["choices"][0]["message"]["content"])
                return response_model.model_validate(content)
            except httpx.HTTPError as exc:
                raise RuntimeError(f"Deep chat upstream request failed: {exc}") from exc
            except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
                validation_error = str(exc)
                if attempt == 1:
                    raise RuntimeError(f"Deep chat returned invalid structured output: {exc}") from exc
        raise RuntimeError("Deep chat model did not return valid structured output.")

    def _get_http_client(self) -> httpx.Client:
        if self._http_client is not None:
            return self._http_client
        with self._client_lock:
            if self._http_client is None:
                self._http_client = httpx.Client(timeout=self.settings.request_timeout)
            return self._http_client

    @staticmethod
    def _limit_plan_hypotheses(plan: InterpretationPlan, *, max_hypotheses: int) -> InterpretationPlan:
        if len(plan.hypotheses) <= max_hypotheses:
            return plan
        limited = sorted(plan.hypotheses, key=lambda item: item.priority, reverse=True)[:max_hypotheses]
        return plan.model_copy(update={"hypotheses": limited})

    @staticmethod
    def _trim_evidence_list(
        evidence: list,
        *,
        keep_ids: set[str] | None = None,
        top_k: int = 0,
    ) -> list:
        keep_ids = keep_ids or set()
        ranked = sorted(evidence, key=lambda item: item.score, reverse=True)
        selected: list = []
        seen: set[str] = set()
        for item in ranked:
            if item.evidence_id in keep_ids:
                selected.append(item)
                seen.add(item.evidence_id)
        if top_k > 0:
            for item in ranked:
                if item.evidence_id in seen:
                    continue
                selected.append(item)
                seen.add(item.evidence_id)
                if len(selected) >= len(keep_ids) + top_k:
                    break
        return selected

    def _prepare_verifier_context(
        self,
        *,
        plan: InterpretationPlan,
        fact_batch: FactExtractionBatch,
        hypothesis_packs: list[HypothesisEvidencePack],
        generated: GeneratedAnswer,
    ) -> tuple[InterpretationPlan, FactExtractionBatch, list[HypothesisEvidencePack]]:
        if not generated.winning_hypothesis_id:
            return plan, fact_batch, hypothesis_packs
        pack_by_id = {item.hypothesis_id: item for item in hypothesis_packs}
        selected_ids = [generated.winning_hypothesis_id]
        winning_pack = pack_by_id.get(generated.winning_hypothesis_id)
        strongest_competitor_id = (
            winning_pack.competition_signal.strongest_competitor_id if winning_pack is not None else None
        )
        if strongest_competitor_id and strongest_competitor_id in pack_by_id:
            selected_ids.append(strongest_competitor_id)
        selected_id_set = set(selected_ids)
        verifier_plan = plan.model_copy(
            update={"hypotheses": [item for item in plan.hypotheses if item.hypothesis_id in selected_id_set]}
        )
        verifier_fact_batch = fact_batch.model_copy(
            update={"fact_sets": [item for item in fact_batch.fact_sets if item.hypothesis_id in selected_id_set]}
        )
        winning_evidence_ids = set(generated.used_evidence_ids)
        verifier_hypothesis_packs: list[HypothesisEvidencePack] = []
        for item in hypothesis_packs:
            if item.hypothesis_id not in selected_id_set:
                continue
            keep_ids = winning_evidence_ids if item.hypothesis_id == generated.winning_hypothesis_id else set()
            verifier_hypothesis_packs.append(
                item.model_copy(
                    update={
                        "evidence_pack": item.evidence_pack.model_copy(
                            update={
                                "global_evidence": self._trim_evidence_list(
                                    item.evidence_pack.global_evidence,
                                    keep_ids=keep_ids,
                                    top_k=2,
                                ),
                                "local_evidence": self._trim_evidence_list(
                                    item.evidence_pack.local_evidence,
                                    keep_ids=keep_ids,
                                    top_k=2,
                                ),
                            }
                        ),
                        "discriminative_evidence": self._trim_evidence_list(
                            item.discriminative_evidence,
                            top_k=1,
                        ),
                        "conflicting_evidence": self._trim_evidence_list(
                            item.conflicting_evidence,
                            top_k=1,
                        ),
                    }
                )
            )
        return verifier_plan, verifier_fact_batch, verifier_hypothesis_packs

    @staticmethod
    def _assert_fact_batch_alignment(
        *,
        plan: InterpretationPlan,
        fact_batch: FactExtractionBatch,
        hypothesis_packs: list[HypothesisEvidencePack],
    ) -> None:
        allowed_hypothesis_ids = {item.hypothesis_id for item in plan.hypotheses}
        evidence_ids_by_hypothesis = {
            item.hypothesis_id: {evidence.evidence_id for evidence in item.all_evidence()}
            for item in hypothesis_packs
        }
        if not fact_batch.fact_sets:
            raise RuntimeError("Deep chat fact extractor returned no fact sets.")
        seen_hypothesis_ids: set[str] = set()
        for fact_set in fact_batch.fact_sets:
            if fact_set.hypothesis_id not in allowed_hypothesis_ids:
                raise RuntimeError("Deep chat fact extractor returned an unknown hypothesis id.")
            if fact_set.hypothesis_id in seen_hypothesis_ids:
                raise RuntimeError("Deep chat fact extractor returned duplicate hypothesis fact sets.")
            seen_hypothesis_ids.add(fact_set.hypothesis_id)
            allowed_evidence_ids = evidence_ids_by_hypothesis.get(fact_set.hypothesis_id, set())
            for fact in fact_set.facts:
                if fact.evidence_id not in allowed_evidence_ids:
                    raise RuntimeError("Deep chat fact extractor returned an invalid grounded evidence id.")
        if seen_hypothesis_ids != allowed_hypothesis_ids:
            raise RuntimeError("Deep chat fact extractor did not return fact sets for every interpretation hypothesis.")

    @staticmethod
    def _display_query(*, plan: InterpretationPlan, generated: GeneratedAnswer) -> str:
        if generated.winning_hypothesis_id:
            for hypothesis in plan.hypotheses:
                if hypothesis.hypothesis_id == generated.winning_hypothesis_id:
                    return hypothesis.normalized_question
        return plan.hypotheses[0].normalized_question

    @staticmethod
    def _assert_hypothesis_grounding(
        *,
        hypothesis_packs: list[HypothesisEvidencePack],
        hypothesis_id: str | None,
        evidence_ids: list[str],
        source: str,
    ) -> None:
        if not hypothesis_id:
            raise RuntimeError(f"Deep chat {source} did not specify a winning interpretation hypothesis.")
        allowed_evidence_ids = set()
        for pack in hypothesis_packs:
            if pack.hypothesis_id == hypothesis_id:
                allowed_evidence_ids = {item.evidence_id for item in pack.all_evidence()}
                break
        if not allowed_evidence_ids:
            raise RuntimeError(f"Deep chat {source} selected a hypothesis with no retrieved evidence.")
        invalid_ids = [evidence_id for evidence_id in evidence_ids if evidence_id not in allowed_evidence_ids]
        if invalid_ids:
            raise RuntimeError(f"Deep chat {source} cited evidence outside the winning interpretation hypothesis.")

    @staticmethod
    def _select_response_evidence(
        *,
        hypothesis_packs: list[HypothesisEvidencePack],
        hypothesis_id: str | None,
        used_evidence_ids: list[str],
    ) -> list:
        if not used_evidence_ids:
            return []
        selected: list = []
        seen: set[str] = set()
        lookup = {}
        for pack in hypothesis_packs:
            if hypothesis_id is not None and pack.hypothesis_id != hypothesis_id:
                continue
            for item in pack.all_evidence():
                lookup[item.evidence_id] = item
        for evidence_id in used_evidence_ids:
            item = lookup.get(evidence_id)
            if item is None or item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            selected.append(item)
        if not selected:
            raise RuntimeError("Deep chat returned no valid grounded evidence ids.")
        return selected

    @staticmethod
    def _build_citations(evidence: list) -> list[dict[str, object]]:
        return [
            {
                "evidence_id": item.evidence_id,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "section_path": list(item.section_path),
                "snippet": truncate_text(item.text, limit=260),
                "html": truncate_text(item.html, limit=1000) or None,
            }
            for item in evidence
        ]

    @staticmethod
    def _assert_runtime_alignment(
        *,
        index_meta: dict,
        unit_ids: list[str],
        vector_ids: list[str],
        vectors,
    ) -> None:
        meta_ids = [str(item) for item in index_meta.get("ids", [])]
        if not index_meta:
            raise RuntimeError("Deep chat index metadata is missing. Run build-index again.")
        if meta_ids != unit_ids or meta_ids != vector_ids:
            raise RuntimeError("Deep chat evidence ids do not align across units, metadata, and vectors.")
        if vectors.shape[0] != len(vector_ids):
            raise RuntimeError("Deep chat evidence vector row count does not match ids.")
        for key in ("texts", "tokens", "paper_ids", "evidence_types", "section_ids"):
            values = index_meta.get(key)
            if not isinstance(values, list) or len(values) != len(meta_ids):
                raise RuntimeError(f"Deep chat index metadata field '{key}' is incomplete or misaligned.")

    def _assert_encoder_compatibility(self, *, index_meta: dict, vectors) -> None:
        if self.engine.runtime is None:
            raise RuntimeError("Search runtime must be loaded before deep chat.")
        encoder = self.engine.runtime.chunk_encoder
        backend = str(index_meta.get("encoder_backend", "")).strip()
        model = str(index_meta.get("encoder_model", "")).strip()
        vector_dim = int(index_meta.get("vector_dim", 0) or 0)
        if not backend or not model or vector_dim <= 0:
            raise RuntimeError("Deep chat index metadata is missing encoder provenance. Rebuild indexes.")
        if backend != encoder.backend_name:
            raise RuntimeError(
                f"Deep chat index was built with backend {backend!r}, but runtime requires {encoder.backend_name!r}."
            )
        if model != encoder.model_name:
            raise RuntimeError(
                f"Deep chat index was built with model {model!r}, but runtime requires {encoder.model_name!r}."
            )
        if vectors.ndim != 2 or vectors.shape[1] != vector_dim:
            raise RuntimeError(
                f"Deep chat index vector_dim={vector_dim} does not match stored matrix shape {tuple(vectors.shape)}."
            )
