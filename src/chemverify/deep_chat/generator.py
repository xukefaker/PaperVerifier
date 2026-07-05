from __future__ import annotations

from ..models import PaperRecord
from .models import (
    EvidenceFact,
    FactExtractionBatch,
    GeneratedAnswer,
    HypothesisEvidencePack,
    InterpretationHypothesis,
    InterpretationPlan,
    RetrievedEvidence,
)


def _serialize_hypothesis(item: InterpretationHypothesis) -> dict[str, object]:
    return {
        "hypothesis_id": item.hypothesis_id,
        "normalized_question": item.normalized_question,
        "target_relation": item.target_relation,
        "target_objects": list(item.target_objects),
        "required_constraints": list(item.required_constraints),
        "disambiguation_focus": list(item.disambiguation_focus),
        "priority": item.priority,
    }


def _serialize_fact(item: EvidenceFact) -> dict[str, object]:
    return {
        "fact_id": item.fact_id,
        "subject": item.subject,
        "relation": item.relation,
        "object": item.object,
        "value": item.value,
        "unit": item.unit,
        "scope": item.scope,
        "setting": item.setting,
        "evidence_id": item.evidence_id,
        "confidence": item.confidence,
    }


def _serialize_evidence(item: RetrievedEvidence) -> dict[str, object]:
    return {
        "evidence_id": item.evidence_id,
        "evidence_type": item.evidence_type,
        "score": item.score,
        "source_query": item.source_query,
        "heading": item.heading,
        "section_path": list(item.section_path),
        "page_start": item.page_start,
        "page_end": item.page_end,
        "text": item.text,
    }


class DeepChatAnswerGenerator:
    def build_system_prompt(self) -> str:
        return (
            "You answer multi-turn questions about a single indexed scientific paper by comparing competing interpretations. "
            "Use only the supplied metadata, dialogue history, dialogue state, interpretation hypotheses, hypothesis fact sets, and hypothesis evidence packs. "
            "Return strict JSON only. "
            "decision must be one of: ask_clarification, answer, unsupported. "
            "If decision is answer, winning_hypothesis_id must identify the best-supported hypothesis and used_evidence_ids must be a non-empty list containing only ids from the supplied evidence packs. "
            "Base your answer on the extracted fact sets first; use raw evidence only to preserve provenance or resolve ties. "
            "Use competition signals plus discriminative/conflicting evidence to prefer the interpretation that is best separated from its alternatives. "
            "If the evidence does not support a reliable answer, do not guess. "
            "If the paper evidence supports multiple competing readings and user intent is still unresolved, ask for clarification instead of forcing a single interpretation."
        )

    def build_user_payload(
        self,
        *,
        paper: PaperRecord,
        query: str,
        plan: InterpretationPlan,
        history: list[dict[str, str]],
        fact_batch: FactExtractionBatch,
        hypothesis_packs: list[HypothesisEvidencePack],
    ) -> dict[str, object]:
        return {
            "paper": {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "venue": paper.venue,
                "year": paper.year,
                "track": paper.track,
                "abstract": paper.abstract,
                "intro_summary": paper.intro_summary,
            },
            "query": query,
            "dialogue_state": plan.dialogue_state.model_dump(mode="json"),
            "interpretation_hypotheses": [_serialize_hypothesis(item) for item in plan.hypotheses],
            "hypothesis_fact_sets": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "normalized_question": item.normalized_question,
                    "extraction_summary": item.extraction_summary,
                    "facts": [_serialize_fact(fact) for fact in item.facts],
                    "unresolved_points": list(item.unresolved_points),
                }
                for item in fact_batch.fact_sets
            ],
            "history": history,
            "hypothesis_evidence_packs": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "normalized_question": item.normalized_question,
                    "target_relation": item.target_relation,
                    "target_objects": list(item.target_objects),
                    "required_constraints": list(item.required_constraints),
                    "disambiguation_focus": list(item.disambiguation_focus),
                    "competition_signal": item.competition_signal.model_dump(mode="json"),
                    "global_evidence": [_serialize_evidence(evidence) for evidence in item.evidence_pack.global_evidence],
                    "local_evidence": [_serialize_evidence(evidence) for evidence in item.evidence_pack.local_evidence],
                    "discriminative_evidence": [_serialize_evidence(evidence) for evidence in item.discriminative_evidence],
                    "conflicting_evidence": [_serialize_evidence(evidence) for evidence in item.conflicting_evidence],
                }
                for item in hypothesis_packs
            ],
            "required_output": GeneratedAnswer.model_json_schema(),
        }
