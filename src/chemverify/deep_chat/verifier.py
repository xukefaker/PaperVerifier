from __future__ import annotations

from ..models import PaperRecord
from .generator import _serialize_evidence, _serialize_fact, _serialize_hypothesis
from .models import FactExtractionBatch, GeneratedAnswer, HypothesisEvidencePack, InterpretationPlan, VerificationResult


class DeepChatVerifier:
    def build_system_prompt(self) -> str:
        return (
            "You verify whether an answer about a single scientific paper is grounded in the supplied evidence, aligned with the user's latest question, and based on the right interpretation. "
            "Return strict JSON only. "
            "support_verdict must be one of: supported, partially_supported, unsupported. "
            "alignment_verdict must be one of: aligned, partially_aligned, misaligned. "
            "interpretation_verdict must be one of: resolved, ambiguous, incorrect. "
            "competition_verdict must be one of: distinct_winner, weak_winner, no_clear_winner. "
            "Check whether the answer is consistent with the supplied hypothesis fact sets, not only with surface-level evidence relevance. "
            "Use competition signals plus discriminative/conflicting evidence to judge whether there is a real winner among competing interpretations. "
            "verified_evidence_ids must be a non-empty list containing only ids from the supplied evidence packs."
        )

    def build_user_payload(
        self,
        *,
        paper: PaperRecord,
        query: str,
        history: list[dict[str, str]],
        plan: InterpretationPlan,
        fact_batch: FactExtractionBatch,
        generated: GeneratedAnswer,
        hypothesis_packs: list[HypothesisEvidencePack],
    ) -> dict[str, object]:
        return {
            "paper": {
                "paper_id": paper.paper_id,
                "title": paper.title,
            },
            "query": query,
            "history": history,
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
            "generated_answer": generated.model_dump(mode="json"),
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
            "required_output": VerificationResult.model_json_schema(),
        }
