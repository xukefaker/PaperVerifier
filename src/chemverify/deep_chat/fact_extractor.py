from __future__ import annotations

from ..models import PaperRecord
from .models import FactExtractionBatch, HypothesisEvidencePack, InterpretationPlan


class DeepChatFactExtractor:
    def build_system_prompt(self) -> str:
        return (
            "You extract grounded facts for questions about a single indexed scientific paper. "
            "Return strict JSON only. "
            "For each interpretation hypothesis, produce a fact set grounded only in the supplied evidence for that hypothesis. "
            "Every fact must cite an evidence_id from the supplied evidence packs. "
            "Prefer precise facts about objects, relations, values, settings, and scope. "
            "Use supporting evidence first, then use discriminative and conflicting evidence to mark unresolved points or alternative readings. "
            "When table or list evidence is present, preserve row-or-column semantics instead of paraphrasing them away. "
            "Do not infer unsupported numbers, settings, or comparisons."
        )

    def build_user_payload(
        self,
        *,
        paper: PaperRecord,
        plan: InterpretationPlan,
        hypothesis_packs: list[HypothesisEvidencePack],
    ) -> dict[str, object]:
        return {
            "paper": {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "intro_summary": paper.intro_summary,
            },
            "dialogue_state": plan.dialogue_state.model_dump(mode="json"),
            "interpretation_hypotheses": [item.model_dump(mode="json") for item in plan.hypotheses],
            "hypothesis_evidence_packs": [
                {
                    "hypothesis_id": item.hypothesis_id,
                    "normalized_question": item.normalized_question,
                    "target_relation": item.target_relation,
                    "target_objects": list(item.target_objects),
                    "required_constraints": list(item.required_constraints),
                    "disambiguation_focus": list(item.disambiguation_focus),
                    "competition_signal": item.competition_signal.model_dump(mode="json"),
                    "global_evidence": [evidence.model_dump(mode="json") for evidence in item.evidence_pack.global_evidence],
                    "local_evidence": [evidence.model_dump(mode="json") for evidence in item.evidence_pack.local_evidence],
                    "discriminative_evidence": [evidence.model_dump(mode="json") for evidence in item.discriminative_evidence],
                    "conflicting_evidence": [evidence.model_dump(mode="json") for evidence in item.conflicting_evidence],
                }
                for item in hypothesis_packs
            ],
            "required_output": FactExtractionBatch.model_json_schema(),
        }
