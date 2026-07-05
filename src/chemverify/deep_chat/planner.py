from __future__ import annotations

from ..models import PaperRecord
from .models import InterpretationPlan


class InterpretationPlanner:
    def build_system_prompt(self) -> str:
        return (
            "You plan multi-turn questions about a single indexed scientific paper. "
            "Return strict JSON only. "
            "Build a compact dialogue_state plus 1 to 4 interpretation hypotheses. "
            "Default to exactly 1 hypothesis unless the query is genuinely ambiguous or the latest user turn explicitly corrects a previous interpretation. "
            "Only produce multiple hypotheses when there is real competition on object meaning, relation meaning, or scope. "
            "Treat recent user corrections, negations, and narrowed scope as high-priority signals. "
            "Use these preferred_evidence_types only when appropriate: "
            "section_unit, paragraph_unit, table_unit, list_unit, list_item_unit, chunk_unit. "
            "Do not invent facts about the paper. "
            "The hypotheses should compete on object meaning, relation meaning, or scope when the question is ambiguous."
        )

    def build_user_payload(
        self,
        *,
        paper: PaperRecord,
        query: str,
        history: list[dict[str, str]],
    ) -> dict[str, object]:
        return {
            "paper": {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "intro_summary": paper.intro_summary,
                "section_headings": paper.section_headings,
            },
            "query": query,
            "history": history,
            "required_output": InterpretationPlan.model_json_schema(),
        }
