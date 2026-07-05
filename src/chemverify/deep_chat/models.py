from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EvidenceUnitType = Literal[
    "section_unit",
    "paragraph_unit",
    "table_unit",
    "list_unit",
    "list_item_unit",
    "chunk_unit",
]
DeepChatDecision = Literal["ask_clarification", "answer", "unsupported"]
SupportVerdict = Literal["supported", "partially_supported", "unsupported"]
AlignmentVerdict = Literal["aligned", "partially_aligned", "misaligned"]
InterpretationVerdict = Literal["resolved", "ambiguous", "incorrect"]
CompetitionVerdict = Literal["distinct_winner", "weak_winner", "no_clear_winner"]
QuestionType = Literal[
    "summary",
    "definition",
    "usage_check",
    "result_check",
    "comparison",
    "numeric_lookup",
    "location",
    "correction",
    "other",
]
AnswerGranularity = Literal["overview", "targeted", "specific", "table_or_value"]


class DialogueState(BaseModel):
    active_topic: str = ""
    active_objects: list[str] = Field(default_factory=list)
    question_type: QuestionType = "other"
    current_constraints: list[str] = Field(default_factory=list)
    recent_user_corrections: list[str] = Field(default_factory=list)
    discarded_interpretations: list[str] = Field(default_factory=list)
    pending_ambiguities: list[str] = Field(default_factory=list)
    answer_granularity: AnswerGranularity = "targeted"


class InterpretationHypothesis(BaseModel):
    hypothesis_id: str
    normalized_question: str
    target_objects: list[str] = Field(default_factory=list)
    target_relation: str = ""
    required_constraints: list[str] = Field(default_factory=list)
    preferred_evidence_types: list[EvidenceUnitType] = Field(default_factory=list)
    disambiguation_focus: list[str] = Field(default_factory=list)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class InterpretationPlan(BaseModel):
    dialogue_state: DialogueState
    hypotheses: list[InterpretationHypothesis] = Field(default_factory=list, min_length=1, max_length=4)
    planner_summary: str | None = None


class EvidenceUnit(BaseModel):
    evidence_id: str
    paper_id: str
    evidence_type: EvidenceUnitType
    section_id: str | None = None
    heading: str = ""
    section_path: list[str] = Field(default_factory=list)
    page_start: int = 1
    page_end: int = 1
    text: str = ""
    html: str = ""
    object_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedEvidence(BaseModel):
    evidence_id: str
    evidence_type: EvidenceUnitType
    score: float
    source_query: str
    heading: str = ""
    section_path: list[str] = Field(default_factory=list)
    page_start: int = 1
    page_end: int = 1
    text: str = ""
    html: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidencePack(BaseModel):
    global_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    local_evidence: list[RetrievedEvidence] = Field(default_factory=list)

    def all_evidence(self) -> list[RetrievedEvidence]:
        output: list[RetrievedEvidence] = []
        seen: set[str] = set()
        for item in [*self.global_evidence, *self.local_evidence]:
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            output.append(item)
        return output


class HypothesisCompetitionSignal(BaseModel):
    support_score: float = 0.0
    discriminative_score: float = 0.0
    conflicting_score: float = 0.0
    margin_vs_next: float = 0.0
    strongest_competitor_id: str | None = None


class HypothesisEvidencePack(BaseModel):
    hypothesis_id: str
    normalized_question: str
    target_relation: str = ""
    target_objects: list[str] = Field(default_factory=list)
    required_constraints: list[str] = Field(default_factory=list)
    disambiguation_focus: list[str] = Field(default_factory=list)
    evidence_pack: EvidencePack = Field(default_factory=EvidencePack)
    discriminative_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    conflicting_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    competition_signal: HypothesisCompetitionSignal = Field(default_factory=HypothesisCompetitionSignal)

    def all_evidence(self) -> list[RetrievedEvidence]:
        output: list[RetrievedEvidence] = []
        seen: set[str] = set()
        for item in [*self.evidence_pack.all_evidence(), *self.discriminative_evidence, *self.conflicting_evidence]:
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            output.append(item)
        return output


class EvidenceFact(BaseModel):
    fact_id: str
    subject: str
    relation: str
    object: str = ""
    value: str | None = None
    unit: str | None = None
    scope: str | None = None
    setting: str | None = None
    evidence_id: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class HypothesisFactSet(BaseModel):
    hypothesis_id: str
    normalized_question: str
    extraction_summary: str | None = None
    facts: list[EvidenceFact] = Field(default_factory=list)
    unresolved_points: list[str] = Field(default_factory=list)


class FactExtractionBatch(BaseModel):
    fact_sets: list[HypothesisFactSet] = Field(default_factory=list, min_length=1, max_length=4)


class GeneratedAnswer(BaseModel):
    decision: DeepChatDecision
    answer: str
    winning_hypothesis_id: str | None = None
    rejected_hypothesis_ids: list[str] = Field(default_factory=list)
    uncertainty_note: str | None = None
    used_evidence_ids: list[str] = Field(default_factory=list)
    reasoning_summary: str | None = None


class VerificationResult(BaseModel):
    support_verdict: SupportVerdict
    alignment_verdict: AlignmentVerdict
    interpretation_verdict: InterpretationVerdict = "resolved"
    competition_verdict: CompetitionVerdict = "distinct_winner"
    strongest_competitor_id: str | None = None
    confidence: float = 0.0
    verified_evidence_ids: list[str] = Field(default_factory=list)
    failure_reason: str | None = None


class DeepChatResponsePayload(BaseModel):
    paper_id: str
    decision: DeepChatDecision
    answer: str
    rewritten_query: str
    uncertainty_note: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    verifier: dict[str, Any] | None = None
