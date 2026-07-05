from __future__ import annotations

import hashlib
import math
import re
import string
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "via",
    "what",
    "which",
    "with",
}

SECTION_PRIOR_MAP = {
    "method": ["method", "approach", "architecture", "model", "system", "framework"],
    "experiment": ["experiment", "evaluation", "results", "analysis", "dataset", "benchmark"],
    "limitation": ["limitation", "limitations", "discussion", "conclusion", "future"],
    "training": ["training", "optimization", "implementation", "appendix", "hyperparameter"],
    "baseline": ["baseline", "comparison", "ablation", "results", "evaluation"],
}

VENUE_ALIASES = {
    "acl": "acl",
    "emnlp": "emnlp",
    "naacl": "naacl",
    "eacl": "eacl",
    "coling": "coling",
    "tacl": "tacl",
    "iclr": "iclr",
    "neurips": "neurips",
    "nips": "neurips",
    "icml": "icml",
    "aaai": "aaai",
    "ijcai": "ijcai",
}

TRACK_ALIASES = {
    "long": "long",
    "short": "short",
    "findings": "findings",
    "demo": "demo",
    "industry": "industry",
    "library": "library",
}

ENTITY_CONTEXT_MAP = {
    "dataset": ["dataset", "datasets", "benchmark", "benchmarks", "corpus", "task", "tasks", "suite"],
    "benchmark": ["benchmark", "benchmarks", "dataset", "datasets", "suite", "task", "tasks"],
    "model": ["model", "models", "backbone", "architecture"],
    "method": ["method", "methods", "approach", "framework", "algorithm"],
    "metric": ["metric", "metrics", "score", "scores", "performance"],
    "task": ["task", "tasks", "setting", "evaluation"],
}

EVIDENCE_TYPE_RULES = {
    "experiment": {
        "heading": ("experiment", "evaluation", "setup", "benchmark", "dataset"),
        "text": ("we evaluate", "we evaluated", "experiments", "evaluation", "benchmark", "dataset", "test set"),
    },
    "result": {
        "heading": ("result", "results", "analysis"),
        "text": ("results", "performance", "score", "scores", "accuracy", "f1", "bleu", "wins", "improves", "table"),
    },
    "comparison": {
        "heading": ("comparison", "baseline", "ablation"),
        "text": ("compare", "compared", "comparison", "baseline", "versus", "vs", "ablation"),
    },
    "limitation": {
        "heading": ("limitation", "limitations", "discussion", "future work"),
        "text": ("limitation", "limitations", "future work", "failure case", "fails on", "challenge", "weakness"),
    },
    "appendix": {
        "heading": ("appendix", "supplementary"),
        "text": ("appendix", "supplementary"),
    },
    "efficiency": {
        "heading": ("efficiency", "analysis", "results"),
        "text": ("latency", "speed", "runtime", "throughput", "efficiency", "cost", "memory", "flops", "parameter"),
    },
    "human_eval": {
        "heading": ("human evaluation", "evaluation", "analysis"),
        "text": ("human evaluation", "human eval", "human judges", "annotators", "preference study", "user study"),
    },
    "error_analysis": {
        "heading": ("error analysis", "analysis", "discussion"),
        "text": ("error analysis", "failure case", "qualitative analysis", "case study", "errors"),
    },
    "prompt_template": {
        "heading": ("appendix", "prompt", "implementation"),
        "text": ("prompt template", "prompts", "instruction template", "system prompt", "template is shown"),
    },
}

_ENTITY_FOLLOWING_TERMS = {
    "dataset": "dataset",
    "datasets": "dataset",
    "benchmark": "benchmark",
    "benchmarks": "benchmark",
    "corpus": "dataset",
    "suite": "dataset",
    "task": "task",
    "tasks": "task",
    "method": "method",
    "methods": "method",
    "approach": "method",
    "model": "model",
    "models": "model",
    "metric": "metric",
    "metrics": "metric",
}

_COMMON_ENTITY_STOPWORDS = {
    "acl",
    "emnlp",
    "naacl",
    "paper",
    "papers",
    "results",
    "experiments",
    "experiment",
    "appendix",
    "table",
    "tables",
    "methods",
    "method",
    "datasets",
    "dataset",
    "benchmark",
    "benchmarks",
}

_ENTITY_CONTROL_TERMS = {
    "find",
    "show",
    "list",
    "locate",
    "retrieve",
    "search",
    "give",
    "tell",
    "work",
    "works",
    "study",
    "studies",
    "about",
    "within",
    "among",
    "report",
    "reported",
    "reporting",
    "section",
    "sections",
    "result",
    "results",
    "experiment",
    "experiments",
    "appendix",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(text: str) -> str:
    return normalize_whitespace(text.replace("\n", " "))


def tokenize(text: str) -> list[str]:
    lowered = text.lower().translate(str.maketrans({c: " " for c in string.punctuation}))
    return [token for token in lowered.split() if token and token not in STOPWORDS]


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    counts = Counter(tokenize(text))
    return [token for token, _ in counts.most_common(limit)]


def split_text_into_windows(tokens: list[str], target_size: int, overlap: int) -> list[tuple[int, int, list[str]]]:
    if not tokens:
        return []
    windows: list[tuple[int, int, list[str]]] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + target_size)
        windows.append((start, end, tokens[start:end]))
        if end == len(tokens):
            break
        start = max(start + 1, end - overlap)
    return windows


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def make_stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def cosine_similarity_matrix(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    if doc_vectors.size == 0:
        return np.array([], dtype=float)
    query = query_vector / (np.linalg.norm(query_vector) + 1e-12)
    docs = doc_vectors / (np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-12)
    return docs @ query


def reciprocal_rank_fusion(rankings: Iterable[tuple[str, float]], base: float = 60.0) -> dict[str, float]:
    fused: dict[str, float] = {}
    for item_id, rank in rankings:
        fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (base + rank)
    return fused


def weighted_rrf(rankings: dict[str, list[str]], weights: dict[str, float], base: float = 60.0) -> dict[str, float]:
    fused: dict[str, float] = {}
    for key, item_ids in rankings.items():
        weight = weights.get(key, 1.0)
        for rank, item_id in enumerate(item_ids, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + weight * (1.0 / (base + rank))
    return fused


def top_k_from_scores(scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]


def flatten_section_priors(terms: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for term in terms:
        mapped = SECTION_PRIOR_MAP.get(term, [term])
        for value in mapped:
            lowered = value.lower()
            if lowered not in seen:
                seen.append(lowered)
    return seen


def detect_section_prior_labels(text: str) -> list[str]:
    lowered = text.lower()
    labels: list[str] = []
    if any(token in lowered for token in ["method", "approach", "architecture", "algorithm", "framework"]):
        labels.append("method")
    if any(token in lowered for token in ["experiment", "evaluation", "dataset", "result", "benchmark", "baseline", "ablation"]):
        labels.append("experiment")
    if any(token in lowered for token in ["limitation", "weakness", "failure", "future work", "discussion"]):
        labels.append("limitation")
    if any(token in lowered for token in ["train", "optimization", "hyperparameter", "implementation"]):
        labels.append("training")
    if any(token in lowered for token in ["compare", "baseline"]):
        labels.append("baseline")
    return labels


def truncate_text(text: str, limit: int = 320) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        normalized = normalize_whitespace(str(item))
        if normalized and normalized not in seen:
            seen.append(normalized)
    return seen


def infer_evidence_scores(
    *,
    heading: str,
    section_path: list[str],
    text: str,
) -> dict[str, float]:
    combined_heading = normalize_whitespace(" ".join([heading, *section_path])).lower()
    normalized_text = normalize_whitespace(text).lower()
    scores: dict[str, float] = {}
    for evidence_type, rule in EVIDENCE_TYPE_RULES.items():
        score = 0.0
        if any(token in combined_heading for token in rule["heading"]):
            score += 0.55
        text_hits = sum(1 for token in rule["text"] if token in normalized_text)
        if text_hits:
            score += min(0.45, 0.15 * text_hits)
        if evidence_type == "appendix" and any(token in combined_heading for token in ("appendix", "supplementary")):
            score = max(score, 0.9)
        if score > 0.0:
            scores[evidence_type] = round(min(score, 1.0), 4)
    return scores


def select_evidence_types(scores: dict[str, float], threshold: float = 0.35) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [evidence_type for evidence_type, score in ranked if score >= threshold]


def build_typed_evidence_summary(items: Iterable[dict[str, float]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for score_map in items:
        for evidence_type, score in score_map.items():
            if score >= 0.35:
                summary[evidence_type] = summary.get(evidence_type, 0) + 1
    return summary


def is_valid_entity_candidate(term: str) -> bool:
    normalized = normalize_constraint_term(term)
    if not normalized:
        return False
    if normalized in _COMMON_ENTITY_STOPWORDS or normalized in _ENTITY_CONTROL_TERMS:
        return False
    if normalized in VENUE_ALIASES or normalized in TRACK_ALIASES:
        return False
    if normalized in STOPWORDS:
        return False
    if normalized.isdigit() or re.fullmatch(r"20\d{2}", normalized):
        return False
    return True


def extract_query_constraints(query: str) -> dict:
    lowered = query.lower()
    venues = [canonical for alias, canonical in VENUE_ALIASES.items() if re.search(rf"\b{re.escape(alias)}\b", lowered)]
    tracks = [canonical for alias, canonical in TRACK_ALIASES.items() if re.search(rf"\b{re.escape(alias)}\b", lowered)]
    years = sorted({int(year) for year in re.findall(r"\b(20\d{2})\b", lowered)})

    entity_terms: list[str] = []
    hard_match_terms: list[str] = []
    entity_kind: str | None = None
    comparison_terms: list[str] = []
    required_evidence_types: list[str] = []
    intent_tags: list[str] = []
    negated_terms: list[str] = []

    for phrase in re.findall(r'"([^"]+)"', query):
        normalized = normalize_constraint_term(phrase)
        if normalized:
            entity_terms.append(normalized)
            hard_match_terms.append(normalized)

    explicit_entity_patterns = [
        r"\b([A-Za-z][A-Za-z0-9+._/-]{1,40})\s+(dataset|benchmark|corpus|suite|task|method|model|metric)\b",
        r"\b([A-Za-z][A-Za-z0-9+._/-]{1,40})\s+(benchmarks|datasets|tasks|methods|models|metrics)\b",
    ]
    for pattern in explicit_entity_patterns:
        for raw_term, raw_kind in re.findall(pattern, query, flags=re.IGNORECASE):
            normalized = normalize_constraint_term(raw_term)
            if is_valid_entity_candidate(normalized):
                entity_terms.append(normalized)
                hard_match_terms.append(normalized)
                entity_kind = entity_kind or _ENTITY_FOLLOWING_TERMS.get(raw_kind.lower(), "dataset")

    contextual_entity_patterns = [
        r"\b(?:on|using|use|uses|used|evaluate on|evaluated on|compare with|compared with|against|versus|vs\.?)\s+([A-Za-z0-9][A-Za-z0-9+._/-]{1,40})\b",
    ]
    for pattern in contextual_entity_patterns:
        for raw_term in re.findall(pattern, query, flags=re.IGNORECASE):
            normalized = normalize_constraint_term(raw_term)
            if is_valid_entity_candidate(normalized):
                entity_terms.append(normalized)
                hard_match_terms.append(normalized)

    if not entity_terms:
        for token in re.findall(r"\b(?:[A-Z][A-Za-z0-9._+-]{1,30}|[a-z]*\d[A-Za-z0-9._+-]{0,30})\b", query):
            normalized = normalize_constraint_term(token)
            if is_valid_entity_candidate(normalized):
                entity_terms.append(normalized)
                hard_match_terms.append(normalized)

    if entity_kind is None:
        for surface, mapped in _ENTITY_FOLLOWING_TERMS.items():
            if re.search(rf"\b{re.escape(surface)}\b", lowered):
                entity_kind = mapped
                break

    result_intent = any(
        phrase in lowered
        for phrase in (
            "experiment",
            "experiments",
            "evaluate on",
            "evaluated on",
            "report result",
            "reporting result",
            "results",
            "appendix",
            "table",
            "tables",
            "ablation",
            "metric",
            "metrics",
        )
    )
    if result_intent:
        intent_tags.extend(["experiment", "result_report"])
        required_evidence_types.extend(["experiment", "result"])

    if any(term in lowered for term in ("appendix", "supplementary")):
        intent_tags.append("appendix")
        required_evidence_types.append("appendix")
    if any(term in lowered for term in ("compare", "compared", "comparison", "baseline", "versus", "vs", "against")):
        intent_tags.append("comparison")
        required_evidence_types.append("comparison")
        comparison_terms.extend(
            normalize_constraint_term(term)
            for term in re.findall(r"\b(?:against|versus|vs\.?|compare(?:d)? with|compare(?:d)? to)\s+([A-Za-z0-9][A-Za-z0-9+._/-]{1,40})\b", query, flags=re.IGNORECASE)
        )
    if any(term in lowered for term in ("limitation", "limitations", "failure case", "weakness", "future work")):
        intent_tags.append("limitation")
        required_evidence_types.append("limitation")
    if any(term in lowered for term in ("error analysis", "failure analysis", "qualitative analysis", "failure case")):
        intent_tags.append("error_analysis")
        required_evidence_types.append("error_analysis")
    if any(term in lowered for term in ("latency", "speed", "runtime", "throughput", "memory", "parameter", "compute cost", "efficiency")):
        intent_tags.append("efficiency")
        required_evidence_types.append("efficiency")
    if any(term in lowered for term in ("human evaluation", "human eval", "human judges", "annotators", "user study", "judge model")):
        intent_tags.append("human_eval")
        required_evidence_types.append("human_eval")
    if any(term in lowered for term in ("prompt template", "prompts", "system prompt", "template")):
        intent_tags.append("prompt_template")
        required_evidence_types.append("prompt_template")

    negated_terms.extend(
        normalize_constraint_term(term)
        for term in re.findall(r"\b(?:not|without|except)\s+([A-Za-z0-9][A-Za-z0-9+._/-]{1,40})\b", query, flags=re.IGNORECASE)
    )

    preferred_sections = []
    if result_intent:
        preferred_sections.extend(["experiment", "experiments", "evaluation", "results", "appendix", "benchmark", "table"])
    if "limitation" in lowered or "limitations" in lowered:
        preferred_sections.extend(["limitation", "limitations", "discussion"])
    if "error analysis" in lowered:
        preferred_sections.extend(["error analysis", "analysis"])
    if "baseline" in lowered or "compare" in lowered or "comparison" in lowered:
        preferred_sections.extend(["baseline", "comparison", "ablation", "results"])

    entity_terms = dedupe_preserve(entity_terms)
    hard_match_terms = dedupe_preserve(hard_match_terms)
    preferred_sections = dedupe_preserve(preferred_sections)
    required_evidence_types = dedupe_preserve(required_evidence_types)
    comparison_terms = dedupe_preserve(comparison_terms)
    intent_tags = dedupe_preserve(intent_tags)
    negated_terms = dedupe_preserve(negated_terms)
    entity_context_terms = ENTITY_CONTEXT_MAP.get(entity_kind or "", [])

    return {
        "venues": dedupe_preserve(venues),
        "years": years,
        "tracks": dedupe_preserve(tracks),
        "entity_terms": entity_terms,
        "hard_match_terms": hard_match_terms,
        "entity_kind": entity_kind,
        "entity_context_terms": entity_context_terms,
        "preferred_evidence_sections": preferred_sections,
        "required_evidence_types": required_evidence_types,
        "comparison_terms": comparison_terms,
        "intent_tags": intent_tags,
        "negated_terms": negated_terms,
        "require_entity_match": bool(hard_match_terms),
        "require_experimental_evidence": result_intent,
    }


def build_constraint_graph(query: str, constraints: dict) -> dict:
    lowered = query.lower()
    operators = []
    if re.search(r"\bor\b", lowered):
        operators.append("or")
    if re.search(r"\b(?:not|without|except)\b", lowered):
        operators.append("not")
    nodes = []
    node_index = 0

    def add_node(kind: str, value: str, *, required: bool = True, weight: float = 1.0, metadata: dict | None = None) -> None:
        nonlocal node_index
        if not value:
            return
        node_index += 1
        nodes.append(
            {
                "node_id": f"{kind}_{node_index}",
                "kind": kind,
                "value": str(value),
                "required": required,
                "weight": weight,
                "metadata": metadata or {},
            }
        )

    for venue in constraints.get("venues", []):
        add_node("scope", venue, metadata={"scope_type": "venue"})
    for year in constraints.get("years", []):
        add_node("scope", str(year), metadata={"scope_type": "year"})
    for track in constraints.get("tracks", []):
        add_node("scope", track, metadata={"scope_type": "track"})
    for term in constraints.get("entity_terms", []):
        add_node("entity", term, metadata={"entity_kind": constraints.get("entity_kind")})
    for tag in constraints.get("intent_tags", []):
        add_node("relation", tag, metadata={"source": "intent"})
    for evidence_type in constraints.get("required_evidence_types", []):
        add_node("evidence_type", evidence_type, metadata={"source": "intent"})
    for section in constraints.get("preferred_evidence_sections", []):
        add_node("section", section, weight=0.8, metadata={"source": "prior"})
    for term in constraints.get("comparison_terms", []):
        add_node("entity", term, required=False, weight=0.7, metadata={"entity_kind": "comparison_target"})
    for term in constraints.get("negated_terms", []):
        add_node("negation", term, required=False, metadata={"operator": "not"})

    return {
        "root_operator": "and",
        "operators": operators,
        "nodes": nodes,
    }


def normalize_constraint_term(text: str) -> str:
    cleaned = normalize_whitespace(text).strip(" ,.;:()[]{}")
    return cleaned.lower()
