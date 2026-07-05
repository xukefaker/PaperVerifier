from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from .config import Settings
from .models import EvidenceBucket, QueryAspect, QueryPlan, ScopeConstraints, TokenUsage, VerifierRubric
from .provider import require_openai_model
from .utils import TRACK_ALIASES, VENUE_ALIASES

PLANNER_SYSTEM_PROMPT = """You are planning scholarly full-paper retrieval.
Return strict JSON with exactly these top-level keys:
- global_query: string
- scope_constraints: {venues: string[], years: number[], tracks: string[]}
- entity_terms: string[]
- exact_phrases: string[]
- aspect_queries: [{aspect_id: string, query: string, weight: number}]
- verifier_rubric: {must_satisfy: string[], should_satisfy: string[], rejection_rules: string[]}
- evidence_buckets: [{bucket_id: string, description: string, queries: string[], target_chunks: number}]

Requirements:
- aspect_queries must contain 3 to 5 items
- weights should sum close to 1.0
- scope_constraints should contain only explicit scope from the user query
- exact_phrases should contain literal phrases worth exact matching in full text
- evidence_buckets must be driven by query intent, not a fixed template
- if the query asks for experiments, results, comparisons, metrics, or reported findings, create a bucket for that evidence
- if the query is about methods or mechanisms, create method-oriented evidence buckets instead
- do not invent paper titles, authors, or venues not stated in the query
- use concise retrieval phrasing
"""


@dataclass(slots=True)
class PlannerResult:
    plan: QueryPlan
    usage: TokenUsage


class QueryPlanner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def plan(self, query: str) -> PlannerResult:
        if not self.settings.openai_enabled or not self.settings.openai_api_key:
            raise RuntimeError("Query parser requires an enabled API model.")
        model = require_openai_model(self.settings)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.settings.request_timeout) as client:
            response = client.post(f"{self.settings.openai_base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()

        raw = response.json()
        content = raw["choices"][0]["message"]["content"]
        plan = self._validate_plan(query, json.loads(content))
        usage = _usage_from_response(self.settings, raw)
        return PlannerResult(plan=plan, usage=usage)

    def _validate_plan(self, query: str, payload: dict) -> QueryPlan:
        aspect_payloads = payload.get("aspect_queries")
        if not isinstance(aspect_payloads, list) or not (3 <= len(aspect_payloads) <= 5):
            raise RuntimeError("Query parser returned an invalid number of aspect queries.")
        bucket_payloads = payload.get("evidence_buckets")
        if not isinstance(bucket_payloads, list) or not bucket_payloads:
            raise RuntimeError("Query parser returned no evidence buckets.")

        aspects: list[QueryAspect] = []
        seen_aspect_ids: set[str] = set()
        for index, item in enumerate(aspect_payloads, start=1):
            aspect_id = str(item["aspect_id"]).strip()
            aspect_query = str(item["query"]).strip()
            if not aspect_id or not aspect_query:
                raise RuntimeError("Query parser returned an invalid aspect query.")
            lowered_aspect_id = aspect_id.lower()
            if lowered_aspect_id in seen_aspect_ids:
                raise RuntimeError("Query parser returned duplicate aspect_id values.")
            seen_aspect_ids.add(lowered_aspect_id)
            aspects.append(
                QueryAspect(
                    aspect_id=aspect_id,
                    query=aspect_query,
                    weight=float(item["weight"]),
                )
            )
        weight_sum = sum(max(aspect.weight, 0.0) for aspect in aspects)
        if any(aspect.weight < 0 for aspect in aspects):
            raise RuntimeError("Query parser returned a negative aspect weight.")
        if weight_sum <= 0:
            raise RuntimeError("Query parser returned non-positive aspect weights.")
        aspects = [aspect.model_copy(update={"weight": aspect.weight / weight_sum}) for aspect in aspects]

        buckets: list[EvidenceBucket] = []
        seen_bucket_ids: set[str] = set()
        for item in bucket_payloads:
            raw_queries = item.get("queries")
            if not isinstance(raw_queries, list):
                raise RuntimeError("Query parser returned an invalid evidence bucket.")
            bucket_id = str(item["bucket_id"]).strip()
            description = str(item["description"]).strip()
            queries = _dedupe_strs(raw_queries)
            if not bucket_id or not description or not queries:
                raise RuntimeError("Query parser returned an invalid evidence bucket.")
            lowered_bucket_id = bucket_id.lower()
            if lowered_bucket_id in seen_bucket_ids:
                raise RuntimeError("Query parser returned duplicate bucket_id values.")
            seen_bucket_ids.add(lowered_bucket_id)
            bucket = EvidenceBucket(
                bucket_id=bucket_id,
                description=description,
                queries=queries,
                target_chunks=int(item["target_chunks"]),
            )
            if bucket.target_chunks <= 0:
                raise RuntimeError("Query parser returned an invalid evidence bucket.")
            buckets.append(bucket)

        scope_constraints = _normalize_scope_constraints(payload.get("scope_constraints") or {})
        plan = QueryPlan(
            mode="api_llm",
            user_query=query,
            global_query=str(payload["global_query"]).strip(),
            scope_constraints=scope_constraints,
            entity_terms=_dedupe_strs(payload.get("entity_terms") or []),
            exact_phrases=_dedupe_strs(payload.get("exact_phrases") or []),
            aspect_queries=aspects,
            verifier_rubric=VerifierRubric.model_validate(payload.get("verifier_rubric") or {}),
            evidence_buckets=buckets,
        )
        if not plan.global_query:
            raise RuntimeError("Query parser returned an empty global_query.")
        return plan


def _dedupe_strs(values: list[object]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(text)
    return deduped


def _normalize_scope_constraints(payload: dict) -> ScopeConstraints:
    raw = ScopeConstraints.model_validate(payload)
    venues: list[str] = []
    tracks: list[str] = []
    years: list[int] = []

    for venue in raw.venues:
        normalized = str(venue).strip().lower()
        if not normalized:
            continue
        canonical = VENUE_ALIASES.get(normalized)
        if canonical:
            venues.append(canonical)
            continue
        tokens = normalized.replace("/", " ").replace("-", " ").split()
        for token in tokens:
            if token in VENUE_ALIASES:
                venues.append(VENUE_ALIASES[token])
        for token in tokens:
            if token.isdigit() and len(token) == 4:
                years.append(int(token))

    for year in raw.years:
        years.append(int(year))

    for track in raw.tracks:
        normalized = str(track).strip().lower()
        if not normalized:
            continue
        canonical = TRACK_ALIASES.get(normalized)
        if canonical:
            tracks.append(canonical)
            continue
        tokens = normalized.replace("/", " ").replace("-", " ").split()
        for token in tokens:
            if token in TRACK_ALIASES:
                tracks.append(TRACK_ALIASES[token])

    return ScopeConstraints(
        venues=_dedupe_strs(venues),
        years=sorted(set(years)),
        tracks=_dedupe_strs(tracks),
    )


def _usage_from_response(settings: Settings, payload: dict) -> TokenUsage:
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
