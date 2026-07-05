from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
import numpy as np

from chemverify.api import app
from chemverify.config import Settings
from chemverify.deep_chat.models import EvidencePack, HypothesisEvidencePack, RetrievedEvidence
from chemverify.indexer import IndexBuilder
from chemverify.models import PaperRecord
from chemverify.search_current import rebuild_search_current
from chemverify.storage import LocalStore


class _FakeOpenAIResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _DeepChatClient:
    def __init__(self, *_, **__) -> None:
        pass

    def __enter__(self) -> "_DeepChatClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> _FakeOpenAIResponse:
        assert json is not None
        system_prompt = json["messages"][0]["content"].lower()
        if "plan multi-turn questions about a single indexed scientific paper" in system_prompt:
            return _FakeOpenAIResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "dialogue_state": {
                                            "active_topic": "reported results on GAIA",
                                            "active_objects": ["GAIA benchmark"],
                                            "question_type": "result_check",
                                            "current_constraints": ["reported results"],
                                            "recent_user_corrections": [],
                                            "discarded_interpretations": [],
                                            "pending_ambiguities": [],
                                            "answer_granularity": "specific",
                                        },
                                        "hypotheses": [
                                            {
                                                "hypothesis_id": "hyp-gaia-results",
                                                "normalized_question": "What results does this paper report on the GAIA benchmark?",
                                                "target_objects": ["GAIA benchmark"],
                                                "target_relation": "result_check",
                                                "required_constraints": ["reported results"],
                                                "preferred_evidence_types": ["paragraph_unit", "table_unit", "chunk_unit"],
                                                "disambiguation_focus": ["benchmark results, not general summary"],
                                                "priority": 0.92,
                                            }
                                        ],
                                        "planner_summary": "The query is specific: find reported GAIA results in the paper.",
                                    }
                                )
                            }
                        }
                    ]
                }
            )
        if "extract grounded facts for questions about a single indexed scientific paper" in system_prompt:
            payload = json_loads(json["messages"][1]["content"])
            packs = payload["hypothesis_evidence_packs"]
            assert "competition_signal" in packs[0]
            assert "discriminative_evidence" in packs[0]
            assert "conflicting_evidence" in packs[0]
            local_evidence = packs[0]["local_evidence"]
            first_local = local_evidence[0]
            second_local = local_evidence[1] if len(local_evidence) > 1 else local_evidence[0]
            return _FakeOpenAIResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "fact_sets": [
                                            {
                                                "hypothesis_id": "hyp-gaia-results",
                                                "normalized_question": "What results does this paper report on the GAIA benchmark?",
                                                "extraction_summary": "The retrieved evidence states that the paper evaluates on GAIA and reports benchmark results.",
                                                "facts": [
                                                    {
                                                        "fact_id": "fact-gaia-1",
                                                        "subject": "paper",
                                                        "relation": "evaluates_on",
                                                        "object": "GAIA benchmark",
                                                        "value": None,
                                                        "unit": None,
                                                        "scope": "experiments",
                                                        "setting": None,
                                                        "evidence_id": first_local["evidence_id"],
                                                        "confidence": 0.95,
                                                    },
                                                    {
                                                        "fact_id": "fact-gaia-2",
                                                        "subject": "paper",
                                                        "relation": "reports_results_for",
                                                        "object": "GAIA benchmark",
                                                        "value": "improved results over strong baselines",
                                                        "unit": None,
                                                        "scope": "results",
                                                        "setting": None,
                                                        "evidence_id": second_local["evidence_id"],
                                                        "confidence": 0.93,
                                                    },
                                                ],
                                                "unresolved_points": [],
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }
            )
        if "answer multi-turn questions about a single indexed scientific paper" in system_prompt:
            payload = json_loads(json["messages"][1]["content"])
            assert "query" in payload
            assert "history" in payload
            assert "competition_signal" in payload["hypothesis_evidence_packs"][0]
            assert "discriminative_evidence" in payload["hypothesis_evidence_packs"][0]
            assert "conflicting_evidence" in payload["hypothesis_evidence_packs"][0]
            fact_sets = payload["hypothesis_fact_sets"]
            used_ids = [item["evidence_id"] for item in fact_sets[0]["facts"][:2]]
            return _FakeOpenAIResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "decision": "answer",
                                        "answer": "The paper evaluates on the GAIA benchmark and reports improved GAIA results over strong baselines.",
                                        "winning_hypothesis_id": "hyp-gaia-results",
                                        "rejected_hypothesis_ids": [],
                                        "uncertainty_note": None,
                                        "used_evidence_ids": used_ids,
                                        "reasoning_summary": "The experiments and results evidence directly mention GAIA evaluation and reported outcomes.",
                                    }
                                )
                            }
                        }
                    ]
                }
            )
        if "verify whether an answer about a single scientific paper" in system_prompt:
            payload = json_loads(json["messages"][1]["content"])
            assert "query" in payload
            assert "history" in payload
            assert "competition_signal" in payload["hypothesis_evidence_packs"][0]
            assert "discriminative_evidence" in payload["hypothesis_evidence_packs"][0]
            assert "conflicting_evidence" in payload["hypothesis_evidence_packs"][0]
            used_ids = payload["generated_answer"]["used_evidence_ids"]
            return _FakeOpenAIResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "support_verdict": "supported",
                                        "alignment_verdict": "aligned",
                                        "interpretation_verdict": "resolved",
                                        "competition_verdict": "distinct_winner",
                                        "strongest_competitor_id": None,
                                        "confidence": 0.96,
                                        "verified_evidence_ids": used_ids,
                                        "failure_reason": None,
                                    }
                                )
                            }
                        }
                    ]
                }
            )
        raise AssertionError(f"Unexpected prompt: {system_prompt}")


def json_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def json_loads(payload: str) -> dict:
    return json.loads(payload)


def _write_mineru_artifacts(
    settings: Settings,
    *,
    paper_id: str,
    title: str,
    sections: list[tuple[str, str]],
) -> Path:
    parse_dir = settings.mineru_output_dir / paper_id / "auto"
    parse_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = settings.root_dir / f"{paper_id}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
    (parse_dir / f"{paper_id}.md").write_text(f"# {title}\n", encoding="utf-8")
    para_blocks = [{"type": "title", "text": title}]
    for heading, text in sections:
        para_blocks.append({"type": "title", "text": heading})
        para_blocks.append({"type": "text", "text": text})
    (parse_dir / f"{paper_id}_middle.json").write_text(
        json_dumps({"pdf_info": [{"page_idx": 0, "para_blocks": para_blocks}]}),
        encoding="utf-8",
    )
    (parse_dir / f"{paper_id}_content_list.json").write_text("[]", encoding="utf-8")
    return pdf_path


def _seed_fixture_data(tmp_path: Path) -> Settings:
    (tmp_path / "config.toml").write_text(
        """
[openai]
enabled = true
base_url = "https://api.openai.com/v1"
model = "gpt-5.4-mini"
""",
        encoding="utf-8",
    )
    settings = Settings.from_env(tmp_path)
    store = LocalStore(settings)
    pdf_path = _write_mineru_artifacts(
        settings,
        paper_id="acl2025.test.3",
        title="Agentic Reasoning and Planning with Large Language Models on GAIA",
        sections=[
            ("1 Introduction", "We study agentic reasoning."),
            ("2 Experiments", "We evaluate on the GAIA benchmark and report results in Table 3."),
            ("3 Results", "GAIA benchmark results show the model improves over strong baselines."),
            ("4 Appendix", "Additional GAIA results and ablations are included."),
        ],
    )
    store.save_raw_papers(
        [
            PaperRecord.model_validate(
                {
                    "paper_id": "acl2025.test.3",
                    "anthology_id": "acl2025.test.3",
                    "title": "Agentic Reasoning and Planning with Large Language Models on GAIA",
                    "authors": ["Katherine Johnson"],
                    "venue": "acl",
                    "year": 2025,
                    "track": "long",
                    "abstract": "We evaluate agentic reasoning systems on the GAIA benchmark and report detailed results.",
                    "url": "https://example.com/3",
                    "local_pdf_path": str(pdf_path),
                }
            )
        ]
    )
    return settings


def _build_and_refresh_search_current(settings: Settings, store: LocalStore) -> None:
    IndexBuilder(settings, store).build()
    _refresh_search_current(settings)


def _refresh_search_current(settings: Settings) -> None:
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    (settings.state_dir / "job_state.json").write_text(
        json.dumps(
            {
                "job_id": "test_job",
                "corpus": settings.corpus.to_dict(),
                "status": "completed",
                "updated_at": "2026-03-28T00:00:00+00:00",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    rebuild_search_current(settings.root_dir)


def _enable_mocked_api(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr("chemverify.deep_chat.service.httpx.Client", _DeepChatClient)
    monkeypatch.setattr("chemverify.encoders.SentenceTransformerEncoder.__init__", _fake_encoder_init)
    monkeypatch.setattr("chemverify.encoders.SentenceTransformerEncoder.encode", _fake_encoder_encode)
    monkeypatch.setattr("chemverify.reranker.CrossEncoderReranker.__init__", _fake_reranker_init)
    monkeypatch.setattr("chemverify.reranker.CrossEncoderReranker.score_pairs", _fake_reranker_scores)


def _fake_encoder_init(self, config) -> None:
    self.config = config
    self._model = object()
    self.backend_name = f"sentence-transformers:{config.model_name}"


def _fake_encoder_encode(self, texts: list[str], *, progress_callback=None) -> np.ndarray:
    matrix = np.zeros((len(texts), 128), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().split():
            column = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % 128
            matrix[row, column] += 1.0
        norm = np.linalg.norm(matrix[row]) + 1e-12
        matrix[row] /= norm
    if progress_callback is not None:
        progress_callback(len(texts), len(texts))
    return matrix


def _fake_reranker_init(self, config) -> None:
    self.config = config
    self._model = object()
    self.backend_name = f"cross-encoder:{config.model_name}"


def _fake_reranker_scores(self, pairs: list[tuple[str, str]]) -> np.ndarray:
    return np.zeros(len(pairs), dtype=np.float32)


def test_deep_chat_api_returns_structured_answer(tmp_path: Path, monkeypatch) -> None:
    _enable_mocked_api(monkeypatch)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/paper",
            json={
                "paper_id": "acl2025.test.3",
                "query": "What does this paper report on the GAIA benchmark?",
                "history": [],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["paper_id"] == "acl2025.test.3"
    assert payload["decision"] == "answer"
    assert "GAIA benchmark" in payload["answer"]
    assert payload["rewritten_query"]
    assert payload["citations"]
    assert payload["citations"][0]["evidence_id"]
    assert payload["evidence"]
    assert payload["verifier"]["support_verdict"] == "supported"


def test_deep_chat_api_returns_409_on_upstream_failure(tmp_path: Path, monkeypatch) -> None:
    class _FailingClient:
        def __init__(self, *_, **__) -> None:
            pass

        def __enter__(self) -> "_FailingClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, headers: dict | None = None, json: dict | None = None):
            request = httpx.Request("POST", url)
            response = httpx.Response(status_code=502, request=request)
            raise httpx.HTTPStatusError("bad gateway", request=request, response=response)

    _enable_mocked_api(monkeypatch)
    monkeypatch.setattr("chemverify.deep_chat.service.httpx.Client", _FailingClient)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/paper",
            json={
                "paper_id": "acl2025.test.3",
                "query": "What does this paper report on the GAIA benchmark?",
                "history": [],
            },
        )

    assert response.status_code == 409
    assert "upstream request failed" in response.json()["detail"]


def test_deep_chat_api_downgrades_to_unsupported_when_verifier_rejects(tmp_path: Path, monkeypatch) -> None:
    class _VerifierRejectClient(_DeepChatClient):
        def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> _FakeOpenAIResponse:
            assert json is not None
            system_prompt = json["messages"][0]["content"].lower()
            if "verify whether an answer about a single scientific paper" not in system_prompt:
                return super().post(url, headers=headers, json=json)
            payload = json_loads(json["messages"][1]["content"])
            used_ids = payload["generated_answer"]["used_evidence_ids"]
            return _FakeOpenAIResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "support_verdict": "unsupported",
                                        "alignment_verdict": "aligned",
                                        "interpretation_verdict": "resolved",
                                        "competition_verdict": "distinct_winner",
                                        "strongest_competitor_id": None,
                                        "confidence": 0.82,
                                        "verified_evidence_ids": used_ids,
                                        "failure_reason": "The cited evidence does not reliably support the claimed result.",
                                    }
                                )
                            }
                        }
                    ]
                }
            )

    _enable_mocked_api(monkeypatch)
    monkeypatch.setattr("chemverify.deep_chat.service.httpx.Client", _VerifierRejectClient)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/paper",
            json={
                "paper_id": "acl2025.test.3",
                "query": "What does this paper report on the GAIA benchmark?",
                "history": [],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] == "unsupported"
    assert "cannot answer this reliably" in payload["answer"].lower()
    assert payload["uncertainty_note"]


def test_deep_chat_api_rejects_invalid_grounding_ids(tmp_path: Path, monkeypatch) -> None:
    class _InvalidGroundingClient(_DeepChatClient):
        def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> _FakeOpenAIResponse:
            assert json is not None
            system_prompt = json["messages"][0]["content"].lower()
            if "answer multi-turn questions about a single indexed scientific paper" not in system_prompt:
                return super().post(url, headers=headers, json=json)
            return _FakeOpenAIResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "decision": "answer",
                                        "answer": "The paper reports GAIA results.",
                                            "winning_hypothesis_id": "hyp-gaia-results",
                                            "rejected_hypothesis_ids": [],
                                            "uncertainty_note": None,
                                            "used_evidence_ids": ["bogus-evidence-id"],
                                            "reasoning_summary": "Invalid grounding on purpose for test coverage.",
                                    }
                                )
                            }
                        }
                    ]
                }
            )

    _enable_mocked_api(monkeypatch)
    monkeypatch.setattr("chemverify.deep_chat.service.httpx.Client", _InvalidGroundingClient)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/paper",
            json={
                "paper_id": "acl2025.test.3",
                "query": "What does this paper report on the GAIA benchmark?",
                "history": [],
            },
        )

    assert response.status_code == 409
    assert "outside the winning interpretation hypothesis" in response.json()["detail"].lower()


def test_deep_chat_api_retries_invalid_structured_output(tmp_path: Path, monkeypatch) -> None:
    class _RetryingClient(_DeepChatClient):
        planner_calls = 0

        def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> _FakeOpenAIResponse:
            assert json is not None
            system_prompt = json["messages"][0]["content"].lower()
            if "plan multi-turn questions about a single indexed scientific paper" not in system_prompt:
                return super().post(url, headers=headers, json=json)
            type(self).planner_calls += 1
            if type(self).planner_calls == 1:
                return _FakeOpenAIResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_dumps(
                                        {
                                            "dialogue_state": {
                                                "active_topic": "reported results on GAIA",
                                                "active_objects": ["GAIA benchmark"],
                                                "question_type": "result_check",
                                                "current_constraints": ["reported results"],
                                                "recent_user_corrections": [],
                                                "discarded_interpretations": [],
                                                "pending_ambiguities": [],
                                                "answer_granularity": "specific",
                                            },
                                            "hypotheses": [
                                                {
                                                    "hypothesis_id": "hyp-gaia-results",
                                                    "normalized_question": "What results does this paper report on the GAIA benchmark?",
                                                    "target_objects": ["GAIA benchmark"],
                                                    "target_relation": "result_check",
                                                    "required_constraints": ["reported results"],
                                                    "preferred_evidence_types": ["not_a_real_type"],
                                                    "disambiguation_focus": [],
                                                    "priority": 0.92,
                                                }
                                            ],
                                        }
                                    )
                                }
                            }
                        ]
                    }
                )
            return super().post(url, headers=headers, json=json)

    _enable_mocked_api(monkeypatch)
    monkeypatch.setattr("chemverify.deep_chat.service.httpx.Client", _RetryingClient)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/paper",
            json={
                "paper_id": "acl2025.test.3",
                "query": "What does this paper report on the GAIA benchmark?",
                "history": [],
            },
        )

    assert response.status_code == 200
    assert _RetryingClient.planner_calls == 2


def test_deep_chat_api_rejects_invalid_fact_grounding_ids(tmp_path: Path, monkeypatch) -> None:
    class _InvalidFactClient(_DeepChatClient):
        def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> _FakeOpenAIResponse:
            assert json is not None
            system_prompt = json["messages"][0]["content"].lower()
            if "extract grounded facts for questions about a single indexed scientific paper" not in system_prompt:
                return super().post(url, headers=headers, json=json)
            return _FakeOpenAIResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "fact_sets": [
                                            {
                                                "hypothesis_id": "hyp-gaia-results",
                                                "normalized_question": "What results does this paper report on the GAIA benchmark?",
                                                "extraction_summary": "Invalid fact grounding on purpose.",
                                                "facts": [
                                                    {
                                                        "fact_id": "fact-invalid",
                                                        "subject": "paper",
                                                        "relation": "reports_results_for",
                                                        "object": "GAIA benchmark",
                                                        "value": "strong results",
                                                        "unit": None,
                                                        "scope": "results",
                                                        "setting": None,
                                                        "evidence_id": "bogus-evidence-id",
                                                        "confidence": 0.99,
                                                    }
                                                ],
                                                "unresolved_points": [],
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }
            )

    _enable_mocked_api(monkeypatch)
    monkeypatch.setattr("chemverify.deep_chat.service.httpx.Client", _InvalidFactClient)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/paper",
            json={
                "paper_id": "acl2025.test.3",
                "query": "What does this paper report on the GAIA benchmark?",
                "history": [],
            },
        )

    assert response.status_code == 409
    assert "fact extractor returned an invalid grounded evidence id" in response.json()["detail"].lower()


def test_deep_chat_api_rejects_cross_hypothesis_answer_grounding(tmp_path: Path, monkeypatch) -> None:
    def _mock_retrieve_for_hypotheses(self, *, paper_id, hypotheses, runtime, query_encoder, reranker):
        assert paper_id == "acl2025.test.cross"
        return [
            HypothesisEvidencePack(
                hypothesis_id="hyp-math-benchmark",
                normalized_question="What result does the paper report on the MATH benchmark?",
                target_relation="result_check",
                target_objects=["MATH benchmark"],
                evidence_pack=EvidencePack(
                    global_evidence=[],
                    local_evidence=[
                        RetrievedEvidence(
                            evidence_id="evidence-benchmark-1",
                            evidence_type="paragraph_unit",
                            score=1.0,
                            source_query="benchmark",
                            heading="Experiments",
                            section_path=["Experiments"],
                            page_start=2,
                            page_end=2,
                            text="We evaluate on the MATH benchmark and report the result in Table 2.",
                            html="",
                            metadata={},
                        )
                    ],
                ),
            ),
            HypothesisEvidencePack(
                hypothesis_id="hyp-math-discussion",
                normalized_question="How does the paper discuss mathematical reasoning more broadly?",
                target_relation="summary",
                target_objects=["mathematical reasoning"],
                evidence_pack=EvidencePack(
                    global_evidence=[],
                    local_evidence=[
                        RetrievedEvidence(
                            evidence_id="evidence-discussion-1",
                            evidence_type="paragraph_unit",
                            score=1.0,
                            source_query="discussion",
                            heading="Introduction",
                            section_path=["Introduction"],
                            page_start=1,
                            page_end=1,
                            text="We discuss mathematical reasoning in language models.",
                            html="",
                            metadata={},
                        )
                    ],
                ),
            ),
        ]

    class _CrossHypothesisGroundingClient(_DeepChatClient):
        def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> _FakeOpenAIResponse:
            assert json is not None
            system_prompt = json["messages"][0]["content"].lower()
            if "plan multi-turn questions about a single indexed scientific paper" in system_prompt:
                return _FakeOpenAIResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_dumps(
                                        {
                                            "dialogue_state": {
                                                "active_topic": "MATH benchmark results",
                                                "active_objects": ["MATH benchmark"],
                                                "question_type": "result_check",
                                                "current_constraints": ["reported results"],
                                                "recent_user_corrections": [],
                                                "discarded_interpretations": [],
                                                "pending_ambiguities": [],
                                                "answer_granularity": "specific",
                                            },
                                            "hypotheses": [
                                                {
                                                    "hypothesis_id": "hyp-math-benchmark",
                                                    "normalized_question": "What result does the paper report on the MATH benchmark?",
                                                    "target_objects": ["MATH benchmark"],
                                                    "target_relation": "result_check",
                                                    "required_constraints": ["reported results"],
                                                    "preferred_evidence_types": ["table_unit", "paragraph_unit", "chunk_unit"],
                                                    "disambiguation_focus": [],
                                                    "priority": 0.95,
                                                },
                                                {
                                                    "hypothesis_id": "hyp-math-discussion",
                                                    "normalized_question": "How does the paper discuss mathematical reasoning more broadly?",
                                                    "target_objects": ["mathematical reasoning"],
                                                    "target_relation": "summary",
                                                    "required_constraints": [],
                                                    "preferred_evidence_types": ["paragraph_unit", "chunk_unit"],
                                                    "disambiguation_focus": [],
                                                    "priority": 0.32,
                                                },
                                            ],
                                        }
                                    )
                                }
                            }
                        ]
                    }
                )
            if "extract grounded facts for questions about a single indexed scientific paper" in system_prompt:
                payload = json_loads(json["messages"][1]["content"])
                packs = payload["hypothesis_evidence_packs"]
                benchmark_pack = next(item for item in packs if item["hypothesis_id"] == "hyp-math-benchmark")
                discussion_pack = next(item for item in packs if item["hypothesis_id"] == "hyp-math-discussion")
                return _FakeOpenAIResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_dumps(
                                        {
                                            "fact_sets": [
                                                {
                                                    "hypothesis_id": "hyp-math-benchmark",
                                                    "normalized_question": "What result does the paper report on the MATH benchmark?",
                                                    "extraction_summary": "Benchmark-level result facts.",
                                                    "facts": [
                                                        {
                                                            "fact_id": "fact-benchmark",
                                                            "subject": "paper",
                                                            "relation": "reports_results_for",
                                                            "object": "MATH benchmark",
                                                            "value": "improved performance over strong baselines",
                                                            "unit": None,
                                                            "scope": "results",
                                                            "setting": None,
                                                            "evidence_id": benchmark_pack["local_evidence"][0]["evidence_id"],
                                                            "confidence": 0.95,
                                                        }
                                                    ],
                                                    "unresolved_points": [],
                                                },
                                                {
                                                    "hypothesis_id": "hyp-math-discussion",
                                                    "normalized_question": "How does the paper discuss mathematical reasoning more broadly?",
                                                    "extraction_summary": "General math discussion facts.",
                                                    "facts": [
                                                        {
                                                            "fact_id": "fact-discussion",
                                                            "subject": "paper",
                                                            "relation": "discusses",
                                                            "object": "mathematical reasoning",
                                                            "value": None,
                                                            "unit": None,
                                                            "scope": "introduction",
                                                            "setting": None,
                                                            "evidence_id": discussion_pack["local_evidence"][0]["evidence_id"],
                                                            "confidence": 0.88,
                                                        }
                                                    ],
                                                    "unresolved_points": [],
                                                },
                                            ]
                                        }
                                    )
                                }
                            }
                        ]
                    }
                )
            if "answer multi-turn questions about a single indexed scientific paper" in system_prompt:
                payload = json_loads(json["messages"][1]["content"])
                discussion_facts = next(item for item in payload["hypothesis_fact_sets"] if item["hypothesis_id"] == "hyp-math-discussion")
                wrong_id = discussion_facts["facts"][0]["evidence_id"]
                return _FakeOpenAIResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_dumps(
                                        {
                                            "decision": "answer",
                                            "answer": "The paper reports a result on the MATH benchmark.",
                                            "winning_hypothesis_id": "hyp-math-benchmark",
                                            "rejected_hypothesis_ids": ["hyp-math-discussion"],
                                            "uncertainty_note": None,
                                            "used_evidence_ids": [wrong_id],
                                            "reasoning_summary": "Intentionally cites evidence from the wrong hypothesis.",
                                        }
                                    )
                                }
                            }
                        ]
                    }
                )
            return super().post(url, headers=headers, json=json)

    _enable_mocked_api(monkeypatch)
    monkeypatch.setattr("chemverify.deep_chat.service.httpx.Client", _CrossHypothesisGroundingClient)
    monkeypatch.setattr(
        "chemverify.deep_chat.retriever.DeepChatRetriever.retrieve_for_hypotheses",
        _mock_retrieve_for_hypotheses,
    )

    (tmp_path / "config.toml").write_text(
        """
[openai]
enabled = true
base_url = "https://api.openai.com/v1"
model = "gpt-5.4-mini"
""",
        encoding="utf-8",
    )
    settings = Settings.from_env(tmp_path)
    store = LocalStore(settings)
    pdf_path = _write_mineru_artifacts(
        settings,
        paper_id="acl2025.test.cross",
        title="Grounded Evaluation on the MATH Benchmark",
        sections=[
            ("1 Introduction", "We discuss mathematical reasoning in language models."),
            ("2 Experiments", "We evaluate on the MATH benchmark and report results in Table 2."),
            ("3 Results", "MATH benchmark results show improved performance over strong baselines."),
        ],
    )
    store.save_raw_papers(
        [
            PaperRecord.model_validate(
                {
                    "paper_id": "acl2025.test.cross",
                    "anthology_id": "acl2025.test.cross",
                    "title": "Grounded Evaluation on the MATH Benchmark",
                    "authors": ["Grace Hopper"],
                    "venue": "acl",
                    "year": 2025,
                    "track": "long",
                    "abstract": "We report evaluation results on the MATH benchmark for mathematical reasoning.",
                    "url": "https://example.com/cross",
                    "local_pdf_path": str(pdf_path),
                }
            )
        ]
    )
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    _build_and_refresh_search_current(settings, store)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/paper",
            json={
                "paper_id": "acl2025.test.cross",
                "query": "What result does the paper report on the MATH benchmark?",
                "history": [],
            },
        )

    assert response.status_code == 409
    assert "outside the winning interpretation hypothesis" in response.json()["detail"].lower()


def test_deep_chat_api_uses_recent_user_correction_to_select_interpretation(tmp_path: Path, monkeypatch) -> None:
    class _CorrectionAwareClient(_DeepChatClient):
        def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> _FakeOpenAIResponse:
            assert json is not None
            system_prompt = json["messages"][0]["content"].lower()
            if "plan multi-turn questions about a single indexed scientific paper" in system_prompt:
                return _FakeOpenAIResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_dumps(
                                        {
                                            "dialogue_state": {
                                                "active_topic": "MATH benchmark results",
                                                "active_objects": ["MATH benchmark"],
                                                "question_type": "result_check",
                                                "current_constraints": ["benchmark results"],
                                                "recent_user_corrections": ["The user means the MATH benchmark, not general mathematical tasks."],
                                                "discarded_interpretations": ["general math task discussion"],
                                                "pending_ambiguities": [],
                                                "answer_granularity": "table_or_value",
                                            },
                                            "hypotheses": [
                                                {
                                                    "hypothesis_id": "hyp-math-task",
                                                    "normalized_question": "How does the paper discuss math tasks in general?",
                                                    "target_objects": ["math tasks"],
                                                    "target_relation": "summary",
                                                    "required_constraints": [],
                                                    "preferred_evidence_types": ["paragraph_unit", "chunk_unit"],
                                                    "disambiguation_focus": ["general discussion"],
                                                    "priority": 0.22,
                                                },
                                                {
                                                    "hypothesis_id": "hyp-math-benchmark",
                                                    "normalized_question": "What result does the paper report on the MATH benchmark?",
                                                    "target_objects": ["MATH benchmark"],
                                                    "target_relation": "result_check",
                                                    "required_constraints": ["reported results"],
                                                    "preferred_evidence_types": ["table_unit", "paragraph_unit", "chunk_unit"],
                                                    "disambiguation_focus": ["benchmark result, not generic math ability"],
                                                    "priority": 0.95,
                                                },
                                            ],
                                            "planner_summary": "The latest user turn explicitly corrects the target object to the MATH benchmark.",
                                        }
                                    )
                                }
                            }
                        ]
                    }
                )
            if "extract grounded facts for questions about a single indexed scientific paper" in system_prompt:
                payload = json_loads(json["messages"][1]["content"])
                packs = payload["hypothesis_evidence_packs"]
                benchmark_pack = next(item for item in packs if item["hypothesis_id"] == "hyp-math-benchmark")
                local_evidence = benchmark_pack["local_evidence"]
                return _FakeOpenAIResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_dumps(
                                        {
                                            "fact_sets": [
                                                {
                                                    "hypothesis_id": "hyp-math-task",
                                                    "normalized_question": "How does the paper discuss math tasks in general?",
                                                    "extraction_summary": "The paper discusses mathematical reasoning, but this is not the corrected target.",
                                                    "facts": [],
                                                    "unresolved_points": ["Generic task-level reading, not the corrected benchmark interpretation."],
                                                },
                                                {
                                                    "hypothesis_id": "hyp-math-benchmark",
                                                    "normalized_question": "What result does the paper report on the MATH benchmark?",
                                                    "extraction_summary": "The evidence directly supports benchmark-level evaluation on MATH.",
                                                    "facts": [
                                                        {
                                                            "fact_id": "fact-math-1",
                                                            "subject": "paper",
                                                            "relation": "evaluates_on",
                                                            "object": "MATH benchmark",
                                                            "value": None,
                                                            "unit": None,
                                                            "scope": "experiments",
                                                            "setting": None,
                                                            "evidence_id": local_evidence[0]["evidence_id"],
                                                            "confidence": 0.95,
                                                        },
                                                        {
                                                            "fact_id": "fact-math-2",
                                                            "subject": "paper",
                                                            "relation": "reports_results_for",
                                                            "object": "MATH benchmark",
                                                            "value": "improved performance over strong baselines",
                                                            "unit": None,
                                                            "scope": "results",
                                                            "setting": None,
                                                            "evidence_id": local_evidence[1]["evidence_id"] if len(local_evidence) > 1 else local_evidence[0]["evidence_id"],
                                                            "confidence": 0.93,
                                                        },
                                                    ],
                                                    "unresolved_points": [],
                                                },
                                            ]
                                        }
                                    )
                                }
                            }
                        ]
                    }
                )
            if "answer multi-turn questions about a single indexed scientific paper" in system_prompt:
                payload = json_loads(json["messages"][1]["content"])
                fact_sets = payload["hypothesis_fact_sets"]
                benchmark_facts = next(item for item in fact_sets if item["hypothesis_id"] == "hyp-math-benchmark")
                used_ids = [item["evidence_id"] for item in benchmark_facts["facts"][:2]]
                return _FakeOpenAIResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_dumps(
                                        {
                                            "decision": "answer",
                                            "answer": "The paper reports a result on the MATH benchmark in its experiments section.",
                                            "winning_hypothesis_id": "hyp-math-benchmark",
                                            "rejected_hypothesis_ids": ["hyp-math-task"],
                                            "uncertainty_note": None,
                                            "used_evidence_ids": used_ids,
                                            "reasoning_summary": "The latest correction resolves the ambiguity toward the benchmark interpretation.",
                                        }
                                    )
                                }
                            }
                        ]
                    }
                )
            if "verify whether an answer about a single scientific paper" in system_prompt:
                payload = json_loads(json["messages"][1]["content"])
                used_ids = payload["generated_answer"]["used_evidence_ids"]
                return _FakeOpenAIResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_dumps(
                                        {
                                        "support_verdict": "supported",
                                        "alignment_verdict": "aligned",
                                        "interpretation_verdict": "resolved",
                                        "competition_verdict": "distinct_winner",
                                        "strongest_competitor_id": None,
                                        "confidence": 0.97,
                                        "verified_evidence_ids": used_ids,
                                        "failure_reason": None,
                                    }
                                )
                                }
                            }
                        ]
                    }
                )
            return super().post(url, headers=headers, json=json)

    _enable_mocked_api(monkeypatch)
    monkeypatch.setattr("chemverify.deep_chat.service.httpx.Client", _CorrectionAwareClient)

    (tmp_path / "config.toml").write_text(
        """
[openai]
enabled = true
base_url = "https://api.openai.com/v1"
model = "gpt-5.4-mini"
""",
        encoding="utf-8",
    )
    settings = Settings.from_env(tmp_path)
    store = LocalStore(settings)
    pdf_path = _write_mineru_artifacts(
        settings,
        paper_id="acl2025.test.math",
        title="Grounded Evaluation on the MATH Benchmark",
        sections=[
            ("1 Introduction", "We study mathematical reasoning in language models."),
            ("2 Experiments", "We evaluate on the MATH benchmark and summarize the reported result in Table 2."),
            ("3 Results", "MATH benchmark results show improved performance over strong baselines."),
        ],
    )
    store.save_raw_papers(
        [
            PaperRecord.model_validate(
                {
                    "paper_id": "acl2025.test.math",
                    "anthology_id": "acl2025.test.math",
                    "title": "Grounded Evaluation on the MATH Benchmark",
                    "authors": ["Grace Hopper"],
                    "venue": "acl",
                    "year": 2025,
                    "track": "long",
                    "abstract": "We report evaluation results on the MATH benchmark for mathematical reasoning.",
                    "url": "https://example.com/math",
                    "local_pdf_path": str(pdf_path),
                }
            )
        ]
    )
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    _build_and_refresh_search_current(settings, store)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/paper",
            json={
                "paper_id": "acl2025.test.math",
                "query": "I mean the MATH benchmark, not math tasks. What result does it report there?",
                "history": [
                    {"role": "user", "content": "How does this paper do on math?"},
                    {"role": "assistant", "content": "It discusses mathematical reasoning broadly."},
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] == "answer"
    assert "MATH benchmark" in payload["rewritten_query"]
    assert payload["verifier"]["interpretation_verdict"] == "resolved"


def test_deep_chat_api_asks_for_clarification_when_verifier_sees_no_clear_winner(tmp_path: Path, monkeypatch) -> None:
    class _NoClearWinnerClient(_DeepChatClient):
        def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> _FakeOpenAIResponse:
            assert json is not None
            system_prompt = json["messages"][0]["content"].lower()
            if "verify whether an answer about a single scientific paper" not in system_prompt:
                return super().post(url, headers=headers, json=json)
            payload = json_loads(json["messages"][1]["content"])
            used_ids = payload["generated_answer"]["used_evidence_ids"]
            return _FakeOpenAIResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "support_verdict": "supported",
                                        "alignment_verdict": "aligned",
                                        "interpretation_verdict": "resolved",
                                        "competition_verdict": "no_clear_winner",
                                        "strongest_competitor_id": "hyp-alt-reading",
                                        "confidence": 0.55,
                                        "verified_evidence_ids": used_ids,
                                        "failure_reason": "The evidence supports multiple competing interpretations with no clear winner.",
                                    }
                                )
                            }
                        }
                    ]
                }
            )

    _enable_mocked_api(monkeypatch)
    monkeypatch.setattr("chemverify.deep_chat.service.httpx.Client", _NoClearWinnerClient)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    with TestClient(app) as client:
        response = client.post(
            "/api/chat/paper",
            json={
                "paper_id": "acl2025.test.3",
                "query": "What does this paper report on the GAIA benchmark?",
                "history": [],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] == "ask_clarification"
    assert "clarification" in payload["answer"].lower()
    assert payload["verifier"]["competition_verdict"] == "no_clear_winner"
