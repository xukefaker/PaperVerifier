from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest
from typer.testing import CliRunner

import chemverify.cli as cli_module
from chemverify.acl_anthology import ACLAnthologyIngestor, _ListingEntry
from chemverify.api import app
from chemverify.cancel import CancelRequested
from chemverify.cli import app as cli_app
from chemverify.config import CorpusSpec, Settings
from chemverify.indexer import IndexBuilder
from chemverify.pdf_parser import PDFParser
from chemverify.planner import QueryPlanner
from chemverify.search_current import rebuild_search_current
from chemverify.storage import LocalStore


class _FakeOpenAIResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeOpenAIClient:
    def __init__(self, *_, **__) -> None:
        pass

    def __enter__(self) -> "_FakeOpenAIClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> _FakeOpenAIResponse:
        assert json is not None
        system_prompt = json["messages"][0]["content"]
        if "planning scholarly full-paper retrieval" in system_prompt.lower():
            planner_payload = {
                "global_query": "ACL 2025 papers that evaluate on GAIA and report GAIA results",
                "scope_constraints": {"venues": ["ACL 2025"], "years": [2025], "tracks": []},
                "entity_terms": ["GAIA"],
                "exact_phrases": ["GAIA benchmark", "GAIA results"],
                "aspect_queries": [
                    {"aspect_id": "aspect_entity", "query": "GAIA benchmark usage", "weight": 0.34},
                    {"aspect_id": "aspect_result", "query": "results on GAIA", "weight": 0.33},
                    {"aspect_id": "aspect_eval", "query": "evaluation on GAIA", "weight": 0.33},
                ],
                "verifier_rubric": {
                    "must_satisfy": [
                        "the paper is within ACL 2025 scope",
                        "GAIA refers to the benchmark or dataset rather than the paper's own method",
                        "the paper reports experimental use or results on GAIA",
                    ],
                    "should_satisfy": ["evidence should come from experiments results or appendix sections when available"],
                    "rejection_rules": ["reject papers where GAIA is the method or system name"],
                },
                "evidence_buckets": [
                    {
                        "bucket_id": "entity",
                        "description": "show what role GAIA plays in the paper",
                        "queries": ["GAIA", "GAIA benchmark"],
                        "target_chunks": 2,
                    },
                    {
                        "bucket_id": "result",
                        "description": "show evaluation or reported results involving GAIA",
                        "queries": ["results on GAIA", "evaluation on GAIA"],
                        "target_chunks": 2,
                    },
                    {
                        "bucket_id": "global",
                        "description": "single globally relevant evidence chunk",
                        "queries": ["ACL 2025 papers that evaluate on GAIA and report GAIA results"],
                        "target_chunks": 1,
                    },
                ],
            }
            return _FakeOpenAIResponse(
                {
                    "choices": [{"message": {"content": json_dumps(planner_payload)}}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 220, "total_tokens": 340},
                }
            )
        user_payload = json_loads(json["messages"][1]["content"])
        paper_id = user_payload["paper"]["paper_id"]
        if paper_id == "acl2025.test.3":
            verifier_payload = {
                "verdict": "satisfied",
                "entity_role": "dataset_or_benchmark",
                "satisfied_constraints": [
                    "paper is in ACL 2025 scope",
                    "GAIA is used as benchmark",
                    "the paper reports GAIA results",
                ],
                "missing_constraints": [],
                "confidence": 0.97,
                "rationale": "The evidence chunks explicitly state evaluation on the GAIA benchmark and report GAIA results.",
            }
        elif paper_id in {"acl2025.test.4", "acl2025.test.5"}:
            verifier_payload = {
                "verdict": "rejected",
                "entity_role": "method_or_system",
                "satisfied_constraints": ["paper is in ACL 2025 scope"],
                "missing_constraints": ["GAIA is not the requested benchmark or dataset"],
                "confidence": 0.91,
                "rationale": "The evidence shows GAIA is the paper's own framework or strategy planner rather than the benchmark.",
            }
        else:
            verifier_payload = {
                "verdict": "partial",
                "entity_role": "ambiguous_or_other",
                "satisfied_constraints": ["paper is in ACL 2025 scope"],
                "missing_constraints": ["the evidence does not clearly show reported GAIA results"],
                "confidence": 0.42,
                "rationale": "The paper is within scope but the supplied chunks do not fully establish the requested GAIA experimental evidence.",
            }
        return _FakeOpenAIResponse(
            {
                "choices": [{"message": {"content": json_dumps(verifier_payload)}}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 120, "total_tokens": 1020},
            }
        )


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
    settings = Settings.from_env(tmp_path)
    store = LocalStore(settings)
    paper_specs = [
        {
            "paper_id": "acl2025.test.1",
            "anthology_id": "acl2025.test.1",
            "title": "Facet-Aware Retrieval for Scientific Papers",
            "authors": ["Ada Lovelace"],
            "venue": "acl",
            "year": 2025,
            "track": "long",
            "abstract": "We retrieve papers using method and evaluation facets.",
            "url": "https://example.com/1",
            "sections": [
                ("1 Introduction", "Facet-aware retrieval handles detailed scientific queries."),
                ("2 Method", "We build section priors for methods and experiments."),
                ("3 Experiments", "We compare against BM25 and dense baselines."),
                ("4 Limitations", "The system still misses appendix evidence."),
            ],
        },
        {
            "paper_id": "acl2025.test.2",
            "anthology_id": "acl2025.test.2",
            "title": "Benchmarking Citation Graph Retrieval",
            "authors": ["Grace Hopper"],
            "venue": "acl",
            "year": 2025,
            "track": "findings",
            "abstract": "A benchmark paper on citation graph retrieval.",
            "url": "https://example.com/2",
            "sections": [
                ("1 Introduction", "Citation graph retrieval focuses on relation-aware search."),
                ("2 Benchmark", "We release a benchmark for retrieval agents."),
                ("3 Results", "The benchmark covers path and relation queries."),
            ],
        },
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
            "sections": [
                ("1 Introduction", "We study agentic reasoning."),
                ("2 Experiments", "We evaluate on the GAIA benchmark and report results in Table 3."),
                ("3 Results", "GAIA benchmark results show the model improves over strong baselines."),
                ("4 Appendix", "Additional GAIA results and ablations are included."),
            ],
        },
        {
            "paper_id": "acl2025.test.4",
            "anthology_id": "acl2025.test.4",
            "title": "GAIA: A General Agent Interface Architecture",
            "authors": ["Barbara Liskov"],
            "venue": "acl",
            "year": 2025,
            "track": "long",
            "abstract": "We propose a framework called GAIA for tool-using agents.",
            "url": "https://example.com/4",
            "sections": [
                ("1 Introduction", "GAIA is a new agent framework."),
                ("2 Method", "We describe the GAIA architecture and implementation."),
                ("3 Results", "We compare our method on internal tasks without using the GAIA benchmark."),
            ],
        },
        {
            "paper_id": "acl2025.test.5",
            "anthology_id": "acl2025.test.5",
            "title": "GAIA: Strategic Planning for Adversarial Dialogue",
            "authors": ["Donald Knuth"],
            "venue": "acl",
            "year": 2025,
            "track": "long",
            "abstract": "We propose GAIA as a strategy planner and report experimental gains in three applications.",
            "url": "https://example.com/5",
            "sections": [
                ("1 Introduction", "We introduce GAIA as a strategy planner for adversarial dialogue."),
                ("2 Experiments", "We evaluate GAIA in three applications and compare against strong baselines."),
                ("3 Results", "Experimental results show GAIA performs strongly across applications."),
            ],
        },
    ]
    from chemverify.models import PaperRecord

    papers = []
    for spec in paper_specs:
        pdf_path = _write_mineru_artifacts(
            settings,
            paper_id=spec["paper_id"],
            title=spec["title"],
            sections=spec.pop("sections"),
        )
        papers.append(PaperRecord.model_validate({**spec, "local_pdf_path": str(pdf_path)}))
    store.save_raw_papers(papers)
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
    monkeypatch.setattr("chemverify.planner.httpx.Client", _FakeOpenAIClient)
    monkeypatch.setattr("chemverify.search.httpx.Client", _FakeOpenAIClient)
    monkeypatch.setattr("chemverify.encoders.SentenceTransformerEncoder.__init__", _fake_encoder_init)
    monkeypatch.setattr("chemverify.encoders.SentenceTransformerEncoder.encode", _fake_encoder_encode)
    monkeypatch.setattr("chemverify.reranker.CrossEncoderReranker.__init__", _fake_reranker_init)
    monkeypatch.setattr("chemverify.reranker.CrossEncoderReranker.score_pairs", _fake_reranker_scores)


def _write_image_fixture(settings: Settings, paper_id: str, image_name: str) -> Path:
    image_path = settings.mineru_output_dir / paper_id / "txt" / "images" / image_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake-image-bytes")
    return image_path


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


def test_query_planner_returns_strict_plan(tmp_path: Path, monkeypatch) -> None:
    _enable_mocked_api(monkeypatch)
    settings = Settings.from_env(tmp_path)

    plan = QueryPlanner(settings).plan("Find ACL 2025 papers that evaluate on GAIA and report GAIA results").plan

    assert plan.mode == "api_llm"
    assert plan.scope_constraints.venues == ["acl"]
    assert plan.scope_constraints.years == [2025]
    assert plan.entity_terms == ["GAIA"]
    assert len(plan.aspect_queries) == 3
    assert {bucket.bucket_id for bucket in plan.evidence_buckets} == {"entity", "result", "global"}


def test_query_planner_rejects_blank_bucket_queries_and_duplicate_ids(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    planner = QueryPlanner(settings)

    with pytest.raises(RuntimeError, match="duplicate aspect_id"):
        planner._validate_plan(
            "demo query",
            {
                "global_query": "demo",
                "scope_constraints": {},
                "entity_terms": [],
                "exact_phrases": [],
                "aspect_queries": [
                    {"aspect_id": "aspect_a", "query": "first aspect", "weight": 0.34},
                    {"aspect_id": "aspect_a", "query": "second aspect", "weight": 0.33},
                    {"aspect_id": "aspect_c", "query": "third aspect", "weight": 0.33},
                ],
                "verifier_rubric": {},
                "evidence_buckets": [
                    {"bucket_id": "entity", "description": "entity evidence", "queries": ["GAIA"], "target_chunks": 1}
                ],
            },
        )

    with pytest.raises(RuntimeError, match="invalid evidence bucket"):
        planner._validate_plan(
            "demo query",
            {
                "global_query": "demo",
                "scope_constraints": {},
                "entity_terms": [],
                "exact_phrases": [],
                "aspect_queries": [
                    {"aspect_id": "aspect_a", "query": "first aspect", "weight": 0.34},
                    {"aspect_id": "aspect_b", "query": "second aspect", "weight": 0.33},
                    {"aspect_id": "aspect_c", "query": "third aspect", "weight": 0.33},
                ],
                "verifier_rubric": {},
                "evidence_buckets": [
                    {"bucket_id": "bucket_a", "description": "entity evidence", "queries": [""], "target_chunks": 1}
                ],
            },
        )

    with pytest.raises(RuntimeError, match="negative aspect weight"):
        planner._validate_plan(
            "demo query",
            {
                "global_query": "demo",
                "scope_constraints": {},
                "entity_terms": [],
                "exact_phrases": [],
                "aspect_queries": [
                    {"aspect_id": "aspect_a", "query": "first aspect", "weight": 0.50},
                    {"aspect_id": "aspect_b", "query": "second aspect", "weight": -0.10},
                    {"aspect_id": "aspect_c", "query": "third aspect", "weight": 0.60},
                ],
                "verifier_rubric": {},
                "evidence_buckets": [
                    {"bucket_id": "entity", "description": "entity evidence", "queries": ["GAIA"], "target_chunks": 1}
                ],
            },
        )


def test_search_returns_grouped_results_and_trace(tmp_path: Path, monkeypatch) -> None:
    _enable_mocked_api(monkeypatch)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["search", "--query", "Find ACL 2025 papers that evaluate on GAIA and report GAIA results"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)

    assert payload["satisfied"][0]["paper_id"] == "acl2025.test.3"
    assert payload["satisfied"][0]["evidence_chunks"]["entity"]
    assert payload["satisfied"][0]["evidence_chunks"]["result"]
    assert any(item["paper_id"] in {"acl2025.test.4", "acl2025.test.5"} for item in payload["rejected"])
    assert payload["trace_id"]

    trace = LocalStore(settings, root_dir=settings.search_current_dir).load_trace(payload["trace_id"])
    assert trace is not None
    assert trace.final_results["satisfied"][0].paper_id == "acl2025.test.3"
    assert trace.evidence_packs["acl2025.test.3"]["result"]


def test_index_builder_honors_max_papers(tmp_path: Path, monkeypatch) -> None:
    _enable_mocked_api(monkeypatch)
    settings = _seed_fixture_data(tmp_path)
    store = LocalStore(settings)

    summary = IndexBuilder(settings, store).build(max_papers=2)

    assert summary.total_papers == 2
    assert summary.indexed_papers == 2
    stored = store.load_papers()
    assert len(stored) == 2
    assert [paper.paper_id for paper in stored] == ["acl2025.test.1", "acl2025.test.2"]
    state = store.load_index_state()
    assert state["total_papers"] == 2
    assert state["indexed_papers"] == 2


def test_index_command_uses_active_corpus_for_demo_papers(tmp_path: Path, monkeypatch) -> None:
    from chemverify.models import PaperRecord

    monkeypatch.setattr(cli_module, "PROJECT_ROOT", str(tmp_path))
    settings = Settings.from_env(tmp_path, corpus=CorpusSpec.from_values("acl", 2025, "long"))
    settings.ensure_dirs()
    settings.active_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    settings.active_corpus_path.write_text(
        json.dumps(settings.corpus.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pdf_path = settings.pdf_dir / "acl" / "2025" / "long" / "demo.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")
    LocalStore(settings).save_raw_papers(
        [
            PaperRecord(
                paper_id="2025.acl-long.demo",
                title="Demo ACL Paper",
                venue="acl",
                year=2025,
                track="long",
                url=pdf_path.as_uri(),
                pdf_url=pdf_path.as_uri(),
                local_pdf_path=str(pdf_path),
            )
        ]
    )

    seen: dict[str, tuple[str, int, str]] = {}

    class _Summary:
        indexed_papers = 1

        def model_dump(self) -> dict[str, int]:
            return {"indexed_papers": self.indexed_papers}

    class _FakeIndexBuilder:
        def __init__(self, builder_settings: Settings, store: LocalStore, *, cancel_check=None, progress=None) -> None:
            self.settings = builder_settings
            seen["corpus"] = (
                builder_settings.corpus.venue,
                builder_settings.corpus.year,
                builder_settings.corpus.track,
            )

        def load_paper_ids(self, paper_id_file: Path) -> list[str]:
            return []

        def build(self, max_papers: int | None = None, paper_ids: list[str] | None = None) -> _Summary:
            assert max_papers == 1
            assert paper_ids is None
            self.settings.current_release_path.mkdir(parents=True, exist_ok=True)
            (self.settings.current_release_path / "index-state.json").write_text("{}", encoding="utf-8")
            return _Summary()

    monkeypatch.setattr(cli_module, "IndexBuilder", _FakeIndexBuilder)
    monkeypatch.setattr(
        cli_module,
        "rebuild_search_current",
        lambda root, corpora=None: {"corpora": [item.to_dict() for item in (corpora or [])]},
    )

    result = CliRunner().invoke(cli_app, ["index", "--skip-parse", "--max-papers", "1"])

    assert result.exit_code == 0, result.stdout
    assert seen["corpus"] == ("acl", 2025, "long")
    assert not list((settings.data_dir / ".runs").glob("index-*"))


def test_index_command_limits_mineru_parse_to_max_papers(tmp_path: Path, monkeypatch) -> None:
    from chemverify.models import PaperRecord
    import chemverify.mineru_pipeline as mineru_module

    monkeypatch.setattr(cli_module, "PROJECT_ROOT", str(tmp_path))
    settings = Settings.from_env(tmp_path, corpus=CorpusSpec.from_values("personal", 2026, "library"))
    settings.ensure_dirs()
    pdfs = []
    for index in range(2):
        pdf_path = tmp_path / f"demo-{index}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        pdfs.append(pdf_path)
    LocalStore(settings).save_raw_papers(
        [
            PaperRecord(
                paper_id=f"personal-demo-{index}",
                title=f"Demo Paper {index}",
                venue="personal",
                year=2026,
                track="library",
                url=pdf_path.as_uri(),
                pdf_url=pdf_path.as_uri(),
                local_pdf_path=str(pdf_path),
            )
            for index, pdf_path in enumerate(pdfs)
        ]
    )

    seen: dict[str, list[str]] = {}

    def _fake_run_mineru_pipeline(**kwargs) -> dict[str, int]:
        seen["parsed"] = [paper.paper_id for paper in kwargs["papers"]]
        return {"processed": len(kwargs["papers"]), "failed": 0, "skipped_failed": 0}

    class _Summary:
        indexed_papers = 1

        def model_dump(self) -> dict[str, int]:
            return {"indexed_papers": self.indexed_papers}

    class _FakeIndexBuilder:
        def __init__(self, builder_settings: Settings, store: LocalStore, *, cancel_check=None, progress=None) -> None:
            self.settings = builder_settings

        def load_paper_ids(self, paper_id_file: Path) -> list[str]:
            return []

        def build(self, max_papers: int | None = None, paper_ids: list[str] | None = None) -> _Summary:
            assert max_papers == 1
            self.settings.current_release_path.mkdir(parents=True, exist_ok=True)
            return _Summary()

    monkeypatch.setattr(mineru_module, "run_mineru_pipeline", _fake_run_mineru_pipeline)
    monkeypatch.setattr(cli_module, "IndexBuilder", _FakeIndexBuilder)
    monkeypatch.setattr(cli_module, "rebuild_search_current", lambda root, corpora=None: {"corpora": []})

    result = CliRunner().invoke(cli_app, ["index", "--year", "2026", "--max-papers", "1"])

    assert result.exit_code == 0, result.stdout
    assert seen["parsed"] == ["personal-demo-0"]
    assert not list((settings.data_dir / ".runs").glob("index-*"))


def test_index_command_cleans_staged_files_on_cancel(tmp_path: Path, monkeypatch) -> None:
    from chemverify.models import PaperRecord

    monkeypatch.setattr(cli_module, "PROJECT_ROOT", str(tmp_path))
    settings = Settings.from_env(tmp_path, corpus=CorpusSpec.from_values("personal", 2026, "library"))
    settings.ensure_dirs()
    pdf_path = tmp_path / "demo.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    LocalStore(settings).save_raw_papers(
        [
            PaperRecord(
                paper_id="personal-demo",
                title="Demo Paper",
                venue="personal",
                year=2026,
                track="library",
                url=pdf_path.as_uri(),
                pdf_url=pdf_path.as_uri(),
                local_pdf_path=str(pdf_path),
            )
        ]
    )

    class _CancelingIndexBuilder:
        def __init__(self, builder_settings: Settings, store: LocalStore, *, cancel_check=None, progress=None) -> None:
            self.settings = builder_settings

        def load_paper_ids(self, paper_id_file: Path) -> list[str]:
            return []

        def build(self, max_papers: int | None = None, paper_ids: list[str] | None = None):
            self.settings.current_release_path.mkdir(parents=True, exist_ok=True)
            (self.settings.current_release_path / "partial.txt").write_text("partial", encoding="utf-8")
            raise CancelRequested("Canceled by user.")

    monkeypatch.setattr(cli_module, "IndexBuilder", _CancelingIndexBuilder)

    result = CliRunner().invoke(cli_app, ["index", "--year", "2026", "--skip-parse", "--max-papers", "1"])

    assert result.exit_code == 1, result.stdout
    assert "Index canceled" in f"{result.stdout}\n{result.stderr}"
    assert not settings.current_release_path.exists()
    assert not list((settings.data_dir / ".runs").glob("index-*"))


def test_index_builder_honors_explicit_paper_id_order(tmp_path: Path, monkeypatch) -> None:
    _enable_mocked_api(monkeypatch)
    settings = _seed_fixture_data(tmp_path)
    store = LocalStore(settings)
    paper_id_file = tmp_path / "paper_ids.txt"
    paper_id_file.write_text("acl2025.test.3\nacl2025.test.1\n", encoding="utf-8")

    builder = IndexBuilder(settings, store)
    summary = builder.build(paper_ids=builder.load_paper_ids(paper_id_file))

    assert summary.total_papers == 2
    assert summary.indexed_papers == 2
    stored = store.load_papers()
    assert [paper.paper_id for paper in stored] == ["acl2025.test.3", "acl2025.test.1"]
    state = store.load_index_state()
    assert state["total_papers"] == 2
    assert state["indexed_papers"] == 2


def test_search_does_not_fabricate_zero_signal_sections_or_evidence(tmp_path: Path, monkeypatch) -> None:
    _enable_mocked_api(monkeypatch)
    settings = _seed_fixture_data(tmp_path)
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    from chemverify.models import EvidenceBucket, QueryAspect, QueryPlan, ScopeConstraints, VerifierRubric
    from chemverify.search import SearchEngine

    engine = SearchEngine(settings, store)
    engine.load()
    assert engine.runtime is not None

    query_plan = QueryPlan(
        user_query="Find papers about the nonexistent ZZZ benchmark",
        global_query="nonexistent ZZZ benchmark",
        scope_constraints=ScopeConstraints(venues=["acl"], years=[2025], tracks=["long"]),
        entity_terms=["ZZZ"],
        exact_phrases=["ZZZ benchmark"],
        aspect_queries=[
            QueryAspect(aspect_id="a1", query="ZZZ benchmark usage", weight=0.34),
            QueryAspect(aspect_id="a2", query="results on ZZZ benchmark", weight=0.33),
            QueryAspect(aspect_id="a3", query="evaluation on ZZZ benchmark", weight=0.33),
        ],
        verifier_rubric=VerifierRubric(),
        evidence_buckets=[
            EvidenceBucket(bucket_id="entity", description="entity evidence", queries=["ZZZ benchmark"], target_chunks=2),
            EvidenceBucket(bucket_id="global", description="global evidence", queries=["nonexistent ZZZ benchmark"], target_chunks=1),
        ],
    )

    candidate_pool = [("acl2025.test.1", 0.1)]
    narrowed_sections, section_summary = engine._section_narrowing(engine.runtime, query_plan, candidate_pool)
    evidence_packs = engine._assemble_evidence(engine.runtime, query_plan, candidate_pool, narrowed_sections)

    assert len(narrowed_sections["acl2025.test.1"]) <= 1
    assert len(section_summary["acl2025.test.1"]) <= 1
    assert evidence_packs["acl2025.test.1"]["entity"] == []
    assert evidence_packs["acl2025.test.1"]["global"] == []


def test_api_search_matches_cli_top_satisfied_result(tmp_path: Path, monkeypatch) -> None:
    _enable_mocked_api(monkeypatch)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    runner = CliRunner()
    cli_result = runner.invoke(
        cli_app,
        ["search", "--query", "Find ACL 2025 papers that evaluate on GAIA and report GAIA results"],
    )
    assert cli_result.exit_code == 0
    cli_payload = json.loads(cli_result.stdout)

    with TestClient(app) as client:
        health_result = client.get("/api/health")
        assert health_result.status_code == 200

        create_result = client.post(
            "/api/search/jobs",
            json={"query": "Find ACL 2025 papers that evaluate on GAIA and report GAIA results", "top_k": 10},
        )
        assert create_result.status_code == 200
        job_payload = create_result.json()
        job_id = job_payload["job_id"]

        status_payload = job_payload
        deadline = time.time() + 20.0
        while time.time() < deadline:
            status_result = client.get(f"/api/search/jobs/{job_id}")
            assert status_result.status_code == 200
            status_payload = status_result.json()
            if status_payload["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)

        assert status_payload["status"] == "completed", status_payload
        assert status_payload["trace_id"]

        result_response = client.get(f"/api/search/jobs/{job_id}/result")
        assert result_response.status_code == 200
        api_payload = result_response.json()

    assert cli_payload["satisfied"][0]["paper_id"] == api_payload["satisfied"][0]["paper_id"]
    assert api_payload["counts"]["satisfied"] >= 1


def test_search_result_includes_main_image_url_and_optional_enrichment_fields(tmp_path: Path, monkeypatch) -> None:
    _enable_mocked_api(monkeypatch)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    from chemverify.models import ObjectRecord

    image_path = _write_image_fixture(settings, "acl2025.test.3", "overview.png")
    store.save_objects(
        [
            ObjectRecord(
                object_id="obj_figure_overview",
                paper_id="acl2025.test.3",
                section_id="section_experiments",
                object_type="figure_block",
                ordinal=0,
                page_idx=1,
                bbox=[0.0, 0.0, 800.0, 600.0],
                section_path=["Experiments"],
                caption="Figure 1: Overall pipeline overview.",
                image_path=str(image_path),
            )
        ]
    )
    _refresh_search_current(settings)

    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["search", "--query", "Find ACL 2025 papers that evaluate on GAIA and report GAIA results"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)

    top_result = payload["satisfied"][0]
    assert top_result["paper_id"] == "acl2025.test.3"
    assert top_result["main_image_url"] == "/api/papers/acl2025.test.3/images/txt/images/overview.png"
    assert top_result["structured_summary"] is None
    assert top_result["enriched_metadata"] is None


def test_main_image_url_can_use_public_api_base_url(tmp_path: Path, monkeypatch) -> None:
    _enable_mocked_api(monkeypatch)
    monkeypatch.setenv("CHEMVERIFY_PUBLIC_API_BASE_URL", "https://demo.example/api")
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    from chemverify.models import ObjectRecord

    image_path = _write_image_fixture(settings, "acl2025.test.3", "public.png")
    store.save_objects(
        [
            ObjectRecord(
                object_id="obj_public_figure",
                paper_id="acl2025.test.3",
                section_id="section_experiments",
                object_type="figure_block",
                ordinal=0,
                page_idx=1,
                bbox=[0.0, 0.0, 600.0, 400.0],
                section_path=["Experiments"],
                caption="Figure 1: Public demo figure.",
                image_path=str(image_path),
            )
        ]
    )
    _refresh_search_current(settings)

    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["search", "--query", "Find ACL 2025 papers that evaluate on GAIA and report GAIA results"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)

    assert payload["satisfied"][0]["main_image_url"] == "https://demo.example/api/papers/acl2025.test.3/images/txt/images/public.png"


def test_api_serves_paper_image_asset(tmp_path: Path, monkeypatch) -> None:
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    from chemverify.models import ObjectRecord

    image_path = _write_image_fixture(settings, "acl2025.test.3", "figure_1.png")
    store.save_objects(
        [
            ObjectRecord(
                object_id="obj_figure_asset",
                paper_id="acl2025.test.3",
                section_id="section_experiments",
                object_type="figure_block",
                ordinal=0,
                page_idx=2,
                bbox=[0.0, 0.0, 400.0, 300.0],
                section_path=["Experiments"],
                caption="Figure 1: Evaluation setup.",
                image_path=str(image_path),
            )
        ]
    )
    _refresh_search_current(settings)

    with TestClient(app) as client:
        response = client.get("/api/papers/acl2025.test.3/images/txt/images/figure_1.png")

    assert response.status_code == 200
    assert response.content == b"fake-image-bytes"


def test_image_urls_keep_relative_paths_to_avoid_basename_collisions(tmp_path: Path, monkeypatch) -> None:
    _enable_mocked_api(monkeypatch)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    from chemverify.models import ObjectRecord

    primary_image = _write_image_fixture(settings, "acl2025.test.3", "figure_1.png")
    alt_image = settings.mineru_output_dir / "acl2025.test.3" / "alt" / "figure_1.png"
    alt_image.parent.mkdir(parents=True, exist_ok=True)
    alt_image.write_bytes(b"alt-bytes")
    store.save_objects(
        [
            ObjectRecord(
                object_id="obj_primary",
                paper_id="acl2025.test.3",
                section_id="section_experiments",
                object_type="figure_block",
                ordinal=0,
                page_idx=1,
                bbox=[0.0, 0.0, 700.0, 500.0],
                section_path=["Experiments"],
                caption="Figure 1: Overall architecture overview.",
                image_path=str(primary_image),
            ),
            ObjectRecord(
                object_id="obj_alt",
                paper_id="acl2025.test.3",
                section_id="section_appendix",
                object_type="figure_block",
                ordinal=1,
                page_idx=6,
                bbox=[0.0, 0.0, 300.0, 200.0],
                section_path=["Appendix"],
                caption="Figure A1: Extra qualitative examples.",
                image_path=str(alt_image),
            ),
        ]
    )
    _refresh_search_current(settings)

    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["search", "--query", "Find ACL 2025 papers that evaluate on GAIA and report GAIA results"],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["satisfied"][0]["main_image_url"].endswith("/txt/images/figure_1.png")

    with TestClient(app) as client:
        response = client.get("/api/papers/acl2025.test.3/images/alt/figure_1.png")

    assert response.status_code == 200
    assert response.content == b"alt-bytes"


def test_invalid_cached_enrichment_is_ignored(tmp_path: Path, monkeypatch) -> None:
    _enable_mocked_api(monkeypatch)
    settings = _seed_fixture_data(tmp_path)
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(settings.data_dir))
    store = LocalStore(settings)
    _build_and_refresh_search_current(settings, store)

    cache_path = settings.data_dir / "enrichment" / "papers" / "acl2025.test.3.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "structured_summary": {
                    "methodology": ["this should be a string"],
                    "benchmarks": "GAIA",
                    "key_findings": ["ok"],
                }
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["search", "--query", "Find ACL 2025 papers that evaluate on GAIA and report GAIA results"],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["satisfied"][0]["structured_summary"] is None


def test_settings_data_dir_override_rehomes_all_runtime_dirs(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("", encoding="utf-8")
    (root / "config.toml").write_text(
        """
[data]
data_dir = "data"

[mineru]
output_dir = "data/parsed/mineru"
""".strip(),
        encoding="utf-8",
    )
    override_dir = tmp_path / "override_data"
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(override_dir))

    settings = Settings.from_env(root)

    assert settings.data_dir == override_dir.resolve()
    assert settings.normalized_dir == (
        override_dir / "corpora" / "acl" / "2025" / "long" / "release" / "current" / "normalized"
    ).resolve()
    assert settings.index_dir == (
        override_dir / "corpora" / "acl" / "2025" / "long" / "release" / "current" / "indexes" / "layout"
    ).resolve()
    assert settings.trace_dir == (override_dir / "corpora" / "acl" / "2025" / "long" / "traces").resolve()
    assert settings.mineru_output_dir == (override_dir / "parsed" / "mineru").resolve()


def test_verifier_repairs_invalid_entity_role_with_llm_retry(tmp_path: Path, monkeypatch) -> None:
    class _RepairingVerifierClient:
        calls = 0

        def __init__(self, *_, **__) -> None:
            pass

        def __enter__(self) -> "_RepairingVerifierClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> _FakeOpenAIResponse:
            _RepairingVerifierClient.calls += 1
            if _RepairingVerifierClient.calls == 1:
                return _FakeOpenAIResponse(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json_dumps(
                                        {
                                            "verdict": "satisfied",
                                            "entity_role": "benchmark_or_dataset",
                                            "satisfied_constraints": ["uses GAIA benchmark"],
                                            "missing_constraints": [],
                                            "confidence": 0.9,
                                            "rationale": "The evidence shows GAIA benchmark evaluation.",
                                        }
                                    )
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                )
            assert json is not None
            assert "did not satisfy the required schema" in json["messages"][-1]["content"]
            return _FakeOpenAIResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "verdict": "satisfied",
                                        "entity_role": "dataset_or_benchmark",
                                        "satisfied_constraints": ["uses GAIA benchmark"],
                                        "missing_constraints": [],
                                        "confidence": 0.9,
                                        "rationale": "The evidence shows GAIA benchmark evaluation.",
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr("chemverify.search.httpx.Client", _RepairingVerifierClient)

    from chemverify.models import EvidenceBucket, PaperRecord, QueryAspect, QueryPlan, ScopeConstraints, VerifierRubric
    from chemverify.search import SearchEngine

    settings = Settings.from_env(tmp_path)
    engine = SearchEngine(settings, LocalStore(settings))
    paper = PaperRecord(
        paper_id="p1",
        title="Test Paper",
        venue="acl",
        year=2025,
        url="https://example.com",
    )
    plan = QueryPlan(
        user_query="Find papers that use GAIA benchmark",
        global_query="GAIA benchmark",
        scope_constraints=ScopeConstraints(venues=["acl"], years=[2025], tracks=[]),
        entity_terms=["GAIA"],
        exact_phrases=["GAIA benchmark"],
        aspect_queries=[
            QueryAspect(aspect_id="a1", query="GAIA benchmark use", weight=0.34),
            QueryAspect(aspect_id="a2", query="evaluation on GAIA", weight=0.33),
            QueryAspect(aspect_id="a3", query="reported GAIA results", weight=0.33),
        ],
        verifier_rubric=VerifierRubric(),
        evidence_buckets=[
            EvidenceBucket(bucket_id="entity", description="entity evidence", queries=["GAIA benchmark"], target_chunks=1),
        ],
    )

    payload, usage = engine._verify_with_openai(plan, paper, {"entity": []})
    assert payload["entity_role"] == "dataset_or_benchmark"
    assert _RepairingVerifierClient.calls == 2
    assert usage.total_tokens == 30


def test_verifier_payload_preserves_bucket_semantics(tmp_path: Path, monkeypatch) -> None:
    class _RecordingVerifierClient:
        last_user_payload: dict | None = None

        def __init__(self, *_, **__) -> None:
            pass

        def __enter__(self) -> "_RecordingVerifierClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, headers: dict | None = None, json: dict | None = None) -> _FakeOpenAIResponse:
            assert json is not None
            _RecordingVerifierClient.last_user_payload = json_loads(json["messages"][1]["content"])
            return _FakeOpenAIResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_dumps(
                                    {
                                        "verdict": "partial",
                                        "entity_role": "dataset_or_benchmark",
                                        "satisfied_constraints": [],
                                        "missing_constraints": [],
                                        "confidence": 0.5,
                                        "rationale": "ok",
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr("chemverify.search.httpx.Client", _RecordingVerifierClient)

    from chemverify.models import EvidenceBucket, EvidenceChunk, PaperRecord, QueryPlan, ScopeConstraints, VerifierRubric
    from chemverify.search import SearchEngine

    settings = Settings.from_env(tmp_path)
    engine = SearchEngine(settings, LocalStore(settings))
    plan = QueryPlan(
        user_query="Find GAIA benchmark papers",
        global_query="GAIA benchmark papers",
        scope_constraints=ScopeConstraints(venues=["acl"], years=[2025], tracks=["long"]),
        entity_terms=["GAIA"],
        exact_phrases=["GAIA benchmark"],
        verifier_rubric=VerifierRubric(),
        evidence_buckets=[
            EvidenceBucket(
                bucket_id="bucket_1",
                description="show whether GAIA is a benchmark or dataset",
                queries=["GAIA benchmark", "GAIA dataset"],
                target_chunks=2,
            )
        ],
    )
    paper = PaperRecord(
        paper_id="acl2025.test.3",
        anthology_id="acl2025.test.3",
        title="GAIA Paper",
        authors=["A"],
        venue="acl",
        year=2025,
        track="long",
        abstract="Abstract",
        url="https://example.com/gaia",
    )
    evidence_pack = {
        "bucket_1": [
            EvidenceChunk(
                paper_id="acl2025.test.3",
                bucket_id="bucket_1",
                chunk_id="chunk_1",
                chunk_type="text_chunk",
                score=0.9,
                source_query="GAIA benchmark",
                heading="Experiments",
                section_path=["Experiments"],
                page_start=3,
                page_end=3,
                text="We evaluate on the GAIA benchmark.",
            )
        ]
    }

    engine._verify_with_openai(plan, paper, evidence_pack)

    assert _RecordingVerifierClient.last_user_payload is not None
    verifier_bucket = _RecordingVerifierClient.last_user_payload["evidence_buckets"]["bucket_1"]
    assert verifier_bucket["description"] == "show whether GAIA is a benchmark or dataset"
    assert verifier_bucket["queries"] == ["GAIA benchmark", "GAIA dataset"]
    assert verifier_bucket["target_chunks"] == 2
    assert verifier_bucket["chunks"][0]["chunk_id"] == "chunk_1"


def test_settings_rejects_non_mineru_parser_backend(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        """
[pdf_parser]
backend = "provided_text_layout"
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Only 'mineru_layout' is allowed"):
        Settings.from_env(tmp_path)


def test_layout_parser_builds_sections_objects_and_drops_references(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    paper_id = "2025.acl-long.9999"
    parse_dir = settings.mineru_output_dir / paper_id / "auto"
    parse_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "fake.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
    (parse_dir / f"{paper_id}.md").write_text("# fake markdown\n", encoding="utf-8")
    (parse_dir / "images").mkdir(exist_ok=True)
    (parse_dir / "images" / "figure_1.png").write_bytes(b"png")
    (parse_dir / f"{paper_id}_middle.json").write_text(
        json_dumps(
            {
                "pdf_info": [
                    {
                        "page_idx": 0,
                        "para_blocks": [
                            {"type": "title", "text": "Demo Layout V2 Paper"},
                            {"type": "title", "text": "1 Introduction"},
                            {"type": "text", "text": "We evaluate on the GAIA benchmark."},
                            {"type": "title", "text": "2 Results"},
                            {"type": "table", "bbox": [0, 0, 1, 1]},
                            {"type": "image", "bbox": [1, 1, 2, 2]},
                            {"type": "title", "text": "References"},
                            {"type": "text", "text": "This should never be indexed."},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (parse_dir / f"{paper_id}_content_list.json").write_text(
        json_dumps(
            [
                {
                    "type": "table",
                    "page_idx": 0,
                    "table_caption": "Table 1: GAIA results.",
                    "table_body": "GAIA 62.3",
                    "table_footnote": "Higher is better.",
                },
                {
                    "type": "image",
                    "page_idx": 0,
                    "image_caption": ["Figure 1: Overall pipeline."],
                    "image_footnote": ["Visual summary."],
                    "img_path": "images/figure_1.png",
                },
            ]
        ),
        encoding="utf-8",
    )

    from chemverify.models import PaperRecord

    bundle = PDFParser(settings).parse(
        PaperRecord(
            paper_id=paper_id,
            anthology_id=paper_id,
            title="Demo Layout V2 Paper",
            authors=["Ada Lovelace"],
            venue="acl",
            year=2025,
            track="long",
            abstract="A paper for parser testing.",
            url="https://example.com/demo",
            local_pdf_path=str(tmp_path / "fake.pdf"),
        )
    )

    assert [section.section_title for section in bundle.sections] == ["1 Introduction", "2 Results"]
    assert all("reference" not in section.section_title.lower() for section in bundle.sections)
    assert {obj.object_type for obj in bundle.objects} == {"text_block", "table_block", "figure_block"}
    assert any(obj.caption == "Figure 1: Overall pipeline." for obj in bundle.objects)
    assert {chunk.chunk_type for chunk in bundle.chunks} == {"text_chunk", "table_chunk", "figure_chunk"}
    assert any("GAIA 62.3" in chunk.text for chunk in bundle.chunks if chunk.chunk_type == "table_chunk")
    assert all("never be indexed" not in chunk.text.lower() for chunk in bundle.chunks)
    assert all(section.char_end > section.char_start for section in bundle.paper.sections)
    assert all(chunk.char_end > chunk.char_start for chunk in bundle.chunks)


def test_layout_parser_keeps_image_without_caption_when_mineru_provides_image_path(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    paper_id = "2025.acl-long.1004"
    parse_dir = settings.mineru_output_dir / paper_id / "auto"
    parse_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "fake.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
    (parse_dir / f"{paper_id}.md").write_text("# fake markdown\n", encoding="utf-8")
    (parse_dir / "images").mkdir(exist_ok=True)
    image_path = parse_dir / "images" / "uncaptioned.jpg"
    image_path.write_bytes(b"jpg")
    (parse_dir / f"{paper_id}_middle.json").write_text(
        json_dumps(
            {
                "pdf_info": [
                    {
                        "page_idx": 0,
                        "para_blocks": [
                            {"type": "title", "text": "Image-only Figure Paper"},
                            {"type": "chart", "bbox": [10, 10, 100, 100]},
                            {"type": "title", "text": "1 Introduction"},
                            {"type": "text", "text": "The paper contains an image without a caption."},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (parse_dir / f"{paper_id}_content_list.json").write_text(
        json_dumps(
            [
                {
                    "type": "chart",
                    "page_idx": 0,
                    "bbox": [10, 10, 100, 100],
                    "image_caption": [],
                    "image_footnote": [],
                    "img_path": "images/uncaptioned.jpg",
                }
            ]
        ),
        encoding="utf-8",
    )

    from chemverify.models import PaperRecord

    bundle = PDFParser(settings).parse(
        PaperRecord(
            paper_id=paper_id,
            anthology_id=paper_id,
            title="Image-only Figure Paper",
            authors=["Ada Lovelace"],
            venue="acl",
            year=2025,
            track="long",
            abstract="A parser regression fixture.",
            url="https://example.com/image-only",
            local_pdf_path=str(tmp_path / "fake.pdf"),
        )
    )

    figures = [obj for obj in bundle.objects if obj.object_type == "figure_block"]
    assert len(figures) == 1
    assert figures[0].image_path == str(image_path.resolve())
    assert figures[0].text == "Figure"
    assert figures[0].section_path == ["Document"]


def test_layout_parser_does_not_consume_table_supplement_on_empty_block(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    paper_id = "2025.acl-long.1000"
    parse_dir = settings.mineru_output_dir / paper_id / "auto"
    parse_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "fake.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
    (parse_dir / f"{paper_id}.md").write_text("# fake markdown\n", encoding="utf-8")
    (parse_dir / f"{paper_id}_middle.json").write_text(
        json_dumps(
            {
                "pdf_info": [
                    {
                        "page_idx": 0,
                        "para_blocks": [
                            {"type": "title", "text": "Demo Layout V2 Paper"},
                            {"type": "title", "text": "1 Results"},
                            {"type": "table", "bbox": [0, 0, 2, 2]},
                            {"type": "table", "bbox": [10, 10, 20, 20]},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (parse_dir / f"{paper_id}_content_list.json").write_text(
        json_dumps(
            [
                {
                    "type": "table",
                    "page_idx": 0,
                    "bbox": [10, 10, 20, 20],
                    "table_caption": "Table 1: Recovered caption.",
                    "table_body": "Recovered body text.",
                }
            ]
        ),
        encoding="utf-8",
    )

    from chemverify.models import PaperRecord

    bundle = PDFParser(settings).parse(
        PaperRecord(
            paper_id=paper_id,
            anthology_id=paper_id,
            title="Demo Layout V2 Paper",
            authors=["Ada Lovelace"],
            venue="acl",
            year=2025,
            track="long",
            abstract="A paper for parser testing.",
            url="https://example.com/demo",
            local_pdf_path=str(tmp_path / "fake.pdf"),
        )
    )

    table_chunks = [chunk for chunk in bundle.chunks if chunk.chunk_type == "table_chunk"]
    assert len(table_chunks) == 1
    assert "Recovered caption." in table_chunks[0].text
    assert "Recovered body text." in table_chunks[0].text


def test_settings_loads_config_toml_and_env_override(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.toml").write_text(
        "\n".join(
            [
                "[pdf_parser]",
                'backend = "mineru_layout"',
                "",
                "[mineru]",
                'command = ".venv/bin/mineru"',
                'backend = "pipeline"',
                'method = "auto"',
                'lang = "en"',
                'source = "huggingface"',
                'output_dir = "custom_mineru"',
                "require_middle_json = true",
                "",
                "[indexing]",
                'paper_dense_model = "demo-paper-model"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("CHEMVERIFY_PAPER_DENSE_MODEL", "env-paper-model")
    settings = Settings.from_env(tmp_path)

    assert settings.pdf_parser_backend == "mineru_layout"
    assert settings.mineru_lang == "en"
    assert settings.mineru_output_dir == (tmp_path / "custom_mineru").resolve()
    assert settings.mineru_require_middle_json is True
    assert settings.paper_dense_model == "env-paper-model"


def test_settings_accepts_openai_official_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_OFFICIAL_API_KEY", "official-test-key")

    settings = Settings.from_env(tmp_path)

    assert settings.openai_api_key == "official-test-key"


def test_settings_require_explicit_openai_model(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.toml").write_text(
        """
[openai]
model = ""
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_MODEL must be explicitly configured"):
        Settings.from_env(tmp_path)


def test_index_builder_records_parse_failures_without_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("chemverify.encoders.SentenceTransformerEncoder.__init__", _fake_encoder_init)
    monkeypatch.setattr("chemverify.encoders.SentenceTransformerEncoder.encode", _fake_encoder_encode)
    settings = Settings.from_env(tmp_path)
    store = LocalStore(settings)
    from chemverify.models import PaperRecord

    store.save_raw_papers(
        [
            PaperRecord(
                paper_id="acl2025.fail.1",
                anthology_id="acl2025.fail.1",
                title="Broken PDF Example",
                authors=["Ada Lovelace"],
                venue="acl",
                year=2025,
                track="long",
                abstract="A paper whose PDF path is missing.",
                url="https://example.com/broken",
                local_pdf_path=str(tmp_path / "missing.pdf"),
            )
        ]
    )

    summary = IndexBuilder(settings, store).build()
    _refresh_search_current(settings)

    assert summary.indexed_papers == 0
    assert summary.failed_papers == 1
    failures = store.load_parse_failures()
    assert len(failures) == 1
    assert failures[0].paper_id == "acl2025.fail.1"
    assert failures[0].analysis
    stored_paper = store.load_papers()[0]
    assert stored_paper.metadata["parse_status"] == "failed"


def test_acl_ingestor_uses_hierarchical_pdf_destination(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    store = LocalStore(settings)
    ingestor = ACLAnthologyIngestor(settings, store)
    listing = _ListingEntry(
        paper_id="2025.acl-long.1383",
        title="Test Paper",
        listing_title="Test Paper",
        authors=["A"],
        abstract="B",
        volume_id="2025.acl-long",
        venue="acl",
        year=2025,
        track="long",
        url="https://aclanthology.org/2025.acl-long.1383/",
        pdf_url="https://aclanthology.org/2025.acl-long.1383.pdf",
    )

    destination = ingestor._pdf_destination(listing)

    assert destination == settings.pdf_dir / "acl" / "2025" / "long" / "2025.acl-long.1383.pdf"


def test_acl_download_timeout_becomes_readable_error(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(tmp_path)
    store = LocalStore(settings)
    ingestor = ACLAnthologyIngestor(settings, store)
    listing = _ListingEntry(
        paper_id="2025.acl-long.20",
        title="Test Paper",
        listing_title="Test Paper",
        authors=["A"],
        abstract="B",
        volume_id="2025.acl-long",
        venue="acl",
        year=2025,
        track="long",
        url="https://aclanthology.org/2025.acl-long.20/",
        pdf_url="https://aclanthology.org/2025.acl-long.20.pdf",
    )

    class _TimeoutClient:
        def __init__(self, *_, **__) -> None:
            pass

        def __enter__(self) -> "_TimeoutClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url: str):
            raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr("chemverify.acl_anthology.httpx.Client", _TimeoutClient)

    with pytest.raises(RuntimeError, match="paper=2025.acl-long.20"):
        ingestor._download_pdfs([listing])

    assert not ingestor._pdf_destination(listing).exists()


def test_demo_acl_reports_download_failure_without_traceback(monkeypatch) -> None:
    class _FailingIngestor:
        def __init__(self, *_, **__) -> None:
            pass

        def ingest_event(self, *_, **__):
            raise RuntimeError("Could not download an ACL Anthology PDF.")

    monkeypatch.setattr(cli_module, "ACLAnthologyIngestor", _FailingIngestor)

    result = CliRunner().invoke(cli_app, ["demo-acl", "--max-papers", "1"])

    assert result.exit_code == 1
    assert "Could not download an ACL Anthology PDF." in result.output
    assert "Traceback" not in result.output


def test_acl_ingestor_treats_main_and_demo_aliases_as_in_scope(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    store = LocalStore(settings)
    ingestor = ACLAnthologyIngestor(settings, store)
    listings = [
        _ListingEntry(
            paper_id="2022.emnlp-main.1",
            title="Main Volume Paper",
            listing_title="Main Volume Paper",
            authors=["A"],
            abstract="B",
            volume_id="2022.emnlp-main",
            venue="emnlp",
            year=2022,
            track="main",
            url="https://aclanthology.org/2022.emnlp-main.1/",
            pdf_url="https://aclanthology.org/2022.emnlp-main.1.pdf",
        ),
        _ListingEntry(
            paper_id="2025.emnlp-demos.1",
            title="Demo Volume Paper",
            listing_title="Demo Volume Paper",
            authors=["A"],
            abstract="B",
            volume_id="2025.emnlp-demos",
            venue="emnlp",
            year=2025,
            track="demo",
            url="https://aclanthology.org/2025.emnlp-demos.1/",
            pdf_url="https://aclanthology.org/2025.emnlp-demos.1.pdf",
        ),
    ]

    filtered = ingestor._filter_listings(listings, ["long", "short", "demo"])

    assert {item.paper_id for item in filtered} == {"2022.emnlp-main.1", "2025.emnlp-demos.1"}


def test_acl_ingestor_does_not_expand_long_or_short_to_full_main_family(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    store = LocalStore(settings)
    ingestor = ACLAnthologyIngestor(settings, store)
    listings = [
        _ListingEntry(
            paper_id="2025.acl-long.1",
            title="Long Paper",
            listing_title="Long Paper",
            authors=["A"],
            abstract="B",
            volume_id="2025.acl-long",
            venue="acl",
            year=2025,
            track="long",
            url="https://aclanthology.org/2025.acl-long.1/",
            pdf_url="https://aclanthology.org/2025.acl-long.1.pdf",
        ),
        _ListingEntry(
            paper_id="2025.acl-short.1",
            title="Short Paper",
            listing_title="Short Paper",
            authors=["A"],
            abstract="B",
            volume_id="2025.acl-short",
            venue="acl",
            year=2025,
            track="short",
            url="https://aclanthology.org/2025.acl-short.1/",
            pdf_url="https://aclanthology.org/2025.acl-short.1.pdf",
        ),
        _ListingEntry(
            paper_id="2025.emnlp-main.1",
            title="Main Paper",
            listing_title="Main Paper",
            authors=["A"],
            abstract="B",
            volume_id="2025.emnlp-main",
            venue="emnlp",
            year=2025,
            track="main",
            url="https://aclanthology.org/2025.emnlp-main.1/",
            pdf_url="https://aclanthology.org/2025.emnlp-main.1.pdf",
        ),
    ]

    filtered_long = ingestor._filter_listings(listings, ["long"])
    filtered_short = ingestor._filter_listings(listings, ["short"])

    assert {item.paper_id for item in filtered_long} == {"2025.acl-long.1"}
    assert {item.paper_id for item in filtered_short} == {"2025.acl-short.1"}


def test_normalized_phrase_match_respects_token_boundaries() -> None:
    from chemverify.search import _contains_normalized_phrase

    assert _contains_normalized_phrase("results on gaia benchmark", "gaia")
    assert _contains_normalized_phrase("results on gaia benchmark", "gaia benchmark")
    assert not _contains_normalized_phrase("drag based planner", "rag")
    assert not _contains_normalized_phrase("surface form features", "ace")
