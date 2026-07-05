from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_CONFIG: dict[str, Any] = {
    "service": {
        "public_api_base_url": None,
    },
    "data": {
        "data_dir": "data",
    },
    "deep_chat": {
        "history_turn_limit": 3,
        "global_evidence_k": 1,
        "local_evidence_k": 3,
        "retrieval_candidate_k": 10,
        "max_evidence_text_chars": 1200,
    },
    "pdf_parser": {
        "backend": "mineru_layout",
    },
    "mineru": {
        "command": ".venv/bin/mineru",
        "backend": "pipeline",
        "method": "auto",
        "lang": "en",
        "source": "huggingface",
        "device": None,
        "formula": False,
        "table": True,
        "output_dir": "data/parsed/mineru",
        "require_middle_json": True,
        "require_markdown": True,
        "timeout_seconds": 3600.0,
    },
    "indexing": {
        "paper_dense_model": "allenai/specter2_base",
        "chunk_dense_model": "BAAI/bge-m3",
        "chunk_target_tokens": 420,
        "chunk_overlap_tokens": 63,
        "dense_device": None,
        "dense_batch_size": 8,
    },
    "retrieval": {
        "candidate_pool_size": 50,
        "verifier_candidate_limit": 20,
        "candidate_source_limit": 120,
        "default_top_k": 10,
        "paper_sparse_rrf_weight": 0.28,
        "paper_dense_rrf_weight": 0.22,
        "chunk_aggregated_rrf_weight": 0.35,
        "literal_entity_rrf_weight": 0.10,
        "exact_phrase_rrf_weight": 0.05,
        "aspect_coverage_bonus": 0.08,
        "source_diversity_bonus": 0.05,
        "literal_entity_bonus": 0.08,
        "exact_phrase_bonus": 0.04,
        "evidence_sparse_weight": 0.35,
        "evidence_dense_weight": 0.40,
        "evidence_reranker_weight": 0.25,
        "evidence_reranker_candidate_chunks": 8,
        "verifier_max_workers": 12,
        "evidence_chunk_text_limit": 1200,
    },
    "reranker": {
        "model": "BAAI/bge-reranker-v2-m3",
        "device": None,
        "batch_size": 8,
    },
    "openai": {
        "enabled": True,
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "request_timeout": 60.0,
        "input_price_per_1m": None,
        "output_price_per_1m": None,
    },
    "grobid": {
        "enabled": False,
        "base_url": "http://127.0.0.1:8070",
        "timeout_seconds": 120.0,
    },
}


@dataclass(slots=True, frozen=True)
class CorpusSpec:
    venue: str
    year: int
    track: str

    @classmethod
    def from_values(cls, venue: str | None, year: int | None, track: str | None) -> "CorpusSpec":
        normalized_venue = (venue or "acl").strip().lower()
        normalized_year = int(year or 2025)
        normalized_track = (track or "long").strip().lower()
        if normalized_track not in {"long", "short", "demo", "all", "library"}:
            raise RuntimeError(
                f"Unsupported track={normalized_track!r}. Expected one of: long, short, demo, all, library."
            )
        return cls(venue=normalized_venue, year=normalized_year, track=normalized_track)

    @property
    def key(self) -> str:
        return f"{self.venue}/{self.year}/{self.track}"

    @property
    def path_parts(self) -> tuple[str, str, str]:
        return self.venue, str(self.year), self.track

    def to_dict(self) -> dict[str, object]:
        return {"venue": self.venue, "year": self.year, "track": self.track}


@dataclass(slots=True)
class Settings:
    root_dir: Path
    config_path: Path
    corpus: CorpusSpec
    data_dir: Path
    manifest_dir: Path
    corpus_dir: Path
    release_dir: Path
    release_snapshots_dir: Path
    current_release_path: Path
    search_current_dir: Path
    search_current_manifest_path: Path
    search_current_staging_dir: Path
    work_dir: Path
    state_dir: Path
    global_state_dir: Path
    active_corpus_path: Path
    active_job_path: Path
    last_job_path: Path
    mineru_failure_manifest_path: Path
    raw_dir: Path
    pdf_dir: Path
    parsed_dir: Path
    normalized_dir: Path
    deep_chat_normalized_dir: Path
    index_dir: Path
    deep_chat_index_dir: Path
    trace_dir: Path
    public_api_base_url: str | None
    pdf_parser_backend: str
    mineru_command: str
    mineru_backend: str
    mineru_method: str
    mineru_lang: str | None
    mineru_source: str
    mineru_device: str | None
    mineru_formula: bool
    mineru_table: bool
    mineru_output_dir: Path
    mineru_require_middle_json: bool
    mineru_require_markdown: bool
    mineru_timeout_seconds: float
    openai_api_key: str | None
    openai_base_url: str
    openai_model: str
    openai_enabled: bool
    grobid_enabled: bool
    grobid_base_url: str
    grobid_timeout_seconds: float
    paper_dense_model: str
    chunk_dense_model: str
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    candidate_pool_size: int
    verifier_candidate_limit: int
    candidate_source_limit: int
    default_top_k: int
    paper_sparse_rrf_weight: float
    paper_dense_rrf_weight: float
    chunk_aggregated_rrf_weight: float
    literal_entity_rrf_weight: float
    exact_phrase_rrf_weight: float
    aspect_coverage_bonus: float
    source_diversity_bonus: float
    literal_entity_bonus: float
    exact_phrase_bonus: float
    evidence_sparse_weight: float
    evidence_dense_weight: float
    evidence_reranker_weight: float
    evidence_reranker_candidate_chunks: int
    verifier_max_workers: int
    evidence_chunk_text_limit: int
    reranker_model: str
    reranker_device: str | None
    reranker_batch_size: int
    deep_chat_history_turn_limit: int
    deep_chat_global_evidence_k: int
    deep_chat_local_evidence_k: int
    deep_chat_retrieval_candidate_k: int
    deep_chat_max_evidence_text_chars: int
    request_timeout: float
    dense_device: str | None
    dense_batch_size: int
    input_price_per_1m: float | None
    output_price_per_1m: float | None

    @classmethod
    def from_env(
        cls,
        root_dir: str | Path | None = None,
        *,
        corpus: CorpusSpec | None = None,
    ) -> "Settings":
        base_dir = Path(root_dir or os.getcwd()).resolve()
        config_path = base_dir / "config.toml"
        load_dotenv(base_dir / ".env", override=False)

        config_payload = _deep_copy(DEFAULT_CONFIG)
        user_payload: dict[str, Any] = {}
        if config_path.exists():
            user_payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
            _deep_update(config_payload, user_payload)

        device = _as_optional_str(_env_value("CHEMVERIFY_DEVICE"))
        data_dir_override = _env_value("CHEMVERIFY_DATA_DIR")
        data_dir = _resolve_dir(
            base_dir,
            _env_or_config(
                env_name="CHEMVERIFY_DATA_DIR",
                payload=config_payload,
                path=("data", "data_dir"),
            ),
        )
        corpus_spec = _resolve_corpus(base_dir=base_dir, data_dir=data_dir, explicit=corpus)
        manifest_dir = data_dir / "manifests" / corpus_spec.venue / str(corpus_spec.year) / corpus_spec.track
        corpus_dir = data_dir / "corpora" / corpus_spec.venue / str(corpus_spec.year) / corpus_spec.track
        release_dir = corpus_dir / "release"
        release_snapshots_dir = release_dir / "snapshots"
        current_release_path = release_dir / "current"
        search_current_dir = data_dir / "search_current"
        search_current_manifest_path = search_current_dir / "manifest.json"
        search_current_staging_dir = data_dir / "search_current_staging"
        work_dir = corpus_dir / "work"
        state_dir = corpus_dir / "state"
        global_state_dir = data_dir / "state"
        active_corpus_path = global_state_dir / "active_corpus.json"
        active_job_path = global_state_dir / "active_job.json"
        last_job_path = global_state_dir / "last_job.json"
        mineru_failure_manifest_path = data_dir / "parsed" / "mineru_failures.jsonl"
        normalized_dir = current_release_path / "normalized"
        index_dir = current_release_path / "indexes" / "layout"
        trace_dir = corpus_dir / "traces"
        deep_chat_normalized_dir = normalized_dir / "deep_chat"
        deep_chat_index_dir = current_release_path / "indexes" / "deep_chat"
        mineru_output_dir = _resolve_dir(
            base_dir,
            _env_value("CHEMVERIFY_MINERU_OUTPUT_DIR")
            or (
                str(data_dir / "parsed" / "mineru")
                if data_dir_override is not None
                else _env_or_config(
                    env_name="CHEMVERIFY_MINERU_OUTPUT_DIR",
                    payload=config_payload,
                    path=("mineru", "output_dir"),
                )
            ),
        )

        settings = cls(
            root_dir=base_dir,
            config_path=config_path,
            corpus=corpus_spec,
            data_dir=data_dir,
            manifest_dir=manifest_dir,
            corpus_dir=corpus_dir,
            release_dir=release_dir,
            release_snapshots_dir=release_snapshots_dir,
            current_release_path=current_release_path,
            search_current_dir=search_current_dir,
            search_current_manifest_path=search_current_manifest_path,
            search_current_staging_dir=search_current_staging_dir,
            work_dir=work_dir,
            state_dir=state_dir,
            global_state_dir=global_state_dir,
            active_corpus_path=active_corpus_path,
            active_job_path=active_job_path,
            last_job_path=last_job_path,
            mineru_failure_manifest_path=mineru_failure_manifest_path,
            raw_dir=manifest_dir,
            pdf_dir=data_dir / "pdfs",
            parsed_dir=data_dir / "parsed",
            normalized_dir=normalized_dir,
            deep_chat_normalized_dir=deep_chat_normalized_dir,
            index_dir=index_dir,
            deep_chat_index_dir=deep_chat_index_dir,
            trace_dir=trace_dir,
            public_api_base_url=_as_optional_str(
                _env_or_config(
                    env_name="CHEMVERIFY_PUBLIC_API_BASE_URL",
                    payload=config_payload,
                    path=("service", "public_api_base_url"),
                )
            ),
            pdf_parser_backend=_as_str(
                _env_or_config(
                    env_name="CHEMVERIFY_PDF_PARSER_BACKEND",
                    payload=config_payload,
                    path=("pdf_parser", "backend"),
                )
            ),
            mineru_command=_as_str(
                _env_or_config(
                    env_name="CHEMVERIFY_MINERU_COMMAND",
                    payload=config_payload,
                    path=("mineru", "command"),
                )
            ),
            mineru_backend=_as_str(
                _env_or_config(
                    env_name="CHEMVERIFY_MINERU_BACKEND",
                    payload=config_payload,
                    path=("mineru", "backend"),
                )
            ),
            mineru_method=_as_str(
                _env_or_config(
                    env_name="CHEMVERIFY_MINERU_METHOD",
                    payload=config_payload,
                    path=("mineru", "method"),
                )
            ),
            mineru_lang=_as_optional_str(
                _env_or_config(
                    env_name="CHEMVERIFY_MINERU_LANG",
                    payload=config_payload,
                    path=("mineru", "lang"),
                )
            ),
            mineru_source=_as_str(
                _env_or_config(
                    env_name="CHEMVERIFY_MINERU_SOURCE",
                    payload=config_payload,
                    path=("mineru", "source"),
                )
            ),
            mineru_device=_as_optional_str(
                _env_value("CHEMVERIFY_MINERU_DEVICE")
                or device
                or _env_or_config(
                    env_name="CHEMVERIFY_MINERU_DEVICE",
                    payload=config_payload,
                    path=("mineru", "device"),
                )
            ),
            mineru_formula=_as_bool(
                _env_or_config(
                    env_name="CHEMVERIFY_MINERU_FORMULA",
                    payload=config_payload,
                    path=("mineru", "formula"),
                )
            ),
            mineru_table=_as_bool(
                _env_or_config(
                    env_name="CHEMVERIFY_MINERU_TABLE",
                    payload=config_payload,
                    path=("mineru", "table"),
                )
            ),
            mineru_output_dir=mineru_output_dir,
            mineru_require_middle_json=_as_bool(
                _env_or_config(
                    env_name="CHEMVERIFY_MINERU_REQUIRE_MIDDLE_JSON",
                    payload=config_payload,
                    path=("mineru", "require_middle_json"),
                )
            ),
            mineru_require_markdown=_as_bool(
                _env_or_config(
                    env_name="CHEMVERIFY_MINERU_REQUIRE_MARKDOWN",
                    payload=config_payload,
                    path=("mineru", "require_markdown"),
                )
            ),
            mineru_timeout_seconds=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_MINERU_TIMEOUT_SECONDS",
                    payload=config_payload,
                    path=("mineru", "timeout_seconds"),
                )
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_OFFICIAL_API_KEY"),
            openai_base_url=_as_str(
                _env_or_config(
                    env_name="OPENAI_BASE_URL",
                    payload=config_payload,
                    path=("openai", "base_url"),
                )
            ).rstrip("/"),
            openai_model=_as_str(
                _env_or_config(
                    env_name="OPENAI_MODEL",
                    payload=config_payload,
                    path=("openai", "model"),
                )
            ),
            openai_enabled=_as_bool(
                _env_or_config(
                    env_name="CHEMVERIFY_OPENAI_ENABLED",
                    payload=config_payload,
                    path=("openai", "enabled"),
                )
            ),
            grobid_enabled=_as_bool(
                _env_or_config(
                    env_name="CHEMVERIFY_GROBID_ENABLED",
                    payload=config_payload,
                    path=("grobid", "enabled"),
                )
            ),
            grobid_base_url=_as_str(
                _env_or_config(
                    env_name="CHEMVERIFY_GROBID_BASE_URL",
                    payload=config_payload,
                    path=("grobid", "base_url"),
                )
            ).rstrip("/"),
            grobid_timeout_seconds=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_GROBID_TIMEOUT_SECONDS",
                    payload=config_payload,
                    path=("grobid", "timeout_seconds"),
                )
            ),
            paper_dense_model=_as_str(
                _env_or_config(
                    env_name="CHEMVERIFY_PAPER_DENSE_MODEL",
                    payload=config_payload,
                    path=("indexing", "paper_dense_model"),
                )
            ),
            chunk_dense_model=_as_str(
                _env_or_config(
                    env_name="CHEMVERIFY_CHUNK_DENSE_MODEL",
                    payload=config_payload,
                    path=("indexing", "chunk_dense_model"),
                )
            ),
            chunk_target_tokens=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_CHUNK_TARGET_TOKENS",
                    payload=config_payload,
                    path=("indexing", "chunk_target_tokens"),
                )
            ),
            chunk_overlap_tokens=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_CHUNK_OVERLAP_TOKENS",
                    payload=config_payload,
                    path=("indexing", "chunk_overlap_tokens"),
                )
            ),
            candidate_pool_size=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_CANDIDATE_POOL_SIZE",
                    payload=config_payload,
                    path=("retrieval", "candidate_pool_size"),
                )
            ),
            verifier_candidate_limit=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_VERIFIER_CANDIDATE_LIMIT",
                    payload=config_payload,
                    path=("retrieval", "verifier_candidate_limit"),
                )
            ),
            candidate_source_limit=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_CANDIDATE_SOURCE_LIMIT",
                    payload=config_payload,
                    path=("retrieval", "candidate_source_limit"),
                )
            ),
            default_top_k=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_DEFAULT_TOP_K",
                    payload=config_payload,
                    path=("retrieval", "default_top_k"),
                )
            ),
            paper_sparse_rrf_weight=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_PAPER_SPARSE_RRF_WEIGHT",
                    payload=config_payload,
                    path=("retrieval", "paper_sparse_rrf_weight"),
                )
            ),
            paper_dense_rrf_weight=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_PAPER_DENSE_RRF_WEIGHT",
                    payload=config_payload,
                    path=("retrieval", "paper_dense_rrf_weight"),
                )
            ),
            chunk_aggregated_rrf_weight=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_CHUNK_AGGREGATED_RRF_WEIGHT",
                    payload=config_payload,
                    path=("retrieval", "chunk_aggregated_rrf_weight"),
                )
            ),
            literal_entity_rrf_weight=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_LITERAL_ENTITY_RRF_WEIGHT",
                    payload=config_payload,
                    path=("retrieval", "literal_entity_rrf_weight"),
                )
            ),
            exact_phrase_rrf_weight=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_EXACT_PHRASE_RRF_WEIGHT",
                    payload=config_payload,
                    path=("retrieval", "exact_phrase_rrf_weight"),
                )
            ),
            aspect_coverage_bonus=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_ASPECT_COVERAGE_BONUS",
                    payload=config_payload,
                    path=("retrieval", "aspect_coverage_bonus"),
                )
            ),
            source_diversity_bonus=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_SOURCE_DIVERSITY_BONUS",
                    payload=config_payload,
                    path=("retrieval", "source_diversity_bonus"),
                )
            ),
            literal_entity_bonus=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_LITERAL_ENTITY_BONUS",
                    payload=config_payload,
                    path=("retrieval", "literal_entity_bonus"),
                )
            ),
            exact_phrase_bonus=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_EXACT_PHRASE_BONUS",
                    payload=config_payload,
                    path=("retrieval", "exact_phrase_bonus"),
                )
            ),
            evidence_sparse_weight=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_EVIDENCE_SPARSE_WEIGHT",
                    payload=config_payload,
                    path=("retrieval", "evidence_sparse_weight"),
                )
            ),
            evidence_dense_weight=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_EVIDENCE_DENSE_WEIGHT",
                    payload=config_payload,
                    path=("retrieval", "evidence_dense_weight"),
                )
            ),
            evidence_reranker_weight=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_EVIDENCE_RERANKER_WEIGHT",
                    payload=config_payload,
                    path=("retrieval", "evidence_reranker_weight"),
                )
            ),
            evidence_reranker_candidate_chunks=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_EVIDENCE_RERANKER_CANDIDATE_CHUNKS",
                    payload=config_payload,
                    path=("retrieval", "evidence_reranker_candidate_chunks"),
                )
            ),
            verifier_max_workers=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_VERIFIER_MAX_WORKERS",
                    payload=config_payload,
                    path=("retrieval", "verifier_max_workers"),
                )
            ),
            evidence_chunk_text_limit=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_EVIDENCE_CHUNK_TEXT_LIMIT",
                    payload=config_payload,
                    path=("retrieval", "evidence_chunk_text_limit"),
                )
            ),
            reranker_model=_as_str(
                _env_or_config(
                    env_name="CHEMVERIFY_RERANKER_MODEL",
                    payload=config_payload,
                    path=("reranker", "model"),
                )
            ),
            reranker_device=_as_optional_str(
                _env_value("CHEMVERIFY_RERANKER_DEVICE")
                or device
                or _env_or_config(
                    env_name="CHEMVERIFY_RERANKER_DEVICE",
                    payload=config_payload,
                    path=("reranker", "device"),
                )
            ),
            reranker_batch_size=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_RERANKER_BATCH_SIZE",
                    payload=config_payload,
                    path=("reranker", "batch_size"),
                )
            ),
            deep_chat_history_turn_limit=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_DEEP_CHAT_HISTORY_TURN_LIMIT",
                    payload=config_payload,
                    path=("deep_chat", "history_turn_limit"),
                )
            ),
            deep_chat_global_evidence_k=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_DEEP_CHAT_GLOBAL_EVIDENCE_K",
                    payload=config_payload,
                    path=("deep_chat", "global_evidence_k"),
                )
            ),
            deep_chat_local_evidence_k=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_DEEP_CHAT_LOCAL_EVIDENCE_K",
                    payload=config_payload,
                    path=("deep_chat", "local_evidence_k"),
                )
            ),
            deep_chat_retrieval_candidate_k=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_DEEP_CHAT_RETRIEVAL_CANDIDATE_K",
                    payload=config_payload,
                    path=("deep_chat", "retrieval_candidate_k"),
                )
            ),
            deep_chat_max_evidence_text_chars=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_DEEP_CHAT_MAX_EVIDENCE_TEXT_CHARS",
                    payload=config_payload,
                    path=("deep_chat", "max_evidence_text_chars"),
                )
            ),
            request_timeout=_as_float(
                _env_or_config(
                    env_name="CHEMVERIFY_REQUEST_TIMEOUT",
                    payload=config_payload,
                    path=("openai", "request_timeout"),
                )
            ),
            dense_device=_as_optional_str(
                _env_value("CHEMVERIFY_DENSE_DEVICE")
                or device
                or _env_or_config(
                    env_name="CHEMVERIFY_DENSE_DEVICE",
                    payload=config_payload,
                    path=("indexing", "dense_device"),
                )
            ),
            dense_batch_size=_as_int(
                _env_or_config(
                    env_name="CHEMVERIFY_DENSE_BATCH_SIZE",
                    payload=config_payload,
                    path=("indexing", "dense_batch_size"),
                )
            ),
            input_price_per_1m=_as_optional_float(
                _env_or_config(
                    env_name="CHEMVERIFY_INPUT_PRICE_PER_1M",
                    payload=config_payload,
                    path=("openai", "input_price_per_1m"),
                )
            ),
            output_price_per_1m=_as_optional_float(
                _env_or_config(
                    env_name="CHEMVERIFY_OUTPUT_PRICE_PER_1M",
                    payload=config_payload,
                    path=("openai", "output_price_per_1m"),
                )
            ),
        )
        _validate_settings(settings)
        settings.ensure_dirs()
        return settings

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.manifest_dir,
            self.corpus_dir,
            self.release_dir,
            self.release_snapshots_dir,
            self.work_dir,
            self.state_dir,
            self.global_state_dir,
            self.raw_dir,
            self.pdf_dir,
            self.parsed_dir,
            self.trace_dir,
            self.mineru_output_dir,
            self.search_current_staging_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolve_mineru_command(self) -> str:
        command_path = Path(self.mineru_command)
        if command_path.is_absolute():
            return str(command_path)
        if "/" in self.mineru_command:
            return str((self.root_dir / command_path).resolve())
        return self.mineru_command

    @property
    def corpus_label(self) -> str:
        return f"{self.corpus.venue} {self.corpus.year} {self.corpus.track}"


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _deep_update(target: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _env_or_config(*, env_name: str, payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    raw = _env_value(env_name)
    if raw not in (None, ""):
        return raw
    current: Any = payload
    for part in path:
        current = current[part]
    return current


def _env_value(env_name: str) -> str | None:
    return os.getenv(env_name)


def _resolve_dir(base_dir: Path, value: Any) -> Path:
    path = Path(_as_str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_corpus(*, base_dir: Path, data_dir: Path, explicit: CorpusSpec | None) -> CorpusSpec:
    if explicit is not None:
        return explicit

    env_venue = _env_value("CHEMVERIFY_VENUE")
    env_year = _env_value("CHEMVERIFY_YEAR")
    env_track = _env_value("CHEMVERIFY_TRACK")
    if any(value not in (None, "") for value in (env_venue, env_year, env_track)):
        return CorpusSpec.from_values(env_venue, int(env_year) if env_year not in (None, "") else None, env_track)

    active_corpus_path = data_dir / "state" / "active_corpus.json"
    if active_corpus_path.exists():
        try:
            payload = json.loads(active_corpus_path.read_text(encoding="utf-8"))
            return CorpusSpec.from_values(
                str(payload.get("venue") or ""),
                int(payload.get("year")) if payload.get("year") is not None else None,
                str(payload.get("track") or ""),
            )
        except Exception:
            pass

    return CorpusSpec.from_values("acl", 2025, "long")


def _as_str(value: Any) -> str:
    return str(value)


def _as_optional_str(value: Any) -> str | None:
    if value in (None, "", "None", "none", "null"):
        return None
    return str(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any) -> int:
    return int(value)


def _as_float(value: Any) -> float:
    return float(value)


def _as_optional_float(value: Any) -> float | None:
    if value in (None, "", "None", "none", "null"):
        return None
    return float(value)


def _validate_settings(settings: Settings) -> None:
    if settings.pdf_parser_backend != "mineru_layout":
        raise RuntimeError(
            f"Unsupported pdf_parser.backend={settings.pdf_parser_backend!r}. Only 'mineru_layout' is allowed."
        )
    if not settings.openai_model.strip():
        raise RuntimeError("OPENAI_MODEL must be explicitly configured.")
