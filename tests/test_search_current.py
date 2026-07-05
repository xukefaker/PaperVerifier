from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chemverify.config import CorpusSpec, Settings
from chemverify.search_current import rebuild_search_current
from chemverify.storage import LocalStore


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")


def _write_index_bundle(
    root: Path,
    *,
    name: str,
    ids: list[str],
    texts: list[str],
    tokens: list[list[str]],
    vector_dim: int,
    encoder_model: str,
    extra: dict[str, list] | None = None,
) -> None:
    meta_path = root / f"{name}_index_meta.json"
    vector_path = root / f"{name}_vectors.npz"
    meta = {
        "ids": ids,
        "texts": texts,
        "tokens": tokens,
        "encoder_backend": f"sentence-transformers:{encoder_model}",
        "encoder_model": encoder_model,
        "vector_dim": vector_dim,
        "built_at": "2026-03-28T00:00:00+00:00",
    }
    if extra:
        meta.update(extra)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    matrix = np.arange(max(len(ids), 1) * vector_dim, dtype=np.float32).reshape(max(len(ids), 1), vector_dim)
    if not ids:
        matrix = np.empty((0, vector_dim), dtype=np.float32)
    else:
        matrix = matrix[: len(ids)]
    np.savez_compressed(vector_path, ids=np.array(ids, dtype=object), matrix=matrix)


def _seed_completed_corpus(
    tmp_path: Path,
    *,
    venue: str,
    year: int,
    track: str,
    ordinal: int,
    paper_id: str | None = None,
) -> Settings:
    settings = Settings.from_env(tmp_path, corpus=CorpusSpec.from_values(venue, year, track))
    release_root = settings.current_release_path
    (release_root / "normalized" / "deep_chat").mkdir(parents=True, exist_ok=True)
    (release_root / "indexes" / "layout").mkdir(parents=True, exist_ok=True)
    (release_root / "indexes" / "deep_chat").mkdir(parents=True, exist_ok=True)

    paper_id = paper_id or f"{year}.{venue}-{track}.{ordinal}"
    section_id = f"section_{paper_id}"
    object_id = f"object_{paper_id}"
    chunk_id = f"chunk_{paper_id}"
    evidence_id = f"evidence_{paper_id}"

    _write_jsonl(
        release_root / "normalized" / "papers.jsonl",
        [
            {
                "paper_id": paper_id,
                "anthology_id": paper_id,
                "title": f"title {paper_id}",
                "authors": ["tester"],
                "venue": venue,
                "year": year,
                "track": track,
                "url": f"https://example.com/{paper_id}",
                "abstract": f"abstract {paper_id}",
                "text": f"full text {paper_id}",
                "intro_summary": f"intro {paper_id}",
                "section_headings": ["Introduction"],
                "sections": ["Introduction"],
                "section_ids": [section_id],
                "object_ids": [object_id],
                "chunk_ids": [chunk_id],
                "typed_evidence_summary": {},
                "metadata": {},
            }
        ],
    )
    _write_jsonl(
        release_root / "normalized" / "sections.jsonl",
        [
            {
                "section_id": section_id,
                "paper_id": paper_id,
                "section_title": "Introduction",
                "section_path": ["Introduction"],
                "page_start": 1,
                "page_end": 1,
                "text": f"section text {paper_id}",
            }
        ],
    )
    _write_jsonl(
        release_root / "normalized" / "objects.jsonl",
        [
            {
                "object_id": object_id,
                "paper_id": paper_id,
                "section_id": section_id,
                "object_type": "figure_block",
                "ordinal": 0,
                "page_idx": 0,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "section_path": ["Introduction"],
                "caption": f"caption {paper_id}",
                "image_path": f"/images/{paper_id}.png",
            }
        ],
    )
    _write_jsonl(
        release_root / "normalized" / "chunks.jsonl",
        [
            {
                "chunk_id": chunk_id,
                "chunk_type": "text_chunk",
                "paper_id": paper_id,
                "section_id": section_id,
                "heading": "Introduction",
                "section_path": ["Introduction"],
                "page_start": 1,
                "page_end": 1,
                "char_start": 0,
                "char_end": 20,
                "token_count": 5,
                "text": f"chunk text {paper_id}",
            }
        ],
    )
    _write_jsonl(release_root / "normalized" / "parse_failures.jsonl", [])
    _write_jsonl(
        release_root / "normalized" / "deep_chat" / "evidence_units.jsonl",
        [
            {
                "evidence_id": evidence_id,
                "paper_id": paper_id,
                "section_id": section_id,
                "chunk_id": chunk_id,
                "evidence_type": "claim",
                "heading": "Introduction",
                "section_path": ["Introduction"],
                "page_start": 1,
                "page_end": 1,
                "text": f"evidence text {paper_id}",
                "html": None,
            }
        ],
    )

    layout_root = release_root / "indexes" / "layout"
    _write_index_bundle(
        layout_root,
        name="paper",
        ids=[paper_id],
        texts=[f"paper search text {paper_id}"],
        tokens=[[paper_id, "paper"]],
        vector_dim=4,
        encoder_model="allenai/specter2_base",
    )
    _write_index_bundle(
        layout_root,
        name="section",
        ids=[section_id],
        texts=[f"section search text {paper_id}"],
        tokens=[[paper_id, "section"]],
        vector_dim=6,
        encoder_model="BAAI/bge-m3",
        extra={
            "paper_ids": [paper_id],
            "section_titles": ["Introduction"],
            "section_paths": [["Introduction"]],
        },
    )
    _write_index_bundle(
        layout_root,
        name="chunk",
        ids=[chunk_id],
        texts=[f"chunk search text {paper_id}"],
        tokens=[[paper_id, "chunk"]],
        vector_dim=6,
        encoder_model="BAAI/bge-m3",
        extra={
            "paper_ids": [paper_id],
            "section_ids": [section_id],
            "chunk_types": ["text_chunk"],
        },
    )
    _write_index_bundle(
        layout_root,
        name="text_chunk",
        ids=[chunk_id],
        texts=[f"text chunk search text {paper_id}"],
        tokens=[[paper_id, "text_chunk"]],
        vector_dim=6,
        encoder_model="BAAI/bge-m3",
        extra={
            "paper_ids": [paper_id],
            "section_ids": [section_id],
        },
    )
    _write_index_bundle(
        layout_root,
        name="table_chunk",
        ids=[],
        texts=[],
        tokens=[],
        vector_dim=6,
        encoder_model="BAAI/bge-m3",
        extra={"paper_ids": [], "section_ids": []},
    )
    _write_index_bundle(
        layout_root,
        name="figure_chunk",
        ids=[],
        texts=[],
        tokens=[],
        vector_dim=6,
        encoder_model="BAAI/bge-m3",
        extra={"paper_ids": [], "section_ids": []},
    )
    _write_index_bundle(
        release_root / "indexes" / "deep_chat",
        name="evidence_unit",
        ids=[evidence_id],
        texts=[f"evidence search text {paper_id}"],
        tokens=[[paper_id, "evidence"]],
        vector_dim=6,
        encoder_model="BAAI/bge-m3",
        extra={
            "paper_ids": [paper_id],
            "evidence_types": ["claim"],
            "section_ids": [section_id],
        },
    )

    (layout_root / "index_state.json").write_text(
        json.dumps(
            {
                "built_at": "2026-03-28T00:00:00+00:00",
                "total_papers": 1,
                "papers": 1,
                "indexed_papers": 1,
                "failed_papers": 0,
                "sections": 1,
                "objects": 1,
                "chunks": 1,
                "text_chunks": 1,
                "table_chunks": 0,
                "figure_chunks": 0,
                "deep_chat_evidence_units": 1,
                "paper_dense_backend": "sentence-transformers:allenai/specter2_base",
                "chunk_dense_backend": "sentence-transformers:BAAI/bge-m3",
                "paper_dense_model": "allenai/specter2_base",
                "chunk_dense_model": "BAAI/bge-m3",
                "paper_vector_dim": 4,
                "chunk_vector_dim": 6,
                "pdf_parser_backend": "mineru_layout",
                "parse_failure_path": str(release_root / "normalized" / "parse_failures.jsonl"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    (settings.state_dir / "job_state.json").write_text(
        json.dumps(
            {
                "job_id": f"job_{paper_id}",
                "corpus": settings.corpus.to_dict(),
                "status": "completed",
                "updated_at": "2026-03-28T00:00:00+00:00",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return settings


def test_rebuild_search_current_merges_completed_corpora(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.toml").write_text(
        """
[openai]
enabled = false
base_url = "https://api.openai.com/v1"
model = "gpt-5.4-mini"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(tmp_path / "data"))
    settings_one = _seed_completed_corpus(tmp_path, venue="acl", year=2025, track="long", ordinal=1)
    settings_two = _seed_completed_corpus(tmp_path, venue="acl", year=2024, track="long", ordinal=2)

    manifest = rebuild_search_current(tmp_path)

    search_current_root = settings_one.search_current_dir
    assert search_current_root.exists()
    assert manifest["counts"]["papers"] == 2
    assert len(manifest["corpora"]) == 2

    merged_papers = [
        json.loads(line)
        for line in (search_current_root / "normalized" / "papers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {item["paper_id"] for item in merged_papers} == {
        "2025.acl-long.1",
        "2024.acl-long.2",
    }

    merged_index_state = json.loads(
        (search_current_root / "indexes" / "layout" / "index_state.json").read_text(encoding="utf-8")
    )
    assert merged_index_state["papers"] == 2
    assert merged_index_state["deep_chat_evidence_units"] == 2

    vectors = np.load(search_current_root / "indexes" / "layout" / "paper_vectors.npz", allow_pickle=True)
    assert len(vectors["ids"].tolist()) == 2
    assert vectors["matrix"].shape == (2, 4)

    deep_chat_vectors = np.load(
        search_current_root / "indexes" / "deep_chat" / "evidence_unit_vectors.npz",
        allow_pickle=True,
    )
    assert len(deep_chat_vectors["ids"].tolist()) == 2
    assert deep_chat_vectors["matrix"].shape == (2, 6)

    assert settings_two.search_current_manifest_path.exists()


def test_rebuild_search_current_skips_overlap_between_specific_track_and_all(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.toml").write_text(
        """
[openai]
enabled = false
base_url = "https://api.openai.com/v1"
model = "gpt-5.4-mini"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(tmp_path / "data"))
    shared_paper_id = "2025.acl-overlap.1"
    _seed_completed_corpus(
        tmp_path,
        venue="acl",
        year=2025,
        track="long",
        ordinal=1,
        paper_id=shared_paper_id,
    )
    _seed_completed_corpus(
        tmp_path,
        venue="acl",
        year=2025,
        track="all",
        ordinal=1,
        paper_id=shared_paper_id,
    )

    manifest = rebuild_search_current(tmp_path)
    search_current_root = Settings.from_env(tmp_path).search_current_dir
    merged_papers = [
        json.loads(line)
        for line in (search_current_root / "normalized" / "papers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(merged_papers) == 1
    assert merged_papers[0]["paper_id"] == shared_paper_id
    assert len(manifest["corpora"]) == 2


def test_rebuild_search_current_preserves_existing_traces_and_float32_vectors(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.toml").write_text(
        """
[openai]
enabled = false
base_url = "https://api.openai.com/v1"
model = "gpt-5.4-mini"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(tmp_path / "data"))
    settings = _seed_completed_corpus(tmp_path, venue="acl", year=2025, track="long", ordinal=1)

    rebuild_search_current(tmp_path)
    trace_path = settings.search_current_dir / "traces" / "trace-kept.json"
    trace_path.write_text('{"trace_id":"trace-kept"}', encoding="utf-8")

    rebuild_search_current(tmp_path)

    assert trace_path.exists()
    _, store_vectors = LocalStore(settings, root_dir=settings.search_current_dir).load_vectors("paper")
    assert store_vectors.dtype == np.float32


def test_rebuild_search_current_can_publish_selected_corpus_only(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.toml").write_text(
        """
[openai]
enabled = false
base_url = "https://api.openai.com/v1"
model = "gpt-5.4-mini"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(tmp_path / "data"))
    settings_2025 = _seed_completed_corpus(tmp_path, venue="acl", year=2025, track="long", ordinal=1)
    _seed_completed_corpus(tmp_path, venue="acl", year=2024, track="long", ordinal=2)

    manifest = rebuild_search_current(tmp_path, corpora=[CorpusSpec.from_values("acl", 2024, "long")])

    search_current_root = settings_2025.search_current_dir
    merged_papers = [
        json.loads(line)
        for line in (search_current_root / "normalized" / "papers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["counts"]["papers"] == 1
    assert [item["corpus"] for item in manifest["corpora"]] == ["acl/2024/long"]
    assert [item["paper_id"] for item in merged_papers] == ["2024.acl-long.2"]


def test_rebuild_search_current_allows_selected_running_corpus_when_requested(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.toml").write_text(
        """
[openai]
enabled = false
base_url = "https://api.openai.com/v1"
model = "gpt-5.4-mini"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHEMVERIFY_DATA_DIR", str(tmp_path / "data"))
    settings = _seed_completed_corpus(tmp_path, venue="acl", year=2024, track="long", ordinal=1)
    state_path = settings.state_dir / "job_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["status"] = "running"
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = rebuild_search_current(
        tmp_path,
        corpora=[CorpusSpec.from_values("acl", 2024, "long")],
        allow_uncompleted_selected=True,
    )

    assert manifest["counts"]["papers"] == 1
    assert [item["corpus"] for item in manifest["corpora"]] == ["acl/2024/long"]
