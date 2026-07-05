from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from chemverify.config import CorpusSpec, Settings
from chemverify.models import BuildIndexSummary, PaperRecord, ParseFailureRecord
from chemverify.offline import (
    IncrementalIndexBuilder,
    OfflineJobController,
    OfflineRunner,
    PauseRequested,
    _publish_release_snapshot,
    request_pause,
)


def _make_paper(*, paper_id: str, pdf_path: Path) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        anthology_id=paper_id,
        title=f"title {paper_id}",
        authors=["tester"],
        venue="acl",
        year=2025,
        track="long",
        url=f"https://example.com/{paper_id}",
        local_pdf_path=str(pdf_path),
    )


def _make_failure(*, paper_id: str, message: str) -> ParseFailureRecord:
    return ParseFailureRecord(
        paper_id=paper_id,
        venue="acl",
        year=2025,
        track="long",
        parser_backend="mineru_layout",
        stage="parse",
        error_type="test_failure",
        error_message=message,
        analysis="test analysis",
        suggestion="test suggestion",
        occurred_at="2026-03-27T00:00:00+00:00",
    )


def _build_summary() -> BuildIndexSummary:
    return BuildIndexSummary(
        papers=1,
        total_papers=1,
        indexed_papers=1,
        failed_papers=0,
        paper_vector_dim=768,
        chunk_vector_dim=1024,
        paper_dense_backend="sentence-transformers:allenai/specter2_base",
        chunk_dense_backend="sentence-transformers:BAAI/bge-m3",
        pdf_parser_backend="mineru_layout",
        built_at="2026-03-27T00:00:00+00:00",
    )


class _ControllerStub:
    def check_pause_requested(self) -> None:
        return None

    def update_progress(self, **_: object) -> None:
        return None

    def update_state(self, **_: object) -> None:
        return None


def _fake_encoder_init(self, config) -> None:
    self.config = config
    self._model = object()
    self.backend_name = f"sentence-transformers:{config.model_name}"


def _fake_encoder_encode(self, texts: list[str], *, progress_callback=None) -> np.ndarray:
    matrix = np.zeros((len(texts), 8), dtype=np.float32)
    if progress_callback is not None:
        progress_callback(len(texts), len(texts))
    return matrix


def test_clear_for_rebuild_preserves_live_outputs(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path, corpus=CorpusSpec.from_values("acl", 2025, "long"))
    runner = OfflineRunner(settings)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    paper = _make_paper(paper_id="2025.acl-long.1", pdf_path=pdf_path)

    live_normalized_file = settings.normalized_dir / "papers.jsonl"
    live_index_file = settings.index_dir / "index_state.json"
    live_normalized_file.parent.mkdir(parents=True, exist_ok=True)
    live_index_file.parent.mkdir(parents=True, exist_ok=True)
    live_normalized_file.write_text("live-normalized\n", encoding="utf-8")
    live_index_file.write_text('{"status":"live"}', encoding="utf-8")

    work_file = settings.work_dir / "build" / "build_state.json"
    work_file.parent.mkdir(parents=True, exist_ok=True)
    work_file.write_text('{"status":"running"}', encoding="utf-8")

    artifact_dir = settings.mineru_output_dir / paper.paper_id / "auto"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / f"{paper.paper_id}.md").write_text("# parsed", encoding="utf-8")
    (artifact_dir / f"{paper.paper_id}_content_list.json").write_text("[]", encoding="utf-8")
    (artifact_dir / f"{paper.paper_id}_middle.json").write_text("{}", encoding="utf-8")

    settings.mineru_failure_manifest_path.write_text(
        json.dumps({"paper_id": paper.paper_id, "error_message": "old failure"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    runner._clear_for_rebuild([paper])

    assert live_normalized_file.read_text(encoding="utf-8") == "live-normalized\n"
    assert live_index_file.read_text(encoding="utf-8") == '{"status":"live"}'
    assert not work_file.exists()
    assert not artifact_dir.parent.exists()
    assert not settings.mineru_failure_manifest_path.exists()


def test_publish_release_snapshot_switches_current_symlink(tmp_path: Path) -> None:
    current_link = tmp_path / "release" / "current"
    old_snapshot = tmp_path / "release" / "snapshots" / "old"
    new_snapshot = tmp_path / "release" / "snapshots" / "new"

    (old_snapshot / "normalized").mkdir(parents=True, exist_ok=True)
    (new_snapshot / "normalized").mkdir(parents=True, exist_ok=True)
    (old_snapshot / "normalized" / "papers.jsonl").write_text("old", encoding="utf-8")
    (new_snapshot / "normalized" / "papers.jsonl").write_text("new", encoding="utf-8")

    current_link.parent.mkdir(parents=True, exist_ok=True)
    current_link.symlink_to(Path(os.path.relpath(old_snapshot, start=current_link.parent)))

    _publish_release_snapshot(snapshot_root=new_snapshot, current_link=current_link)

    assert current_link.is_symlink()
    assert current_link.resolve() == new_snapshot.resolve()
    assert (current_link / "normalized" / "papers.jsonl").read_text(encoding="utf-8") == "new"


def test_build_completed_skip_requires_matching_release_signature(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(tmp_path, corpus=CorpusSpec.from_values("acl", 2025, "long"))
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    source_paper = _make_paper(paper_id="2025.acl-long.1", pdf_path=pdf_path)
    monkeypatch.setattr("chemverify.encoders.SentenceTransformerEncoder.__init__", _fake_encoder_init)
    monkeypatch.setattr("chemverify.encoders.SentenceTransformerEncoder.encode", _fake_encoder_encode)
    builder = IncrementalIndexBuilder(settings, _ControllerStub())
    summary = _build_summary()

    seed_snapshot = settings.release_snapshots_dir / "seed"
    (seed_snapshot / "normalized").mkdir(parents=True, exist_ok=True)
    (seed_snapshot / "indexes" / "layout").mkdir(parents=True, exist_ok=True)
    (seed_snapshot / "indexes" / "deep_chat").mkdir(parents=True, exist_ok=True)
    (seed_snapshot / "normalized" / "papers.jsonl").write_text("{}", encoding="utf-8")
    (seed_snapshot / "indexes" / "layout" / "paper_vectors.npz").write_bytes(b"vectors")
    (seed_snapshot / "indexes" / "layout" / "index_state.json").write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (seed_snapshot / "indexes" / "deep_chat" / "evidence_unit_vectors.npz").write_bytes(b"vectors")
    settings.current_release_path.parent.mkdir(parents=True, exist_ok=True)
    settings.current_release_path.symlink_to(Path(os.path.relpath(seed_snapshot, start=settings.current_release_path.parent)))

    release_signature = builder._release_signature(
        all_manifest_papers=[source_paper],
        terminal_parse_failures=[],
        total_manifest_papers=1,
    )
    builder._save_meta(
        {
            "fingerprint": builder._build_fingerprint([source_paper]),
            "release_signature": release_signature,
            "status": "completed",
        }
    )

    prepare_called = {"value": False}
    finalize_called = {"value": False}

    def fail_prepare(_: list[PaperRecord]) -> None:
        prepare_called["value"] = True

    def fake_finalize(*args, **kwargs) -> BuildIndexSummary:
        finalize_called["value"] = True
        return summary

    monkeypatch.setattr(builder, "_prepare_bundles", fail_prepare)
    monkeypatch.setattr(
        builder,
        "_aggregate_records",
        lambda _: {
            "papers": [],
            "sections": [],
            "objects": [],
            "chunks": [],
            "evidence_units": [],
            "prepare_failures": [],
        },
    )
    monkeypatch.setattr(builder, "_encode_all", lambda _: None)
    monkeypatch.setattr(builder, "_finalize", fake_finalize)

    same_summary = builder.build(
        source_papers=[source_paper],
        all_manifest_papers=[source_paper],
        terminal_parse_failures=[],
        mode="resume",
        total_manifest_papers=1,
    )
    assert same_summary == summary
    assert not prepare_called["value"]
    assert not finalize_called["value"]

    changed_summary = builder.build(
        source_papers=[source_paper],
        all_manifest_papers=[source_paper],
        terminal_parse_failures=[_make_failure(paper_id=source_paper.paper_id, message="new failure metadata")],
        mode="resume",
        total_manifest_papers=1,
    )
    assert changed_summary == summary
    assert prepare_called["value"]
    assert finalize_called["value"]


def test_job_state_is_reinitialized_for_each_run(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(tmp_path, corpus=CorpusSpec.from_values("acl", 2025, "long"))

    timestamps = iter([1000.0, 1001.0])
    iso_values = iter(
        [
            "2026-03-27T10:00:00+00:00",
            "2026-03-27T10:01:00+00:00",
            "2026-03-27T10:05:00+00:00",
            "2026-03-27T10:10:00+00:00",
        ]
    )
    monkeypatch.setattr("chemverify.offline.time.time", lambda: next(timestamps))
    monkeypatch.setattr("chemverify.offline.now_iso", lambda: next(iso_values))

    controller_one = OfflineJobController(settings, mode="resume")
    controller_two = OfflineJobController(settings, mode="resume")

    controller_one.start()
    controller_one.update_state(status="paused", phase="build_encode", message="first run paused")
    first_state = json.loads(controller_one.job_state_path.read_text(encoding="utf-8"))
    controller_one.close()

    controller_two.start()
    second_state = json.loads(controller_two.job_state_path.read_text(encoding="utf-8"))
    last_job = json.loads(settings.last_job_path.read_text(encoding="utf-8"))
    controller_two.close()

    assert first_state["job_id"] != second_state["job_id"]
    assert second_state["job_id"] == controller_two.job_id
    assert second_state["started_at"] != first_state["started_at"]
    assert second_state["message"] == "Offline job started."
    assert "progress" not in second_state
    assert last_job["job_id"] == controller_two.job_id


def test_job_id_is_unique_even_within_same_second(tmp_path: Path, monkeypatch) -> None:
    settings = Settings.from_env(tmp_path, corpus=CorpusSpec.from_values("acl", 2025, "long"))
    values = iter([1_000_000_001, 1_000_000_002])
    monkeypatch.setattr("chemverify.offline.time.time_ns", lambda: next(values))

    controller_one = OfflineJobController(settings, mode="resume")
    controller_two = OfflineJobController(settings, mode="resume")

    assert controller_one.job_id != controller_two.job_id


def test_pause_is_controlled_only_by_external_request(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path, corpus=CorpusSpec.from_values("acl", 2025, "long"))
    controller = OfflineJobController(settings, mode="resume")

    controller.start()
    active_job = json.loads(settings.active_job_path.read_text(encoding="utf-8"))
    control_payload = json.loads(Path(active_job["control_path"]).read_text(encoding="utf-8"))
    assert control_payload == {"pause_requested": False, "requested_at": None}

    pause_payload = request_pause(settings)
    assert pause_payload["status"] == "pause_requested"

    with pytest.raises(PauseRequested):
        controller.check_pause_requested()

    controller.close()
