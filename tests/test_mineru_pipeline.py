from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from chemverify.mineru_pipeline import BatchItem, MinerUPipelineConfig, _build_batches, _progress_heartbeat, _run_batch
from chemverify.models import PaperRecord


def _batch_item(pdf_path: Path) -> BatchItem:
    return BatchItem(
        paper=PaperRecord(
            paper_id="paper-1",
            title="Paper 1",
            venue="test",
            year=2026,
            track="demo",
            url=pdf_path.as_uri(),
            pdf_url=pdf_path.as_uri(),
            local_pdf_path=str(pdf_path),
        ),
        pdf_path=pdf_path,
        pages=1,
    )


def test_run_batch_quiet_output_hides_noisy_parser_output(tmp_path: Path, monkeypatch, capsys) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr("chemverify.mineru_pipeline.read_fn", lambda path: b"pdf")

    def _fake_parse(**kwargs) -> None:
        print("noisy stdout")
        print("noisy stderr", file=sys.stderr)

    monkeypatch.setattr("chemverify.mineru_pipeline.do_parse", _fake_parse)

    _run_batch(
        [_batch_item(pdf_path)],
        output_dir=tmp_path / "out",
        lang="en",
        parse_method="txt",
        backend="pipeline",
        formula=False,
        table=True,
        quiet_output=True,
    )

    captured = capsys.readouterr()
    assert "noisy" not in captured.out
    assert "noisy" not in captured.err


def test_run_batch_quiet_output_exposes_text_stream_attributes(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr("chemverify.mineru_pipeline.read_fn", lambda path: b"pdf")

    def _fake_parse(**kwargs) -> None:
        assert sys.stdout.encoding
        assert sys.stderr.encoding
        assert not sys.stdout.isatty()

    monkeypatch.setattr("chemverify.mineru_pipeline.do_parse", _fake_parse)

    _run_batch(
        [_batch_item(pdf_path)],
        output_dir=tmp_path / "out",
        lang="en",
        parse_method="txt",
        backend="pipeline",
        formula=False,
        table=True,
        quiet_output=True,
    )


def test_progress_heartbeat_updates_while_batch_is_blocked() -> None:
    class _Progress:
        def __init__(self) -> None:
            self.calls = 0
            self.last = {}

        def mineru_heartbeat(self, **kwargs) -> None:
            self.calls += 1
            self.last = kwargs

    progress = _Progress()

    with _progress_heartbeat(progress, batch_index=1, batches=2, papers=5, pages=115, interval=0.01):
        time.sleep(0.03)

    assert progress.calls > 0
    assert progress.last["current"] == "current PDF"


def test_progress_heartbeat_reports_queued_cancel() -> None:
    class _Progress:
        def __init__(self) -> None:
            self.last = {}

        def mineru_heartbeat(self, **kwargs) -> None:
            self.last = kwargs

    progress = _Progress()

    with _progress_heartbeat(
        progress,
        batch_index=1,
        batches=1,
        papers=1,
        pages=12,
        current="paper.pdf",
        cancel_requested=lambda: True,
        interval=0.01,
    ):
        time.sleep(0.03)

    assert progress.last["cancel_requested"] is True
    assert progress.last["current"] == "paper.pdf"


def test_default_mineru_batches_are_paper_sized_for_responsive_progress(tmp_path: Path) -> None:
    pdfs = []
    for index in range(3):
        pdf_path = tmp_path / f"paper-{index}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        pdfs.append(BatchItem(paper=_batch_item(pdf_path).paper, pdf_path=pdf_path, pages=20))

    config = MinerUPipelineConfig()
    batches = _build_batches(pdfs, max_pdfs=config.max_pdfs_per_batch, max_pages=config.max_pages_per_batch)

    assert [len(batch) for batch in batches] == [1, 1, 1]


def test_run_batch_quiet_output_keeps_failure_tail(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr("chemverify.mineru_pipeline.read_fn", lambda path: b"pdf")

    def _fake_parse(**kwargs) -> None:
        print("parser tail")
        raise ValueError("boom")

    monkeypatch.setattr("chemverify.mineru_pipeline.do_parse", _fake_parse)

    with pytest.raises(RuntimeError, match="MinerU output tail"):
        _run_batch(
            [_batch_item(pdf_path)],
            output_dir=tmp_path / "out",
            lang="en",
            parse_method="txt",
            backend="pipeline",
            formula=False,
            table=True,
            quiet_output=True,
        )
