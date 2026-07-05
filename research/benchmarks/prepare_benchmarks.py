#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


PASS_BENCHMARK_REPO = "CarlanLark/pasa-dataset"
PEERQA_REPO = "UKPLab/PeerQA"
QASPER_REPO = "allenai/qasper"

PEERQA_URL = (
    "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/"
    "tudatalib/4467/peerqa-data-v1.0.zip?sequence=5&isAllowed=y"
)
QASPER_TRAIN_DEV_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
)
QASPER_TEST_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
)


def log(msg: str) -> None:
    print(msg, flush=True)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def dedupe_preserve(items: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else item
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def clean_scalar(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"[skip] 已存在: {dest}")
        return dest
    ensure_dir(dest.parent)
    log(f"[download] {url} -> {dest}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return dest


def is_safe_member(base: Path, name: str) -> bool:
    target = (base / name).resolve()
    return str(target).startswith(str(base.resolve()))


def extract_zip(archive: Path, dest: Path) -> None:
    ensure_dir(dest)
    marker = dest / ".extracted"
    if marker.exists():
        log(f"[skip] zip 已解压: {archive}")
        return
    log(f"[extract] zip {archive} -> {dest}")
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if not is_safe_member(dest, member.filename):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(dest)
    marker.write_text("ok\n")


def extract_tgz(archive: Path, dest: Path) -> None:
    ensure_dir(dest)
    marker = dest / f".{archive.stem}.extracted"
    if marker.exists():
        log(f"[skip] tgz 已解压: {archive}")
        return
    log(f"[extract] tgz {archive} -> {dest}")
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            if not is_safe_member(dest, member.name):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tf.extractall(dest)
    marker.write_text("ok\n")


def snapshot_dataset(repo_id: str, dest: Path, token: str | None) -> Path:
    ensure_dir(dest)
    log(f"[hf] snapshot {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        token=token,
    )
    return dest


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_pasa(root: Path, token: str | None) -> dict[str, Any]:
    benchmark_root = root / "data" / "benchmarks" / "pasa"
    snapshot_root = benchmark_root / "raw" / "hf_snapshot"
    normalized_root = ensure_dir(benchmark_root / "normalized")
    analysis_root = ensure_dir(benchmark_root / "analysis")

    snapshot_dataset(PASS_BENCHMARK_REPO, snapshot_root, token)

    query_specs = [
        ("AutoScholarQuery", "train", snapshot_root / "AutoScholarQuery" / "train.jsonl"),
        ("AutoScholarQuery", "dev", snapshot_root / "AutoScholarQuery" / "dev.jsonl"),
        ("AutoScholarQuery", "test", snapshot_root / "AutoScholarQuery" / "test.jsonl"),
        ("RealScholarQuery", "test", snapshot_root / "RealScholarQuery" / "test.jsonl"),
    ]

    summary: dict[str, Any] = {"benchmark": "pasa", "splits": {}}

    for dataset_name, split, path in query_specs:
        rows = load_jsonl(path)
        normalized = []
        for row in rows:
            normalized.append(
                {
                    "benchmark_name": "pasa",
                    "task_type": "paper_search",
                    "dataset_name": dataset_name,
                    "split": split,
                    "query_id": row["qid"],
                    "query_text": row["question"],
                    "gold_paper_ids": row.get("answer_arxiv_id", []),
                    "gold_titles": row.get("answer", []),
                    "source_metadata": row.get("source_meta", {}),
                }
            )
        out_path = normalized_root / "search_queries" / dataset_name / f"{split}.jsonl"
        write_jsonl(normalized, out_path)
        summary["splits"][f"{dataset_name}:{split}"] = {
            "rows": len(normalized),
            "path": str(out_path),
            "sample_query_id": normalized[0]["query_id"] if normalized else None,
        }

    id2paper_path = snapshot_root / "paper_database" / "id2paper.json"
    with id2paper_path.open("r", encoding="utf-8") as f:
        id2paper = json.load(f)
    paper_catalog = [
        {
            "benchmark_name": "pasa",
            "task_type": "paper_search_corpus",
            "paper_id": paper_id,
            "title": title,
        }
        for paper_id, title in id2paper.items()
    ]
    catalog_path = normalized_root / "paper_catalog.jsonl"
    write_jsonl(paper_catalog, catalog_path)

    zip_path = snapshot_root / "paper_database" / "cs_paper_2nd.zip"
    zip_count = 0
    zip_sample = None
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zip_count = len(names)
        if names:
            sample_name = names[0]
            with zf.open(sample_name) as f:
                sample = json.loads(f.read().decode("utf-8"))
            zip_sample = {
                "entry_name": sample_name,
                "keys": list(sample.keys()),
                "title": sample.get("title"),
            }

    summary.update(
        {
            "paper_catalog_rows": len(paper_catalog),
            "paper_catalog_path": str(catalog_path),
            "paper_zip_path": str(zip_path),
            "paper_zip_entries": zip_count,
            "paper_zip_sample": zip_sample,
        }
    )
    write_json(summary, analysis_root / "summary.json")
    return summary


def group_peerqa_papers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["paper_id"]].append(row)
    papers = []
    for paper_id, blocks in grouped.items():
        blocks = sorted(blocks, key=lambda x: (x["pidx"], x["sidx"], x["idx"]))
        title = next((b["content"] for b in blocks if b.get("type") == "title"), None)
        papers.append(
            {
                "benchmark_name": "peerqa",
                "task_type": "paper_chat_corpus",
                "paper_id": paper_id,
                "title": title,
                "blocks": [
                    {
                        "block_id": str(block["idx"]),
                        "paragraph_id": block["pidx"],
                        "sentence_id": block["sidx"],
                        "block_type": block["type"],
                        "section_heading": block.get("last_heading"),
                        "text": block["content"],
                    }
                    for block in blocks
                ],
            }
        )
    return papers


def normalize_peerqa(root: Path, token: str | None) -> dict[str, Any]:
    del token
    benchmark_root = root / "data" / "benchmarks" / "peerqa"
    snapshot_root = benchmark_root / "raw" / "hf_snapshot"
    source_root = ensure_dir(benchmark_root / "raw" / "source_payloads")
    extracted_root = ensure_dir(benchmark_root / "raw" / "extracted")
    normalized_root = ensure_dir(benchmark_root / "normalized")
    analysis_root = ensure_dir(benchmark_root / "analysis")

    snapshot_dataset(PEERQA_REPO, snapshot_root, None)
    zip_path = download_file(PEERQA_URL, source_root / "peerqa-data-v1.0.zip")
    extract_zip(zip_path, extracted_root)

    qa_path = extracted_root / "qa.jsonl"
    qa_aug_path = extracted_root / "qa-augmented-answers.jsonl"
    papers_path = extracted_root / "papers.jsonl"

    qa_rows = load_jsonl(qa_path)
    qa_augmented = {}
    if qa_aug_path.exists():
        for row in load_jsonl(qa_aug_path):
            qa_augmented[row["question_id"]] = row

    normalized_questions = []
    qa_paper_ids = set()
    for row in qa_rows:
        qa_paper_ids.add(row["paper_id"])
        aug = qa_augmented.get(row["question_id"], {})
        evidence_idx = []
        for item in row.get("answer_evidence_mapped") or []:
            evidence_idx.extend([idx for idx in item.get("idx", []) if idx is not None])
        gold_answers = dedupe_preserve(
            [
                answer
                for answer in [
                    clean_scalar(row.get("answer_free_form_augmented")),
                    clean_scalar(row.get("answer_free_form")),
                    clean_scalar(aug.get("augmented_answer_free_form")),
                ]
                if answer not in (None, "", "nan")
            ]
        )
        normalized_questions.append(
            {
                "benchmark_name": "peerqa",
                "task_type": "paper_chat",
                "split": "test",
                "paper_id": row["paper_id"],
                "question_id": row["question_id"],
                "question_text": row["question"],
                "answerable": row.get("answerable_mapped", row.get("answerable")),
                "gold_answers": gold_answers,
                "gold_evidence_texts": dedupe_preserve(row.get("answer_evidence_sent") or row.get("raw_answer_evidence") or []),
                "gold_evidence_block_ids": dedupe_preserve([str(idx) for idx in evidence_idx]),
                "raw_answerable": row.get("answerable"),
                "source_metadata": {
                    "answerable_mapped": row.get("answerable_mapped"),
                    "raw_answer_evidence_count": len(row.get("raw_answer_evidence") or []),
                },
            }
        )

    questions_path = normalized_root / "paper_chat_questions" / "test.jsonl"
    write_jsonl(normalized_questions, questions_path)

    papers = []
    if papers_path.exists():
        papers = group_peerqa_papers(load_jsonl(papers_path))
        write_jsonl(papers, normalized_root / "papers.jsonl")
    paper_ids_with_text = {paper["paper_id"] for paper in papers}
    questions_with_paper_text = sum(1 for row in normalized_questions if row["paper_id"] in paper_ids_with_text)
    unique_question_papers = len(qa_paper_ids)

    qrels_summaries = {}
    for filename in sorted(extracted_root.glob("qrels-*.jsonl")):
        rows = load_jsonl(filename)
        qrels_summaries[filename.name] = len(rows)
        write_jsonl(rows, normalized_root / "qrels" / filename.name)

    extracted_files = sorted(str(p.relative_to(extracted_root)) for p in extracted_root.rglob("*") if p.is_file())
    summary = {
        "benchmark": "peerqa",
        "question_rows": len(normalized_questions),
        "papers_rows": len(papers),
        "qa_unique_papers": unique_question_papers,
        "papers_with_text": len(paper_ids_with_text),
        "questions_with_available_paper_text": questions_with_paper_text,
        "qrels": qrels_summaries,
        "questions_path": str(questions_path),
        "extracted_files": extracted_files[:100],
    }
    write_json(summary, analysis_root / "summary.json")
    return summary


def normalize_qasper_answers(answers: list[dict[str, Any]]) -> tuple[list[str], bool | None, list[str], list[str], list[dict[str, Any]]]:
    gold_answers: list[str] = []
    answerable_votes: list[bool] = []
    evidence_texts: list[str] = []
    highlighted: list[str] = []
    raw_annotations: list[dict[str, Any]] = []

    for answer_entry in answers:
        answer = answer_entry.get("answer", {})
        answerable = not bool(answer.get("unanswerable"))
        answerable_votes.append(answerable)

        free_form = answer.get("free_form_answer")
        yes_no = answer.get("yes_no")
        extractive = answer.get("extractive_spans") or []
        if free_form:
            gold_answers.append(free_form)
        elif extractive:
            gold_answers.extend([span for span in extractive if span])
        elif answer.get("unanswerable"):
            gold_answers.append("unanswerable")
        elif isinstance(yes_no, bool):
            gold_answers.append("yes" if yes_no else "no")

        evidence_texts.extend(answer.get("evidence") or [])
        highlighted.extend(answer.get("highlighted_evidence") or [])
        raw_annotations.append(answer_entry)

    answerable_majority = None
    if answerable_votes:
        answerable_majority = sum(1 for x in answerable_votes if x) >= (len(answerable_votes) / 2)
    return (
        dedupe_preserve([answer for answer in gold_answers if answer not in (None, "", "nan")]),
        answerable_majority,
        dedupe_preserve([text for text in evidence_texts if text]),
        dedupe_preserve([text for text in highlighted if text]),
        raw_annotations,
    )


def normalize_qasper(root: Path, token: str | None) -> dict[str, Any]:
    del token
    benchmark_root = root / "data" / "benchmarks" / "qasper"
    snapshot_root = benchmark_root / "raw" / "hf_snapshot"
    source_root = ensure_dir(benchmark_root / "raw" / "source_payloads")
    extracted_root = ensure_dir(benchmark_root / "raw" / "extracted")
    normalized_root = ensure_dir(benchmark_root / "normalized")
    analysis_root = ensure_dir(benchmark_root / "analysis")

    snapshot_dataset(QASPER_REPO, snapshot_root, None)
    train_dev_tgz = download_file(QASPER_TRAIN_DEV_URL, source_root / "qasper-train-dev-v0.3.tgz")
    test_tgz = download_file(QASPER_TEST_URL, source_root / "qasper-test-and-evaluator-v0.3.tgz")
    extract_tgz(train_dev_tgz, extracted_root)
    extract_tgz(test_tgz, extracted_root)

    split_files = {
        "train": extracted_root / "qasper-train-v0.3.json",
        "validation": extracted_root / "qasper-dev-v0.3.json",
        "test": extracted_root / "qasper-test-v0.3.json",
    }

    paper_records: dict[str, dict[str, Any]] = {}
    split_counts: dict[str, int] = {}
    all_questions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for split, path in split_files.items():
        with path.open("r", encoding="utf-8") as f:
            papers = json.load(f)
        count = 0
        for paper_id, paper in papers.items():
            if paper_id not in paper_records:
                blocks = []
                for sec_idx, section in enumerate(paper.get("full_text", [])):
                    section_name = section.get("section_name")
                    for para_idx, para in enumerate(section.get("paragraphs", [])):
                        blocks.append(
                            {
                                "block_id": f"s{sec_idx}_p{para_idx}",
                                "section_name": section_name,
                                "text": para,
                            }
                        )
                paper_records[paper_id] = {
                    "benchmark_name": "qasper",
                    "task_type": "paper_chat_corpus",
                    "paper_id": paper_id,
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                    "blocks": blocks,
                    "figures_and_tables": paper.get("figures_and_tables", []),
                }

            for qa in paper.get("qas", []):
                count += 1
                gold_answers, answerable_majority, evidence_texts, highlighted, raw_annotations = normalize_qasper_answers(
                    qa.get("answers", [])
                )
                all_questions[split].append(
                    {
                        "benchmark_name": "qasper",
                        "task_type": "paper_chat",
                        "split": split,
                        "paper_id": paper_id,
                        "question_id": qa.get("question_id"),
                        "question_text": qa.get("question"),
                        "answerable": answerable_majority,
                        "gold_answers": gold_answers,
                        "gold_evidence_texts": evidence_texts,
                        "gold_highlighted_evidence": highlighted,
                        "source_metadata": {
                            "nlp_background": qa.get("nlp_background"),
                            "topic_background": qa.get("topic_background"),
                            "paper_read": qa.get("paper_read"),
                            "search_query": qa.get("search_query"),
                            "question_writer": qa.get("question_writer"),
                            "annotation_count": len(raw_annotations),
                        },
                    }
                )
        split_counts[split] = count

    write_jsonl(list(paper_records.values()), normalized_root / "papers.jsonl")
    for split, rows in all_questions.items():
        write_jsonl(rows, normalized_root / "paper_chat_questions" / f"{split}.jsonl")

    summary = {
        "benchmark": "qasper",
        "paper_rows": len(paper_records),
        "question_rows_by_split": split_counts,
        "papers_path": str(normalized_root / "papers.jsonl"),
        "question_paths": {
            split: str(normalized_root / "paper_chat_questions" / f"{split}.jsonl")
            for split in split_counts
        },
    }
    write_json(summary, analysis_root / "summary.json")
    return summary


def build_registry(root: Path, summaries: list[dict[str, Any]]) -> None:
    registry = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "benchmarks": summaries,
        "task_families": {
            "paper_search": {
                "record_keys": [
                    "benchmark_name",
                    "task_type",
                    "dataset_name",
                    "split",
                    "query_id",
                    "query_text",
                    "gold_paper_ids",
                    "gold_titles",
                    "source_metadata",
                ]
            },
            "paper_chat": {
                "record_keys": [
                    "benchmark_name",
                    "task_type",
                    "split",
                    "paper_id",
                    "question_id",
                    "question_text",
                    "answerable",
                    "gold_answers",
                    "gold_evidence_texts",
                    "source_metadata",
                ]
            },
        },
    }
    write_json(registry, root / "data" / "benchmarks" / "registry.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/ChemVerify"))
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    root = args.project_root.resolve()
    summaries = [
        normalize_pasa(root, args.hf_token),
        normalize_peerqa(root, args.hf_token),
        normalize_qasper(root, args.hf_token),
    ]
    build_registry(root, summaries)
    log("[done] benchmark 下载、解包、归一化与注册完成")


if __name__ == "__main__":
    main()
