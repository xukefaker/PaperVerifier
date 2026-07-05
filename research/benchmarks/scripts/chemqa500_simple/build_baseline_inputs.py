#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")
ABSTRACT_INLINE_RE = re.compile(
    r"\bABSTRACT\s*:\s*(.+?)(?=\b(?:INTRODUCTION|KEYWORDS|AUTHOR INFORMATION|REFERENCES|ACKNOWLEDGMENTS?)\b|$)",
    re.IGNORECASE | re.DOTALL,
)

SKIP_SECTION_HEADINGS = {
    "ACCESS",
    "ARTICLE RECOMMENDATIONS",
    "METRICS",
    "METRICS & MORE",
    "READ ONLINE",
}

STOP_SECTION_HEADINGS = {
    "ASSOCIATED CONTENT",
    "AUTHOR INFORMATION",
    "AUTHOR CONTRIBUTIONS",
    "NOTES",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENTS",
    "REFERENCES",
    "SUPPORTING INFORMATION",
    "CONFLICT OF INTERESTS",
    "CONFLICTS OF INTEREST",
    "DATA AVAILABILITY STATEMENT",
}


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\u00ad", "")
    text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_heading(text: str | None) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^A-Za-z0-9 &]+", " ", text)
    return WHITESPACE_RE.sub(" ", text).strip().upper()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def sort_paper_id(paper_id: str) -> tuple[int, str]:
    return (int(paper_id), paper_id) if str(paper_id).isdigit() else (10**9, str(paper_id))


def extract_abstract(paper: dict[str, Any], sections: list[dict[str, Any]]) -> tuple[str, str]:
    abstract = normalize_text(paper.get("abstract"))
    if len(abstract.split()) >= 20:
        return abstract, "paper_record.abstract"

    for section in sections:
        title = normalize_heading(section.get("section_title"))
        text = normalize_text(section.get("text"))
        if not text:
            continue
        if "ABSTRACT" in title:
            return text, "section.abstract"
        match = ABSTRACT_INLINE_RE.search(text)
        if match:
            candidate = normalize_text(match.group(1))
            if len(candidate.split()) >= 20:
                return candidate, "section.inline_abstract"

    intro_summary = normalize_text(paper.get("intro_summary"))
    if len(intro_summary.split()) >= 20:
        return intro_summary, "paper_record.intro_summary"

    text = normalize_text(paper.get("text"))
    return text[:2000], "paper_record.text_prefix"


def should_skip_section(section: dict[str, Any]) -> bool:
    heading = normalize_heading(section.get("section_title"))
    if heading in SKIP_SECTION_HEADINGS:
        return True
    if heading in STOP_SECTION_HEADINGS:
        return True
    if any(heading.startswith(stop) for stop in STOP_SECTION_HEADINGS):
        return True
    text = normalize_text(section.get("text"))
    return len(text) < 40


def build_full_text(title: str, abstract: str, sections: list[dict[str, Any]], fallback_text: str) -> str:
    parts = [f"Title: {title}"]
    if abstract:
        parts.append(f"Abstract: {abstract}")

    body_parts: list[str] = []
    for section in sorted(sections, key=lambda row: int(row.get("ordinal") or 0)):
        if should_skip_section(section):
            continue
        heading = normalize_text(section.get("section_title")) or "Section"
        text = normalize_text(section.get("text"))
        if text:
            body_parts.append(f"{heading}: {text}")

    if body_parts:
        parts.extend(body_parts)
    elif fallback_text:
        parts.append(normalize_text(fallback_text))
    return "\n\n".join(part for part in parts if part.strip())


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChemQA500 baseline input assets.")
    parser.add_argument(
        "--package-root",
        default="/workspace/ChemVerify/data/benchmarks/chemqa500_simple/package",
    )
    parser.add_argument(
        "--normalized-root",
        default="/workspace/ChemVerify/data/search_current/normalized",
    )
    parser.add_argument(
        "--output-root",
        default="/workspace/ChemVerify/data/benchmarks/chemqa500_simple/baseline_inputs",
    )
    parser.add_argument("--topic", default="all")
    parser.add_argument("--min-passage-chars", type=int, default=80)
    args = parser.parse_args()

    package_root = Path(args.package_root)
    normalized_root = Path(args.normalized_root)
    output_root = Path(args.output_root)

    corpus_rows = read_jsonl(package_root / "corpus_manifest.jsonl")
    query_rows_raw = read_jsonl(package_root / "queries.jsonl")
    paper_rows = read_jsonl(normalized_root / "papers.jsonl")
    section_rows = read_jsonl(normalized_root / "sections.jsonl")
    chunk_rows = read_jsonl(normalized_root / "chunks.jsonl")

    corpus_by_id = {str(row["paper_id"]): row for row in corpus_rows}
    papers_by_id = {str(row["paper_id"]): row for row in paper_rows}
    sections_by_paper: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in section_rows:
        sections_by_paper[str(row["paper_id"])].append(row)

    missing_papers = sorted(set(corpus_by_id) - set(papers_by_id), key=sort_paper_id)
    if missing_papers:
        raise SystemExit(f"normalized papers missing for ids: {missing_papers[:20]}")

    paper_text_rows: list[dict[str, Any]] = []
    abstract_rows: list[dict[str, Any]] = []
    specter_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    abstract_source_counts: dict[str, int] = defaultdict(int)

    for paper_id in sorted(corpus_by_id, key=sort_paper_id):
        corpus = corpus_by_id[paper_id]
        paper = papers_by_id[paper_id]
        title = normalize_text(corpus.get("title") or paper.get("title") or f"Paper {paper_id}")
        sections = sections_by_paper.get(paper_id, [])
        abstract, abstract_source = extract_abstract(paper, sections)
        abstract_source_counts[abstract_source] += 1
        full_text = build_full_text(title, abstract, sections, paper.get("text") or "")

        paper_text_rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "topic": args.topic,
                "text": full_text,
            }
        )
        abstract_rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "topic": args.topic,
                "abstract": abstract,
                "abstract_source": abstract_source,
            }
        )
        specter_rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "input_text": f"{title} [SEP] {abstract}".strip(),
                "topic": args.topic,
            }
        )
        manifest_rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "topic": args.topic,
                "is_target": str(parse_bool(corpus.get("is_target"))).lower(),
                "pdf_path": corpus.get("pdf_path", ""),
                "text_chars": len(full_text),
                "abstract_chars": len(abstract),
                "abstract_source": abstract_source,
            }
        )

    query_rows: list[dict[str, Any]] = []
    qrels_rows: list[dict[str, Any]] = []
    for row in query_rows_raw:
        query_id = str(row["query_id"])
        target_id = str(row["target_paper_id"])
        if target_id not in corpus_by_id:
            raise SystemExit(f"query {query_id} target_paper_id {target_id} not in corpus")
        query_rows.append(
            {
                "query_id": query_id,
                "query_text": normalize_text(row["query_text"]),
                "target_paper_id": target_id,
                "original_qa_id": query_id,
                "paper_title": manifest_rows[sorted(corpus_by_id, key=sort_paper_id).index(target_id)]["title"],
                "topic": args.topic,
            }
        )
        qrels_rows.append({"query_id": query_id, "paper_id": target_id, "relevance": 1})

    passage_rows: list[dict[str, Any]] = []
    colbert_rows: list[tuple[str, str]] = []
    passage_map_rows: list[dict[str, Any]] = []
    passage_index = 0
    for chunk in chunk_rows:
        paper_id = str(chunk.get("paper_id"))
        if paper_id not in corpus_by_id:
            continue
        text = normalize_text(chunk.get("text"))
        if len(text) < args.min_passage_chars:
            continue
        heading = normalize_text(chunk.get("heading"))
        if normalize_heading(heading) in STOP_SECTION_HEADINGS:
            continue
        passage_index += 1
        passage_id = str(passage_index)
        title = papers_by_id[paper_id].get("title") or corpus_by_id[paper_id].get("title") or f"Paper {paper_id}"
        passage_text = f"{normalize_text(title)}. {heading}. {text}".strip()
        passage_row = {
            "passage_id": passage_id,
            "paper_id": paper_id,
            "paper_title": normalize_text(title),
            "topic": args.topic,
            "section_path": " > ".join(chunk.get("section_path") or []),
            "section_id": chunk.get("section_id", ""),
            "chunk_id": chunk.get("chunk_id", ""),
            "page_start": chunk.get("page_start", ""),
            "page_end": chunk.get("page_end", ""),
            "passage_rank_in_paper": "",
            "passage_text": passage_text,
        }
        passage_rows.append(passage_row)
        colbert_rows.append((passage_id, passage_text))
        passage_map_rows.append(
            {
                "passage_id": passage_id,
                "paper_id": paper_id,
                "chunk_id": chunk.get("chunk_id", ""),
                "section_id": chunk.get("section_id", ""),
            }
        )

    rank_by_paper: dict[str, int] = defaultdict(int)
    for row in passage_rows:
        rank_by_paper[row["paper_id"]] += 1
        row["passage_rank_in_paper"] = str(rank_by_paper[row["paper_id"]])

    write_jsonl(output_root / "views" / "paper_text_view.jsonl", paper_text_rows)
    write_jsonl(output_root / "views" / "paper_abstracts.jsonl", abstract_rows)
    write_jsonl(output_root / "views" / "paper_passage_view.jsonl", passage_rows)

    write_tsv(output_root / "views" / "paper_text_view.tsv", ["paper_id", "title", "topic", "text"], paper_text_rows)
    write_tsv(
        output_root / "views" / "paper_abstracts.tsv",
        ["paper_id", "title", "topic", "abstract", "abstract_source"],
        abstract_rows,
    )
    write_tsv(
        output_root / "views" / "paper_passage_view.tsv",
        [
            "passage_id",
            "paper_id",
            "paper_title",
            "topic",
            "section_path",
            "section_id",
            "chunk_id",
            "page_start",
            "page_end",
            "passage_rank_in_paper",
            "passage_text",
        ],
        passage_rows,
    )

    write_jsonl(output_root / "method_inputs" / "anserini_papers.jsonl", [
        {
            "id": row["paper_id"],
            "paper_id": row["paper_id"],
            "title": row["title"],
            "text": row["text"],
            "contents": f"{row['title']} {row['text']}".strip(),
        }
        for row in paper_text_rows
    ])
    write_jsonl(output_root / "method_inputs" / "specter2_papers.jsonl", specter_rows)

    (output_root / "method_inputs").mkdir(parents=True, exist_ok=True)
    with (output_root / "method_inputs" / "colbert_collection.tsv").open("w", encoding="utf-8") as f:
        for passage_id, passage_text in colbert_rows:
            f.write(f"{passage_id}\t{passage_text}\n")
    write_tsv(
        output_root / "method_inputs" / "colbert_passage_to_paper.tsv",
        ["passage_id", "paper_id", "chunk_id", "section_id"],
        passage_map_rows,
    )

    write_jsonl(output_root / "queries" / "queries.jsonl", query_rows)
    write_tsv(
        output_root / "queries" / "queries.tsv",
        ["query_id", "query_text", "target_paper_id", "original_qa_id", "paper_title", "topic"],
        query_rows,
    )
    write_tsv(output_root / "queries" / "qrels.tsv", ["query_id", "paper_id", "relevance"], qrels_rows)
    with (output_root / "queries" / "qrels.trec").open("w", encoding="utf-8") as f:
        for row in qrels_rows:
            f.write(f"{row['query_id']} 0 {row['paper_id']} {row['relevance']}\n")

    write_tsv(
        output_root / "manifests" / "papers_manifest.csv",
        ["paper_id", "title", "topic", "is_target", "pdf_path", "text_chars", "abstract_chars", "abstract_source"],
        manifest_rows,
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "normalized_root": str(normalized_root),
        "output_root": str(output_root),
        "paper_count": len(paper_text_rows),
        "query_count": len(query_rows),
        "qrel_count": len(qrels_rows),
        "passage_count": len(passage_rows),
        "target_paper_count": sum(1 for row in manifest_rows if row["is_target"] == "true"),
        "distractor_paper_count": sum(1 for row in manifest_rows if row["is_target"] != "true"),
        "abstract_source_counts": dict(sorted(abstract_source_counts.items())),
        "min_passage_chars": args.min_passage_chars,
        "outputs": {
            "paper_text_view": "views/paper_text_view.jsonl",
            "paper_abstracts": "views/paper_abstracts.jsonl",
            "paper_passage_view": "views/paper_passage_view.jsonl",
            "queries": "queries/queries.jsonl",
            "qrels": "queries/qrels.tsv",
            "anserini_papers": "method_inputs/anserini_papers.jsonl",
            "specter2_papers": "method_inputs/specter2_papers.jsonl",
            "colbert_collection": "method_inputs/colbert_collection.tsv",
            "colbert_passage_to_paper": "method_inputs/colbert_passage_to_paper.tsv",
        },
    }
    (output_root / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# ChemQA500 Simple Baseline Inputs",
                "",
                "This directory contains derived input views for baseline methods on ChemQA500 Simple.",
                "It is generated from the clean benchmark package plus the parsed normalized search index.",
                "",
                "Key views:",
                "",
                "- `views/paper_text_view.jsonl`: paper-level full-text view for BM25/full-text baselines.",
                "- `views/paper_abstracts.jsonl`: title+abstract view for BM25(title+abstract), SPECTER2, and SemRank-style baselines.",
                "- `views/paper_passage_view.jsonl`: passage/chunk view for passage retrieval baselines.",
                "- `method_inputs/`: method-specific convenience inputs.",
                "- `queries/`: query and qrels files shared by all baselines.",
                "",
                "All paper IDs are the clean ChemQA500 IDs (`1`-`500`).",
                "The topic field is intentionally set to `all`; ChemQA500 Simple does not use topic labels.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
