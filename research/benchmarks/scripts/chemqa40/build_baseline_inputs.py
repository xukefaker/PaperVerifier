#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PRE_BODY_SKIP_HEADINGS = {
    "READ ONLINE",
    "ACCESS",
    "ARTICLE RECOMMENDATIONS",
    "METRICS MORE",
    "METRICS  MORE",
    "CHECK FOR UPDATES",
    "A R T I C L E I N F O",
    "ARTICLE INFO",
}

STOP_HEADINGS = {
    "ASSOCIATED CONTENT",
    "AUTHOR INFORMATION",
    "NOTES",
    "ACKNOWLEDGMENTS",
    "ACKNOWLEDGEMENTS",
    "REFERENCES",
    "SUPPORTING INFORMATION",
}

CAPTION_RE = re.compile(r"^(Figure|FIGURE|Table|TABLE|Scheme|SCHEME)\s+([S]?\d+|\d+)\b")
IMAGE_RE = re.compile(r"^!\[\]\(")
ABSTRACT_RE = re.compile(r"ABSTRACT:\s*(.+?)(?=\n#\s+|\Z)", re.S | re.I)
SUPPORTING_INLINE_RE = re.compile(r"^\*?\s*s[ıi]\s+Supporting Information\s*", re.I)
WHITESPACE_RE = re.compile(r"\s+")
HYPHENATION_RE = re.compile(r"([A-Za-z])-\s+([A-Za-z])")
HEADING_RE = re.compile(r"^#+\s*")
METADATA_LINE_RE = re.compile(
    r"^(Received:|Accepted:|Published online:|Published:|Available online:|DOI:|https?://|www\.)",
    re.I,
)


@dataclass
class PaperAsset:
    paper_id: str
    title: str
    topic: str
    pdf_relpath: str
    mineru_relpath: str
    original_query_count: int
    markdown_path: Path


def normalize_heading(raw: str) -> str:
    text = re.sub(r"^#+\s*", "", raw).strip()
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = WHITESPACE_RE.sub(" ", text).strip().upper()
    return text


def heading_compact(raw: str) -> str:
    return normalize_heading(raw).replace(" ", "")


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = HYPHENATION_RE.sub(r"\1\2", text)
    text = text.replace("\n", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def clean_body_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if IMAGE_RE.match(line):
        return ""
    if CAPTION_RE.match(line):
        return ""
    if line.startswith("Cite This:"):
        return ""
    if line.startswith("Complete contact information is available at:"):
        return ""
    return normalize_text(line)


def looks_like_metadata_paragraph(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned:
        return True
    if METADATA_LINE_RE.match(cleaned):
        return True
    if cleaned.startswith("©") or cleaned.startswith("Open Access"):
        return True
    # Author lines in MinerU often contain digits for affiliations and many commas but no sentence punctuation.
    if len(cleaned.split()) < 40 and any(ch.isdigit() for ch in cleaned) and "." not in cleaned:
        return True
    if "&" in cleaned and "." not in cleaned and len(cleaned.split()) < 50:
        return True
    if cleaned.count(",") >= 5 and any(ch.isdigit() for ch in cleaned) and "." not in cleaned:
        return True
    return False


def extract_abstract_from_leading_paragraphs(md_text: str) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    title_seen = False

    def flush() -> None:
        nonlocal current
        if current:
            paragraphs.append("\n".join(current).strip())
            current = []

    for raw in md_text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("#"):
            heading = re.sub(r"^#+\s*", "", stripped).strip()
            norm = normalize_heading(heading)
            if not title_seen:
                title_seen = True
                continue
            if norm in PRE_BODY_SKIP_HEADINGS:
                flush()
                continue
            flush()
            break
        if not title_seen:
            continue
        current.append(stripped)
    flush()

    for paragraph in paragraphs:
        cleaned_lines = []
        for raw in paragraph.splitlines():
            raw = raw.strip()
            if not raw or HEADING_RE.match(raw):
                continue
            raw = SUPPORTING_INLINE_RE.sub("", raw)
            raw = clean_body_line(raw)
            if raw:
                cleaned_lines.append(raw)
        candidate = normalize_text(" ".join(cleaned_lines))
        if not candidate:
            continue
        if looks_like_metadata_paragraph(candidate):
            continue
        if len(candidate.split()) < 40:
            continue
        return candidate
    return ""


def extract_abstract(md_text: str) -> str:
    match = ABSTRACT_RE.search(md_text)
    if match:
        text = match.group(1)
        lines = []
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            if IMAGE_RE.match(raw):
                continue
            if raw.startswith("#"):
                break
            raw = SUPPORTING_INLINE_RE.sub("", raw)
            raw = clean_body_line(raw)
            if raw:
                lines.append(raw)
        abstract = normalize_text(" ".join(lines))
        if abstract:
            return abstract

    lines = md_text.splitlines()
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped.startswith("#"):
            continue
        if heading_compact(stripped) != "ABSTRACT":
            continue
        block_lines = []
        in_keywords_block = False
        for follow in lines[idx + 1 :]:
            candidate = follow.strip()
            if not candidate:
                in_keywords_block = False
                if block_lines:
                    break
                continue
            if candidate.startswith("#"):
                break
            candidate_norm = normalize_text(candidate).lower()
            if candidate_norm.startswith("keywords"):
                in_keywords_block = True
                continue
            if in_keywords_block:
                continue
            cleaned = clean_body_line(SUPPORTING_INLINE_RE.sub("", candidate))
            if cleaned:
                block_lines.append(cleaned)
        abstract = normalize_text(" ".join(block_lines))
        if abstract:
            return abstract
    return extract_abstract_from_leading_paragraphs(md_text)


def parse_markdown_sections(md_text: str) -> list[tuple[str, str]]:
    lines = md_text.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    started = False

    def flush() -> None:
        nonlocal current_heading, current_lines
        if not current_heading:
            current_lines = []
            return
        text = normalize_text(" ".join(current_lines))
        if text:
            sections.append((current_heading, text))
        current_lines = []

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("#"):
            heading = re.sub(r"^#+\s*", "", line).strip()
            norm = normalize_heading(heading)
            if norm in PRE_BODY_SKIP_HEADINGS:
                continue
            if norm in STOP_HEADINGS:
                if started:
                    flush()
                    break
                continue
            if not started:
                # Start body at the first substantive heading.
                started = True
            else:
                flush()
            current_heading = heading
            continue

        if not started or not current_heading:
            continue
        cleaned = clean_body_line(line)
        if cleaned:
            current_lines.append(cleaned)

    if started:
        flush()

    return sections


def build_paper_text_view(title: str, abstract: str, sections: list[tuple[str, str]]) -> str:
    blocks: list[str] = [f"Title: {title}"]
    if abstract:
        blocks.append(f"Abstract: {abstract}")
    for heading, text in sections:
        blocks.append(f"{heading}: {text}")
    return "\n\n".join(blocks).strip()


def iter_word_windows(text: str, window_words: int = 120, overlap_words: int = 30) -> Iterable[str]:
    words = text.split()
    if not words:
        return
    if len(words) <= window_words:
        yield " ".join(words)
        return
    step = max(1, window_words - overlap_words)
    idx = 0
    while idx < len(words):
        chunk = words[idx : idx + window_words]
        if not chunk:
            break
        yield " ".join(chunk)
        if idx + window_words >= len(words):
            break
        idx += step


def load_benchmark_assets(benchmark_jsonl: Path, parsed_root: Path) -> tuple[list[PaperAsset], list[dict]]:
    rows = [json.loads(line) for line in benchmark_jsonl.read_text().splitlines() if line.strip()]
    unique: dict[str, PaperAsset] = {}
    counts: dict[str, int] = {}
    for row in rows:
        paper_id = row["target_paper_id"]
        counts[paper_id] = counts.get(paper_id, 0) + 1
        if paper_id in unique:
            continue
        unique[paper_id] = PaperAsset(
            paper_id=paper_id,
            title=row["paper_title"],
            topic=row["topic"],
            pdf_relpath=row["pdf_relpath"],
            mineru_relpath=row["mineru_relpath"],
            original_query_count=0,
            markdown_path=parsed_root / paper_id / "txt" / f"{paper_id}.md",
        )
    for paper_id, asset in unique.items():
        asset.original_query_count = counts.get(paper_id, 0)
    return sorted(unique.values(), key=lambda x: x.paper_id), rows


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "root": base,
        "views": base / "views",
        "queries": base / "queries",
        "manifests": base / "manifests",
        "method_inputs": base / "method_inputs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_tsv(path: Path, rows: Iterable[dict], fieldnames: list[str], include_header: bool = True) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if include_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            count += 1
    return count


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            count += 1
    return count


def build_assets(benchmark_jsonl: Path, parsed_root: Path, out_root: Path) -> dict:
    papers, benchmark_rows = load_benchmark_assets(benchmark_jsonl, parsed_root)
    paths = ensure_dirs(out_root)

    paper_text_rows = []
    abstract_rows = []
    passage_rows = []
    anserini_rows = []
    manifest_rows = []

    abstract_success = 0
    abstract_failures: list[str] = []
    total_passages = 0

    for asset in papers:
        md_text = asset.markdown_path.read_text(encoding="utf-8")
        abstract = extract_abstract(md_text)
        if abstract:
            abstract_success += 1
        else:
            abstract_failures.append(asset.paper_id)

        sections = parse_markdown_sections(md_text)
        paper_text = build_paper_text_view(asset.title, abstract, sections)

        paper_text_rows.append(
            {
                "paper_id": asset.paper_id,
                "title": asset.title,
                "topic": asset.topic,
                "text": paper_text,
            }
        )
        abstract_rows.append(
            {
                "paper_id": asset.paper_id,
                "title": asset.title,
                "topic": asset.topic,
                "abstract": abstract,
            }
        )
        manifest_rows.append(
            {
                "paper_id": asset.paper_id,
                "title": asset.title,
                "topic": asset.topic,
                "pdf_relpath": asset.pdf_relpath,
                "mineru_relpath": asset.mineru_relpath,
                "query_count": asset.original_query_count,
            }
        )
        anserini_rows.append(
            {
                "id": asset.paper_id,
                "title": asset.title,
                "text": paper_text,
                "contents": f"{asset.title}. {paper_text}",
            }
        )

        passage_rank = 0
        for heading, section_text in sections:
            for window in iter_word_windows(section_text):
                passage_rank += 1
                total_passages += 1
                passage_id = f"{asset.paper_id}::p{passage_rank:04d}"
                passage_text = normalize_text(f"{asset.title}. {heading}. {window}")
                passage_rows.append(
                    {
                        "passage_id": passage_id,
                        "paper_id": asset.paper_id,
                        "paper_title": asset.title,
                        "topic": asset.topic,
                        "section_path": heading,
                        "passage_rank_in_paper": passage_rank,
                        "passage_text": passage_text,
                    }
                )

    # Views
    write_jsonl(paths["views"] / "paper_text_view.jsonl", paper_text_rows)
    write_jsonl(paths["views"] / "paper_abstracts.jsonl", abstract_rows)
    write_jsonl(paths["views"] / "paper_passage_view.jsonl", passage_rows)
    write_tsv(paths["views"] / "paper_text_view.tsv", paper_text_rows, ["paper_id", "title", "topic", "text"])
    write_tsv(paths["views"] / "paper_abstracts.tsv", abstract_rows, ["paper_id", "title", "topic", "abstract"])
    write_tsv(
        paths["views"] / "paper_passage_view.tsv",
        passage_rows,
        ["passage_id", "paper_id", "paper_title", "topic", "section_path", "passage_rank_in_paper", "passage_text"],
    )

    # Queries / qrels
    query_rows = []
    qrel_rows = []
    qrel_trec_rows = []
    for row in benchmark_rows:
        query_rows.append(
            {
                "query_id": row["query_id"],
                "query_text": row["query_text"],
                "target_paper_id": row["target_paper_id"],
                "original_qa_id": row["original_qa_id"],
                "paper_title": row["paper_title"],
                "topic": row["topic"],
            }
        )
        qrel_rows.append({"query_id": row["query_id"], "target_paper_id": row["target_paper_id"]})
        qrel_trec_rows.append({"query_id": row["query_id"], "iter": 0, "doc_id": row["target_paper_id"], "rel": 1})

    write_jsonl(paths["queries"] / "queries.jsonl", query_rows)
    write_tsv(paths["queries"] / "queries.tsv", query_rows, ["query_id", "query_text"], include_header=False)
    write_tsv(paths["queries"] / "qrels.tsv", qrel_rows, ["query_id", "target_paper_id"])
    write_tsv(paths["queries"] / "qrels.trec", qrel_trec_rows, ["query_id", "iter", "doc_id", "rel"], include_header=False)

    # Manifests
    write_csv(
        paths["manifests"] / "papers_manifest.csv",
        manifest_rows,
        ["paper_id", "title", "topic", "pdf_relpath", "mineru_relpath", "query_count"],
    )

    # Method-specific convenience exports
    write_jsonl(paths["method_inputs"] / "anserini_papers.jsonl", anserini_rows)
    write_tsv(paths["method_inputs"] / "colbert_collection.tsv", passage_rows, ["passage_id", "passage_text"], include_header=False)
    write_tsv(paths["method_inputs"] / "colbert_passage_to_paper.tsv", passage_rows, ["passage_id", "paper_id"], include_header=True)
    write_jsonl(
        paths["method_inputs"] / "specter2_papers.jsonl",
        [
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "input_text": f"{row['title']} [SEP] {row['abstract']}".strip(),
            }
            for row in abstract_rows
        ],
    )

    summary = {
        "paper_count": len(papers),
        "query_count": len(query_rows),
        "qrel_count": len(qrel_rows),
        "passage_count": total_passages,
        "abstract_success_count": abstract_success,
        "abstract_failure_count": len(abstract_failures),
        "abstract_failure_papers": abstract_failures,
        "output_root": str(out_root),
    }

    (out_root / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# ChemQA40 Baseline Inputs",
                "",
                "This directory contains frozen input assets for running search baselines on ChemQA40.",
                "",
                "## Views",
                "",
                "- `views/paper_text_view.jsonl|tsv`: cleaned main-paper full text view",
                "- `views/paper_abstracts.jsonl|tsv`: extracted title + abstract view",
                "- `views/paper_passage_view.jsonl|tsv`: passage view for passage-retrieval baselines",
                "",
                "## Queries",
                "",
                "- `queries/queries.jsonl|tsv`",
                "- `queries/qrels.tsv`",
                "- `queries/qrels.trec`",
                "",
                "## Manifests",
                "",
                "- `manifests/papers_manifest.csv`",
                "",
                "## Method-specific convenience exports",
                "",
                "- `method_inputs/anserini_papers.jsonl`",
                "- `method_inputs/colbert_collection.tsv`",
                "- `method_inputs/colbert_passage_to_paper.tsv`",
                "- `method_inputs/specter2_papers.jsonl`",
                "",
                "## Build summary",
                "",
                f"- paper_count: {len(papers)}",
                f"- query_count: {len(query_rows)}",
                f"- passage_count: {total_passages}",
                f"- abstract_success_count: {abstract_success}",
                f"- abstract_failure_count: {len(abstract_failures)}",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-jsonl",
        default="/workspace/ChemVerify/data/benchmarks/chemqa40/ablation/input/chemqa40_search_benchmark_final.jsonl",
    )
    parser.add_argument(
        "--parsed-root",
        default="/workspace/ChemVerify/data/parsed/mineru",
    )
    parser.add_argument(
        "--out-root",
        default="/workspace/ChemVerify/data/benchmarks/chemqa40/baseline_inputs",
    )
    args = parser.parse_args()

    summary = build_assets(Path(args.benchmark_jsonl), Path(args.parsed_root), Path(args.out_root))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
