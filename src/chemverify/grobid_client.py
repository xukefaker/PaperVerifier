from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import httpx

from .config import Settings
from .models import StructuredAuthor
from .utils import normalize_whitespace

_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
_XML_ID_KEY = "{http://www.w3.org/XML/1998/namespace}id"


@dataclass(slots=True)
class GrobidHeaderResult:
    title: str | None
    affiliations: list[str]
    authors_structured: list[StructuredAuthor]


class GrobidHeaderClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract_header(self, pdf_path: Path) -> GrobidHeaderResult:
        if not self.settings.grobid_enabled:
            raise RuntimeError("GROBID is disabled. Enable [grobid].enabled in config.toml first.")
        if not pdf_path.exists():
            raise RuntimeError(f"GROBID input PDF does not exist: {pdf_path}")

        with pdf_path.open("rb") as handle:
            response = httpx.post(
                f"{self.settings.grobid_base_url}/api/processHeaderDocument",
                headers={"Accept": "application/xml"},
                files={"input": (pdf_path.name, handle, "application/pdf")},
                timeout=self.settings.grobid_timeout_seconds,
            )
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").lower()
        if "xml" not in content_type:
            preview = response.text[:400].strip()
            raise RuntimeError(
                "GROBID header API did not return XML."
                f" content_type={response.headers.get('content-type')!r}"
                f" body_preview={preview!r}"
            )
        return _parse_header_tei(response.text)


def _parse_header_tei(xml_text: str) -> GrobidHeaderResult:
    root = ET.fromstring(xml_text)
    title = _first_non_empty(
        [
            _node_text(root.find(".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title[@type='main']", _TEI_NS)),
            _node_text(root.find(".//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title", _TEI_NS)),
        ]
    )

    global_affiliations: dict[str, str] = {}
    for affiliation_node in root.findall(".//tei:teiHeader//tei:affiliation", _TEI_NS):
        affiliation_text = _extract_affiliation_text(affiliation_node)
        if not affiliation_text:
            continue
        affiliation_id = affiliation_node.get(_XML_ID_KEY) or affiliation_node.get("key") or ""
        if affiliation_id:
            global_affiliations[affiliation_id] = affiliation_text

    authors_structured: list[StructuredAuthor] = []
    affiliation_pool: list[str] = []
    for author_node in root.findall(".//tei:teiHeader//tei:author", _TEI_NS):
        name = _extract_author_name(author_node)
        if not name:
            continue
        author_affiliations = _collect_author_affiliations(author_node, global_affiliations)
        affiliation_pool.extend(author_affiliations)
        authors_structured.append(
            StructuredAuthor(
                name=name,
                affiliation="; ".join(author_affiliations) if author_affiliations else None,
            )
        )

    deduped_affiliations = _dedupe_preserve([*global_affiliations.values(), *affiliation_pool])
    if not authors_structured:
        raise RuntimeError("GROBID did not return usable author metadata.")

    return GrobidHeaderResult(
        title=title,
        affiliations=deduped_affiliations,
        authors_structured=authors_structured,
    )


def _extract_author_name(author_node: ET.Element) -> str:
    pers_name = author_node.find(".//tei:persName", _TEI_NS)
    if pers_name is None:
        return normalize_whitespace(" ".join(author_node.itertext()))

    parts: list[str] = []
    for tag in ("forename", "middlename", "surname", "genName"):
        for node in pers_name.findall(f"tei:{tag}", _TEI_NS):
            text = _node_text(node)
            if text:
                parts.append(text)
    if not parts:
        return _node_text(pers_name)
    return normalize_whitespace(" ".join(parts))


def _collect_author_affiliations(author_node: ET.Element, global_affiliations: dict[str, str]) -> list[str]:
    affiliations = [
        _extract_affiliation_text(node)
        for node in author_node.findall(".//tei:affiliation", _TEI_NS)
    ]
    references: list[str] = []
    for ref_node in author_node.findall(".//tei:ref", _TEI_NS):
        ref_type = normalize_whitespace(ref_node.get("type") or "").lower()
        target = normalize_whitespace(ref_node.get("target") or "").lstrip("#")
        if ref_type == "affiliation" and target and target in global_affiliations:
            references.append(global_affiliations[target])
    return _dedupe_preserve([item for item in [*affiliations, *references] if item])


def _extract_affiliation_text(affiliation_node: ET.Element) -> str:
    candidates: list[str] = []
    for xpath in (
        ".//tei:orgName",
        ".//tei:department",
        ".//tei:institution",
        ".//tei:settlement",
        ".//tei:region",
        ".//tei:country",
        ".//tei:addrLine",
    ):
        for node in affiliation_node.findall(xpath, _TEI_NS):
            text = _node_text(node)
            if text:
                candidates.append(text)
    if not candidates:
        candidates.append(_node_text(affiliation_node))
    return normalize_whitespace(", ".join(_dedupe_preserve([item for item in candidates if item])))


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return normalize_whitespace(" ".join(node.itertext()))


def _first_non_empty(values: Iterable[str]) -> str | None:
    for value in values:
        normalized = normalize_whitespace(value)
        if normalized:
            return normalized
    return None


def _dedupe_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = normalize_whitespace(value)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output
