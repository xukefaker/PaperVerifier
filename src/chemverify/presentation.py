from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

from .config import Settings
from .models import (
    EnrichedMetadata,
    ObjectRecord,
    PaperEnrichmentRecord,
    PaperReferenceEntry,
    PaperReferencesRecord,
    PaperRecord,
    StructuredAuthor,
    StructuredRationale,
    StructuredSummary,
)
from .utils import normalize_whitespace, now_iso

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_ARXIV_PATTERN = re.compile(r"\barXiv:(\d{4}\.\d{4,5}(?:v\d+)?)\b|\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")
_URL_PATTERN = re.compile(r"https?://\S+")
_REFERENCE_HEADING_PATTERN = re.compile(r"(?im)^(?:#{1,6}\s*)?(references|bibliography|works cited)\s*$")
_AUTHOR_MARKER_SYMBOLS = "*†‡§¶♠♣♡◊"
_DIGIT_AFFILIATION_START_PATTERN = re.compile(r"(?<![A-Za-z])(?P<marker>\d+)\s*(?=[A-Z])")
_NEXT_MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^#{1,6}\s+\S.*$")
_REFERENCE_CONTINUATION_PREFIXES = (
    "In ",
    "Proceedings",
    "Association for ",
    "Transactions of ",
    "Journal of ",
    "Workshop on ",
    "Conference on ",
    "Preprint",
    "arXiv",
    "pages ",
    "page ",
    "pp. ",
    "Online.",
    "Farrar, ",
    "MIT Press",
    "Springer",
    "IEEE",
    "ACM",
)
_AFFILIATION_KEYWORDS = (
    "department",
    "faculty",
    "university",
    "institute",
    "laboratory",
    "lab",
    "school",
    "college",
    "academy",
    "centre",
    "center",
    "research",
    "hospital",
    "corporation",
    "corp",
    "inc",
    "ltd",
    "llc",
    "openai",
    "deepmind",
    "anthropic",
    "microsoft",
    "google",
    "meta",
    "amazon",
    "nvidia",
    "tencent",
    "alibaba",
    "bytedance",
    "mbzuai",
    "hkust",
)
_SYMBOL_AFFILIATION_START_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9])(?P<marker>[{re.escape(_AUTHOR_MARKER_SYMBOLS)}])\s*(?=[A-Z])"
)
_AFFILIATION_PHRASE_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z&./'()-]+\s+){0,8}"
    r"(?:Department|Faculty|University|Institute|Laboratory|Lab|School|College|Academy|Centre|Center|Research|Hospital|Corporation|Corp\.?|Inc\.?|Ltd\.?|LLC|MBZUAI|HKUST)"
    r"(?:\s+[A-Z][A-Za-z&./'()-]+){0,8}\b"
)
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TOP_AUTHOR_REGION_MAX_Y = 220.0
_MAX_AUTHOR_OBJECTS = 8


def build_main_image_url(settings: Settings, paper_id: str, image_path: str | Path) -> str:
    paper_root = (settings.mineru_output_dir / paper_id).resolve()
    image_ref = _image_ref_from_path(paper_root, image_path)
    if image_ref is None:
        return ""
    base_url = settings.public_api_base_url.rstrip("/") if settings.public_api_base_url else None
    if base_url:
        return f"{base_url}/papers/{quote(paper_id, safe='')}/images/{quote(image_ref, safe='/')}"
    return f"/api/papers/{quote(paper_id, safe='')}/images/{quote(image_ref, safe='/')}"


def select_main_image_url(
    settings: Settings,
    paper_id: str,
    objects: list[ObjectRecord],
) -> str | None:
    selected = select_main_image_object(settings, paper_id, objects)
    if selected is None or not selected.image_path:
        return None
    image_url = build_main_image_url(settings, paper_id, selected.image_path)
    return image_url or None


def select_main_image_object(
    settings: Settings,
    paper_id: str,
    objects: list[ObjectRecord],
) -> ObjectRecord | None:
    paper_root = (settings.mineru_output_dir / paper_id).resolve()
    best_object: ObjectRecord | None = None
    best_score: float | None = None

    for obj in objects:
        if obj.object_type != "figure_block" or not obj.image_path:
            continue
        image_path = Path(obj.image_path).expanduser().resolve()
        if not image_path.exists() or not _is_relative_to(image_path, paper_root):
            continue
        score = _main_image_score(obj)
        if best_score is None or score > best_score:
            best_object = obj
            best_score = score
    return best_object


def resolve_paper_image_path(
    settings: Settings,
    paper_id: str,
    image_name: str,
    objects: list[ObjectRecord] | None = None,
) -> Path | None:
    requested_path = Path(image_name)
    if requested_path.is_absolute() or ".." in requested_path.parts:
        return None

    paper_root = (settings.mineru_output_dir / paper_id).resolve()
    if not paper_root.exists():
        return None

    direct_candidate = (paper_root / requested_path).resolve()
    if direct_candidate.is_file() and _is_relative_to(direct_candidate, paper_root):
        return direct_candidate

    for obj in objects or []:
        if not obj.image_path:
            continue
        image_path = Path(obj.image_path).expanduser().resolve()
        if not image_path.exists() or not _is_relative_to(image_path, paper_root):
            continue
        image_ref = _image_ref_from_path(paper_root, image_path)
        if image_ref == requested_path.as_posix() or image_path.name == requested_path.name:
            return image_path

    for candidate in paper_root.rglob(requested_path.name):
        resolved = candidate.resolve()
        if resolved.is_file() and _is_relative_to(resolved, paper_root):
            image_ref = _image_ref_from_path(paper_root, resolved)
            if image_ref and (image_ref == requested_path.as_posix() or resolved.name == requested_path.name):
                return resolved
    return None


def load_cached_paper_enrichment(
    settings: Settings,
    paper_id: str,
) -> tuple[StructuredSummary | None, EnrichedMetadata | None]:
    record = load_cached_paper_enrichment_record(settings, paper_id)
    if record is None:
        return None, None
    return record.structured_summary, record.enriched_metadata


def load_cached_paper_enrichment_record(
    settings: Settings,
    paper_id: str,
) -> PaperEnrichmentRecord | None:
    cache_path = _paper_enrichment_path(settings, paper_id)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return PaperEnrichmentRecord.model_validate(payload)
    except Exception:
        return None


def load_cached_paper_authorship(
    settings: Settings,
    paper: PaperRecord,
) -> tuple[list[str], list[str], list[StructuredAuthor]]:
    authors = [normalize_whitespace(author) for author in paper.authors if normalize_whitespace(author)]
    record = load_cached_paper_enrichment_record(settings, paper.paper_id)
    if record is None:
        return authors, [], [StructuredAuthor(name=author) for author in authors]
    structured = [StructuredAuthor.model_validate(item) for item in record.authors_structured]
    affiliations = _dedupe_preserve(record.affiliations)
    if not structured:
        structured = [StructuredAuthor(name=author) for author in authors]
    return authors, affiliations, structured


def save_cached_paper_enrichment_record(settings: Settings, record: PaperEnrichmentRecord) -> None:
    path = _paper_enrichment_path(settings, record.paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload.update(existing)
        except Exception:
            payload = {}
    payload.update(record.model_dump(mode="json", exclude_none=True))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cached_paper_references(
    settings: Settings,
    paper_id: str,
) -> PaperReferencesRecord | None:
    path = _paper_references_path(settings, paper_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PaperReferencesRecord.model_validate(payload)
    except Exception:
        return None


def save_cached_paper_references(settings: Settings, record: PaperReferencesRecord) -> None:
    path = _paper_references_path(settings, record.paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def extract_reference_entries_from_markdown(markdown_text: str) -> list[PaperReferenceEntry]:
    if not markdown_text.strip():
        return []
    match = _REFERENCE_HEADING_PATTERN.search(markdown_text)
    if match is None:
        return []
    section = markdown_text[match.end():].strip()
    if not section:
        return []
    next_heading = _NEXT_MARKDOWN_HEADING_PATTERN.search(section)
    if next_heading is not None:
        section = section[:next_heading.start()].rstrip()
    raw_blocks = _merge_reference_blocks(re.split(r"\n\s*\n+", section))
    entries: list[PaperReferenceEntry] = []
    for raw_block in raw_blocks:
        normalized = normalize_whitespace(raw_block)
        if len(normalized) < 20:
            continue
        entries.append(
            PaperReferenceEntry(
                ordinal=len(entries) + 1,
                raw_text=normalized,
                year=_extract_reference_year(normalized),
                doi=_extract_reference_doi(normalized),
                arxiv_id=_extract_reference_arxiv_id(normalized),
                url=_extract_reference_url(normalized),
            )
        )
    return entries


def extract_author_metadata(
    paper: PaperRecord,
    objects: list[ObjectRecord],
) -> tuple[list[str], list[str], list[StructuredAuthor]]:
    authors = [normalize_whitespace(author) for author in paper.authors if normalize_whitespace(author)]
    if not authors:
        return [], [], []

    candidate_objects = _select_author_objects(authors, objects)
    if not candidate_objects:
        return authors, [], [StructuredAuthor(name=author) for author in authors]

    direct_affiliations: dict[str, list[str]] = {}
    for obj in candidate_objects:
        matches = [author for author in authors if _author_name_in_text(author, obj.text)]
        if len(matches) != 1:
            continue
        affiliation = _extract_affiliation_from_single_author_object(matches[0], obj.text)
        if affiliation:
            direct_affiliations[matches[0]] = [affiliation]

    candidate_text = normalize_whitespace(" ".join(obj.text for obj in candidate_objects if normalize_whitespace(obj.text)))
    marker_inventory = _collect_markers_used_by_authors(authors, candidate_text)
    affiliation_by_marker = _extract_affiliation_map(candidate_text, allowed_markers=marker_inventory)
    fallback_affiliations = (
        _extract_fallback_affiliations(candidate_text, authors)
        if not direct_affiliations and not affiliation_by_marker
        else []
    )
    affiliations = _prune_redundant_affiliations(
        [affiliation for values in direct_affiliations.values() for affiliation in values]
        + list(affiliation_by_marker.values())
        + fallback_affiliations
    )

    structured: list[StructuredAuthor] = []
    for author in authors:
        author_affiliations = direct_affiliations.get(author, [])
        if not author_affiliations:
            author_affiliations = _affiliations_for_author(author, candidate_text, affiliation_by_marker)
        if not author_affiliations and len(affiliations) == 1:
            author_affiliations = affiliations[:]
        structured.append(
            StructuredAuthor(
                name=author,
                affiliation="; ".join(author_affiliations) if author_affiliations else None,
            )
        )
    return authors, affiliations, structured


def structure_rationale_text(text: str) -> StructuredRationale | None:
    normalized = normalize_whitespace(text)
    if not normalized:
        return None

    sentences = [sentence.strip(" -•") for sentence in _SENTENCE_SPLIT_PATTERN.split(normalized) if sentence.strip()]
    if not sentences:
        return StructuredRationale(main_reason=normalized, matching_points=[])

    main_reason = sentences[0]
    matching_points = _dedupe_preserve(
        [point for point in sentences[1:4] if point and point.lower() != main_reason.lower()]
    )
    return StructuredRationale(main_reason=main_reason, matching_points=matching_points)


def build_matched_sections_summary(section_rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in section_rows:
        section_title = normalize_whitespace(str(row.get("section_title", "") or ""))
        if not section_title:
            continue
        counts[section_title] = counts.get(section_title, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0].lower())))


def _main_image_score(obj: ObjectRecord) -> float:
    caption = (obj.caption or "").strip().lower()
    bbox = obj.bbox if len(obj.bbox) == 4 else [0.0, 0.0, 0.0, 0.0]
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    area = width * height

    score = 10.0
    if caption:
        score += 3.0
    score += max(0.0, 4.0 - float(obj.page_idx)) * 0.8
    score += min(area, 1_000_000.0) / 100_000.0

    keyword_hits = 0
    for token in ("figure 1", "overview", "architecture", "framework", "pipeline", "system overview"):
        if token in caption:
            keyword_hits += 1
    score += keyword_hits * 1.5
    return score


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _image_ref_from_path(paper_root: Path, image_path: str | Path) -> str | None:
    resolved = Path(image_path).expanduser().resolve()
    if not _is_relative_to(resolved, paper_root):
        return None
    return resolved.relative_to(paper_root).as_posix()


def _paper_enrichment_path(settings: Settings, paper_id: str) -> Path:
    return settings.data_dir / "enrichment" / "papers" / f"{paper_id}.json"


def _paper_references_path(settings: Settings, paper_id: str) -> Path:
    return settings.data_dir / "enrichment" / "references" / f"{paper_id}.json"


def _extract_reference_year(text: str) -> int | None:
    for match in re.finditer(r"\b(19|20)\d{2}[a-z]?\b", text):
        digits = re.match(r"\d{4}", match.group(0))
        if digits is not None:
            return int(digits.group(0))
    return None


def _extract_reference_doi(text: str) -> str | None:
    match = _DOI_PATTERN.search(text)
    return match.group(0).rstrip(".,;)") if match is not None else None


def _extract_reference_arxiv_id(text: str) -> str | None:
    match = _ARXIV_PATTERN.search(text)
    if match is None:
        return None
    return match.group(1) or match.group(2)


def _extract_reference_url(text: str) -> str | None:
    match = _URL_PATTERN.search(text)
    return match.group(0).rstrip(".,);") if match is not None else None


def _select_author_objects(authors: list[str], objects: list[ObjectRecord]) -> list[ObjectRecord]:
    candidates: list[tuple[float, ObjectRecord]] = []
    for obj in objects:
        if obj.object_type not in {"text_block", "list_block"}:
            continue
        if obj.page_idx != 1 or len(obj.bbox) != 4:
            continue
        if float(obj.bbox[1]) > _TOP_AUTHOR_REGION_MAX_Y:
            continue
        text = normalize_whitespace(obj.text)
        if not text or text.lower().startswith("abstract"):
            continue
        score = _author_object_score(text, authors)
        if score <= 0.0:
            continue
        candidates.append((score, obj))
    candidates.sort(key=lambda item: (float(item[1].bbox[1]), float(item[1].bbox[0]), -item[0]))
    return [item[1] for item in candidates[:_MAX_AUTHOR_OBJECTS]]


def _author_object_score(text: str, authors: list[str]) -> float:
    normalized = normalize_whitespace(text)
    lowered = normalized.lower()
    score = 0.0
    if any(_author_name_in_text(author, normalized) for author in authors):
        score += 3.0
    if _EMAIL_PATTERN.search(normalized):
        score += 1.5
    if any(keyword in lowered for keyword in _AFFILIATION_KEYWORDS):
        score += 1.0
    if len(normalized) > 450:
        score -= 2.0
    return score


def _author_name_in_text(author: str, text: str) -> bool:
    normalized_author = normalize_whitespace(author)
    normalized_text = normalize_whitespace(text)
    if not normalized_author or not normalized_text:
        return False
    return normalized_author.lower() in normalized_text.lower()


def _extract_affiliation_from_single_author_object(author: str, text: str) -> str | None:
    cleaned = _clean_affiliation_text(text)
    cleaned = re.sub(
        rf"\b{re.escape(author)}\b(?:\s*[\d*†‡§¶♠♣♡◊,]+)?",
        " ",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    cleaned = normalize_whitespace(cleaned.strip(" ,;:-"))
    if not cleaned:
        return None
    candidates = _extract_fallback_affiliations(cleaned, [])
    if candidates:
        return candidates[0]
    return cleaned if _looks_like_affiliation(cleaned) else None


def _extract_affiliation_map(text: str, allowed_markers: set[str] | None = None) -> dict[str, str]:
    cleaned = _clean_affiliation_text(text)
    mapping: dict[str, str] = {}
    for marker, affiliation in _extract_digit_marker_affiliations(cleaned, allowed_markers=allowed_markers):
        mapping[marker] = affiliation
    for marker, affiliation in _extract_marker_affiliations(cleaned, allowed_markers=allowed_markers):
        mapping[marker] = affiliation
    return mapping


def _get_front_matter_text(paper: PaperRecord, objects: list[ObjectRecord]) -> str:
    front_matter_blocks = [
        obj.text
        for obj in objects
        if obj.object_type in {"text_block", "list_block"}
        and obj.section_path
        and normalize_whitespace(obj.section_path[0]).lower() == "front matter"
        and normalize_whitespace(obj.text)
    ]
    if front_matter_blocks:
        return normalize_whitespace(" ".join(front_matter_blocks))

    text = paper.text or ""
    if not text:
        return ""
    upper_bound = len(text)
    for marker in ("\n\nAbstract", "\nAbstract", " Abstract "):
        position = text.find(marker)
        if position >= 0:
            upper_bound = min(upper_bound, position)
    return normalize_whitespace(text[:upper_bound])


def _extract_marker_affiliations(
    text: str,
    allowed_markers: set[str] | None = None,
) -> list[tuple[str, str]]:
    return _extract_affiliation_slices(text, _SYMBOL_AFFILIATION_START_PATTERN, allowed_markers=allowed_markers)


def _extract_digit_marker_affiliations(
    text: str,
    allowed_markers: set[str] | None = None,
) -> list[tuple[str, str]]:
    return _extract_affiliation_slices(text, _DIGIT_AFFILIATION_START_PATTERN, allowed_markers=allowed_markers)


def _extract_fallback_affiliations(text: str, authors: list[str]) -> list[str]:
    authorless_text = _clean_affiliation_text(text)
    for author in authors:
        if not author:
            continue
        authorless_text = re.sub(rf"\b{re.escape(author)}\b", " ", authorless_text)
    authorless_text = normalize_whitespace(authorless_text)

    candidates = [
        normalize_whitespace(match.group(0).strip(" ,;"))
        for match in _AFFILIATION_PHRASE_PATTERN.finditer(authorless_text)
    ]
    return _prune_redundant_affiliations([candidate for candidate in candidates if _looks_like_affiliation(candidate)])


def _looks_like_affiliation(text: str) -> bool:
    normalized = normalize_whitespace(text)
    if len(normalized) < 6 or len(normalized) > 180:
        return False
    lowered = normalized.lower()
    if lowered in {
        "department",
        "faculty",
        "university",
        "institute",
        "laboratory",
        "lab",
        "school",
        "college",
        "academy",
        "centre",
        "center",
        "research",
        "hospital",
        "corporation",
        "corp",
        "inc",
        "ltd",
        "llc",
    }:
        return False
    return any(keyword in lowered for keyword in _AFFILIATION_KEYWORDS)


def _affiliations_for_author(
    author: str,
    author_segment: str,
    affiliation_by_marker: dict[str, str],
) -> list[str]:
    if not affiliation_by_marker:
        return []
    positions: list[tuple[int, int]] = []
    lowered_segment = author_segment.lower()
    lowered_author = author.lower()
    start = lowered_segment.find(lowered_author)
    while start >= 0:
        positions.append((start, start + len(author)))
        start = lowered_segment.find(lowered_author, start + len(author))
    if not positions:
        return []

    all_author_occurrences: list[tuple[int, str]] = []
    for other_author in set(re.findall(r"[A-Z][A-Za-z.\-']+(?:\s+[A-Z][A-Za-z.\-']+)+", author_segment)):
        lowered_other = other_author.lower()
        if lowered_other == lowered_author:
            continue
        other_start = lowered_segment.find(lowered_other)
        while other_start >= 0:
            all_author_occurrences.append((other_start, other_author))
            other_start = lowered_segment.find(lowered_other, other_start + len(other_author))

    output: list[str] = []
    for window in _marker_windows_for_positions(positions, all_author_occurrences, author_segment):
        markers = _extract_author_markers(window)
        output.extend(affiliation_by_marker[marker] for marker in markers if marker in affiliation_by_marker)
    return _dedupe_preserve(output)


def _merge_reference_blocks(raw_blocks: list[str]) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for raw_block in raw_blocks:
        normalized = normalize_whitespace(raw_block)
        if not normalized:
            continue
        if not buffer:
            buffer = normalized
            continue
        if _should_merge_reference_blocks(buffer, normalized):
            buffer = normalize_whitespace(f"{buffer} {normalized}")
            continue
        if len(buffer) >= 20:
            merged.append(buffer)
        buffer = normalized
    if len(buffer) >= 20:
        merged.append(buffer)
    return merged


def _should_merge_reference_blocks(current: str, nxt: str) -> bool:
    if _extract_reference_year(current) is None:
        return True
    stripped = current.rstrip()
    if not stripped:
        return False
    if stripped.endswith((",", ";", ":", "-", "—")):
        return True
    if _is_reference_continuation_block(nxt):
        return True
    return False


def _extract_affiliation_slices(
    text: str,
    pattern: re.Pattern[str],
    allowed_markers: set[str] | None = None,
) -> list[tuple[str, str]]:
    matches = list(pattern.finditer(text))
    if allowed_markers is not None:
        matches = [match for match in matches if match.group("marker") in allowed_markers]
    output: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        marker = match.group("marker")
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        affiliation = _clean_affiliation_text(text[match.end():next_start].strip(" ,;"))
        if _looks_like_affiliation(affiliation):
            output.append((marker, affiliation))
    return output


def _extract_author_markers(window: str) -> list[str]:
    markers: list[str] = []
    index = 0
    while index < len(window):
        char = window[index]
        if char.isdigit():
            end = index + 1
            while end < len(window) and window[end].isdigit():
                end += 1
            markers.append(window[index:end])
            index = end
            continue
        if char in _AUTHOR_MARKER_SYMBOLS:
            markers.append(char)
            index += 1
            continue
        if char.isalpha():
            break
        index += 1
    return markers


def _clean_affiliation_text(text: str) -> str:
    cleaned = re.sub(r"\{[^}]*\}", " ", text)
    cleaned = _EMAIL_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"(?:^|\s)@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", cleaned)
    return normalize_whitespace(cleaned)


def _prune_redundant_affiliations(items: list[str]) -> list[str]:
    deduped = _dedupe_preserve(items)
    output: list[str] = []
    for item in deduped:
        lowered = item.lower()
        if any(
            lowered != other.lower() and lowered in other.lower()
            for other in deduped
        ):
            continue
        output.append(item)
    return output


def _collect_markers_used_by_authors(authors: list[str], author_segment: str) -> set[str]:
    marker_inventory: set[str] = set()
    lowered_segment = author_segment.lower()
    all_author_occurrences: list[tuple[int, str]] = []
    for other_author in authors:
        lowered_author = other_author.lower()
        start = lowered_segment.find(lowered_author)
        while start >= 0:
            all_author_occurrences.append((start, other_author))
            start = lowered_segment.find(lowered_author, start + len(other_author))

    for author in authors:
        positions: list[tuple[int, int]] = []
        lowered_author = author.lower()
        start = lowered_segment.find(lowered_author)
        while start >= 0:
            positions.append((start, start + len(author)))
            start = lowered_segment.find(lowered_author, start + len(author))
        for window in _marker_windows_for_positions(positions, all_author_occurrences, author_segment):
            marker_inventory.update(_extract_author_markers(window))
    return marker_inventory


def _marker_windows_for_positions(
    positions: list[tuple[int, int]],
    all_author_occurrences: list[tuple[int, str]],
    author_segment: str,
) -> list[str]:
    windows: list[str] = []
    for start_pos, end_pos in positions:
        next_bound = min(
            [other_start for other_start, _ in all_author_occurrences if other_start > start_pos] or [len(author_segment)]
        )
        window = author_segment[end_pos: min(next_bound, end_pos + 48)]
        window = re.split(
            rf"\s+(?=(?:\d+\s*[A-Z]|[{re.escape(_AUTHOR_MARKER_SYMBOLS)}]\s*[A-Z]))",
            window,
            maxsplit=1,
        )[0]
        windows.append(window)
    return windows


def _is_reference_continuation_block(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if any(stripped.startswith(prefix) for prefix in _REFERENCE_CONTINUATION_PREFIXES):
        return True
    lowered = stripped.lower()
    if _extract_reference_year(stripped) is None and len(stripped) <= 80:
        return True
    return lowered.startswith(("in ", "pages ", "page ", "pp. ", "online."))


def _dedupe_preserve(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = normalize_whitespace(item)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output
