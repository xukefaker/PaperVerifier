from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
import logging
import math
import time
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from .config import Settings
from .models import IngestSummary, PaperRecord
from .storage import LocalStore
from .utils import extract_keywords, normalize_title, normalize_whitespace

ACL_BASE_URL = "https://aclanthology.org"
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ListingEntry:
    paper_id: str
    title: str
    listing_title: str
    authors: list[str]
    abstract: str
    volume_id: str
    venue: str
    year: int
    track: str
    url: str
    pdf_url: str | None


class ACLAnthologyIngestor:
    def __init__(self, settings: Settings, store: LocalStore) -> None:
        self.settings = settings
        self.store = store

    def ingest_event(
        self,
        venue: str,
        year: int,
        tracks: list[str] | None = None,
        max_papers: int | None = None,
        download_pdfs: bool = True,
    ) -> IngestSummary:
        tracks = [track.lower() for track in (tracks or ["long"])]
        event_url = f"{ACL_BASE_URL}/events/{venue.lower()}-{year}/"
        logger.info(
            "[bold cyan]Manifest[/] | sync_start corpus=%s/%s tracks=%s",
            venue.lower(),
            year,
            ",".join(tracks),
        )
        try:
            with httpx.Client(timeout=self.settings.request_timeout, follow_redirects=True) as client:
                response = client.get(event_url)
                response.raise_for_status()
                listings = self._parse_event_page(response.text, venue=venue.lower(), year=year)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Could not reach ACL Anthology event page. url={event_url} error={exc}") from exc
        filtered = self._filter_listings(listings, tracks)
        if max_papers is not None:
            filtered = filtered[:max_papers]
        self._hydrate_listing_titles(filtered)
        logger.info(
            "[bold cyan]Manifest[/] | discovered corpus=%s/%s total=%s",
            venue.lower(),
            year,
            len(filtered),
        )

        normalized_existing = {paper.paper_id: paper for paper in self.store.load_papers()}
        existing = {paper.paper_id: paper for paper in self.store.load_raw_papers()}
        downloaded_pdfs = 0
        skipped_existing = 0
        if download_pdfs:
            downloaded_pdfs, skipped_existing, pdf_map = self._download_pdfs(filtered)
        else:
            pdf_map = {}

        for listing in filtered:
            current = existing.get(listing.paper_id) or normalized_existing.get(listing.paper_id)
            paper = PaperRecord(
                paper_id=listing.paper_id,
                anthology_id=listing.paper_id,
                title=listing.title,
                authors=listing.authors,
                venue=listing.venue,
                year=listing.year,
                track=listing.track,
                volume_id=listing.volume_id,
                abstract=listing.abstract,
                url=listing.url,
                pdf_url=listing.pdf_url,
                local_pdf_path=str(pdf_map.get(listing.paper_id)) if listing.paper_id in pdf_map else (current.local_pdf_path if current else None),
                keywords=extract_keywords(f"{listing.title} {listing.abstract}", limit=10),
                metadata={
                    "source_url": event_url,
                    "paper_url": listing.url,
                    "title_source": "paper_page_meta",
                    "listing_title": listing.listing_title,
                },
            )
            if current:
                paper.text = current.text
                paper.intro_summary = current.intro_summary
                paper.section_headings = current.section_headings
                paper.sections = current.sections
                paper.section_ids = current.section_ids
                paper.object_ids = current.object_ids
                paper.chunk_ids = current.chunk_ids
                paper.typed_evidence_summary = current.typed_evidence_summary
            existing[paper.paper_id] = paper

        saved = list(existing.values())
        saved.sort(key=lambda paper: paper.paper_id)
        self.store.save_raw_papers(saved)
        return IngestSummary(
            venue=venue.lower(),
            year=year,
            tracks=tracks,
            fetched_papers=len(filtered),
            saved_papers=len(saved),
            downloaded_pdfs=downloaded_pdfs,
            skipped_existing_pdfs=skipped_existing,
        )

    def _filter_listings(self, listings: list[_ListingEntry], tracks: list[str]) -> list[_ListingEntry]:
        if not tracks or "all" in tracks:
            return listings
        requested = {self._canonicalize_track(track) for track in tracks}
        include_main_alias = "main" in requested or {"long", "short"}.issubset(requested)
        filtered: list[_ListingEntry] = []
        for listing in listings:
            listing_track = self._canonicalize_track(listing.track)
            volume_suffix = self._canonicalize_track(listing.volume_id.split(".", 1)[-1])
            if listing_track in requested or volume_suffix in requested:
                filtered.append(listing)
                continue
            if include_main_alias and (listing_track == "main" or volume_suffix == "main"):
                filtered.append(listing)
                continue
            if "main" in requested and listing_track in {"long", "short"}:
                filtered.append(listing)
                continue
            if "main" in requested and volume_suffix in {"long", "short"}:
                filtered.append(listing)
        return filtered

    def _parse_event_page(self, html: str, venue: str, year: int) -> list[_ListingEntry]:
        soup = BeautifulSoup(html, "html.parser")
        listings: list[_ListingEntry] = []
        for volume_section in soup.select("div[id]"):
            section_id = volume_section.get("id", "")
            if section_id == "main-container":
                continue
            header_candidates = volume_section.select("h4 a[href^='/volumes/']")
            volume_link = None
            for candidate in header_candidates:
                href = candidate.get("href", "")
                if href.endswith(".bib"):
                    continue
                volume_link = candidate
                break
            if volume_link is None:
                continue
            volume_path = volume_link.get("href", "")
            volume_id = volume_path.rstrip("/").split("/")[-1]
            track = self._track_from_volume_id(volume_id)
            for paper_row in volume_section.select("div.d-sm-flex.align-items-stretch.mb-3"):
                strong_link = paper_row.select_one("strong a[href^='/']")
                if strong_link is None:
                    continue
                paper_path = strong_link.get("href", "")
                paper_id = paper_path.strip("/").split("/")[-1]
                if paper_id.endswith(".0"):
                    continue
                title = normalize_title(strong_link.get_text(" ", strip=True))
                authors = [normalize_whitespace(author.get_text(" ", strip=True)) for author in paper_row.select("span.d-block > a[href^='/people/']")]
                abstract_card = paper_row.find_next_sibling("div", class_=lambda value: value and "abstract-collapse" in value)
                abstract = ""
                if abstract_card is not None:
                    abstract = normalize_whitespace(abstract_card.get_text(" ", strip=True))
                pdf_badge = paper_row.select_one("a[href$='.pdf']")
                pdf_url = urljoin(ACL_BASE_URL, pdf_badge.get("href")) if pdf_badge is not None else None
                listings.append(
                    _ListingEntry(
                        paper_id=paper_id,
                        title=title,
                        listing_title=title,
                        authors=authors,
                        abstract=abstract,
                        volume_id=volume_id,
                        venue=venue,
                        year=year,
                        track=track,
                        url=urljoin(ACL_BASE_URL, paper_path),
                        pdf_url=pdf_url,
                    )
                )
        unique = {listing.paper_id: listing for listing in listings}
        return sorted(unique.values(), key=lambda listing: self._paper_sort_key(listing.paper_id))

    def _hydrate_listing_titles(self, listings: list[_ListingEntry]) -> None:
        if not listings:
            return

        logger.info(
            "[bold cyan]Manifest[/] | title_check_start total=%s source=paper_page_meta",
            len(listings),
        )
        started_at = time.time()
        completed = 0

        def worker(listing: _ListingEntry) -> tuple[str, str]:
            try:
                with httpx.Client(timeout=self.settings.request_timeout, follow_redirects=True) as client:
                    response = client.get(listing.url)
                    response.raise_for_status()
                return listing.paper_id, self._extract_paper_page_title(response.text, listing.url)
            except httpx.HTTPError as exc:
                logger.warning(
                    "[bold cyan]Manifest[/] | title_check_failed paper=%s url=%s error=%s",
                    listing.paper_id,
                    listing.url,
                    exc,
                )
                return listing.paper_id, listing.listing_title

        listing_by_id = {listing.paper_id: listing for listing in listings}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, listing) for listing in listings]
            for future in concurrent.futures.as_completed(futures):
                paper_id, resolved_title = future.result()
                listing = listing_by_id[paper_id]
                listing.title = resolved_title
                completed += 1
                if completed % 50 == 0 or completed == len(listings):
                    elapsed = max(time.time() - started_at, 1e-6)
                    rate = completed / elapsed
                    remaining = max(len(listings) - completed, 0)
                    eta_seconds = remaining / rate if rate > 0 else None
                    logger.info(
                        "[bold cyan]Manifest[/] | title_check_progress completed=%s total=%s rate_papers_per_min=%.2f eta=%s last=%s",
                        completed,
                        len(listings),
                        rate * 60,
                        _format_duration(eta_seconds),
                        paper_id,
                    )

    def _extract_paper_page_title(self, html: str, paper_url: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        meta_candidates = (
            soup.select_one('meta[name="citation_title"]'),
            soup.select_one('meta[property="og:title"]'),
            soup.select_one("title"),
        )
        for node in meta_candidates:
            if node is None:
                continue
            content = node.get("content") if node.name == "meta" else node.get_text(" ", strip=True)
            title = normalize_title(str(content or ""))
            if not title:
                continue
            if node.name == "title" and title.endswith(" - ACL Anthology"):
                title = normalize_title(title[: -len(" - ACL Anthology")])
            if title:
                return title
        raise RuntimeError(f"Could not extract a canonical title from paper page: {paper_url}")

    def _download_pdfs(self, listings: list[_ListingEntry]) -> tuple[int, int, dict[str, Path]]:
        pdf_map: dict[str, Path] = {}
        download_jobs = [listing for listing in listings if listing.pdf_url]
        downloaded = 0
        skipped = 0

        def worker(listing: _ListingEntry) -> tuple[str, Path, bool]:
            destination = self._pdf_destination(listing)
            if destination.exists():
                return listing.paper_id, destination, False
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with httpx.Client(timeout=self.settings.request_timeout, follow_redirects=True) as client:
                    response = client.get(listing.pdf_url)
                    response.raise_for_status()
                    destination.write_bytes(response.content)
            except httpx.HTTPError as exc:
                if destination.exists():
                    destination.unlink()
                raise RuntimeError(
                    "Could not download an ACL Anthology PDF. "
                    f"paper={listing.paper_id} url={listing.pdf_url} error={exc}"
                ) from exc
            return listing.paper_id, destination, True

        logger.info(
            "[bold green]PDF download[/] | start total=%s workers=%s output_dir=%s",
            len(download_jobs),
            4,
            str(self.settings.pdf_dir),
        )
        started_at = time.time()
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, listing) for listing in download_jobs]
            for future in concurrent.futures.as_completed(futures):
                paper_id, destination, changed = future.result()
                pdf_map[paper_id] = destination
                completed += 1
                if changed:
                    downloaded += 1
                else:
                    skipped += 1
                if completed % 25 == 0 or completed == len(download_jobs):
                    elapsed = max(time.time() - started_at, 1e-6)
                    rate = completed / elapsed
                    remaining = max(len(download_jobs) - completed, 0)
                    eta_seconds = remaining / rate if rate > 0 else None
                    logger.info(
                        "[bold green]PDF download[/] | progress completed=%s total=%s new=%s cached=%s rate_papers_per_min=%.2f eta=%s last=%s",
                        completed,
                        len(download_jobs),
                        downloaded,
                        skipped,
                        rate * 60,
                        _format_duration(eta_seconds),
                        paper_id,
                    )
        logger.info(
            "[bold green]PDF download[/] | done total=%s new=%s cached=%s",
            len(download_jobs),
            downloaded,
            skipped,
        )
        return downloaded, skipped, pdf_map

    def _pdf_destination(self, listing: _ListingEntry) -> Path:
        track = (listing.track or "unknown").lower()
        return self.settings.pdf_dir / listing.venue.lower() / str(listing.year) / track / f"{listing.paper_id}.pdf"

    def _track_from_volume_id(self, volume_id: str) -> str:
        parts = volume_id.split(".")
        if len(parts) == 2:
            suffix = parts[1]
            if suffix.startswith("findings-"):
                return "findings"
            if suffix.startswith(("acl-", "naacl-", "emnlp-")):
                return self._canonicalize_track(suffix.split("-", 1)[1])
            return self._canonicalize_track(suffix)
        if len(parts) >= 3:
            track = parts[-1]
            if parts[1] == "findings":
                return "findings"
            return self._canonicalize_track(track)
        return volume_id

    def _canonicalize_track(self, track: str | None) -> str:
        normalized = (track or "").strip().lower()
        aliases = {
            "demos": "demo",
            "main": "main",
            "long": "long",
            "short": "short",
        }
        return aliases.get(normalized, normalized)

    def _paper_sort_key(self, paper_id: str) -> tuple[str, str, int | str]:
        parts = paper_id.split(".")
        try:
            numeric_tail = int(parts[-1])
            return ".".join(parts[:-1]), "num", numeric_tail
        except ValueError:
            return ".".join(parts[:-1]), "str", parts[-1]


def _format_duration(seconds: float | None) -> str:
    if seconds is None or math.isinf(seconds):
        return "n/a"
    if seconds <= 0:
        return "0s"
    remaining = int(seconds)
    hours, remaining = divmod(remaining, 3600)
    minutes, secs = divmod(remaining, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
