from __future__ import annotations

import html
import re
from urllib.parse import quote

from .config import Settings
from .models import PaperRecord, StructuredAuthor, ZoteroExportPayload
from .presentation import load_cached_paper_authorship

_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title} | Zotero Save</title>
    <meta name="citation_title" content="{title}" />
{author_meta}
    <meta name="citation_publication_date" content="{year}" />
    <meta name="citation_conference_title" content="{venue}" />
    <meta name="citation_abstract" content="{abstract}" />
    <meta name="citation_language" content="en" />
    <meta name="citation_journal_title" content="{venue}" />
{doi_meta}
{pdf_meta}
{public_url_meta}
    <meta name="DC.Title" content="{title}" />
{dc_creator_meta}
    <meta name="DC.Type" content="Text" />
    <meta name="DC.Identifier" content="{paper_id}" />
{dc_source_meta}
{dc_date_meta}
    <style>
      :root {{
        color-scheme: light;
        --bg: #f5f7fb;
        --card: rgba(255, 255, 255, 0.94);
        --line: rgba(148, 163, 184, 0.22);
        --ink: #0f172a;
        --muted: #475569;
        --accent: #4f46e5;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        min-height: 100vh;
        background:
          radial-gradient(circle at top, rgba(99, 102, 241, 0.12), transparent 34%),
          linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%);
        color: var(--ink);
        font-family: "Inter", "Segoe UI", sans-serif;
      }}
      main {{
        max-width: 860px;
        margin: 0 auto;
        padding: 56px 24px 96px;
      }}
      .card {{
        border: 1px solid var(--line);
        border-radius: 28px;
        background: var(--card);
        box-shadow: 0 28px 70px rgba(15, 23, 42, 0.08);
        padding: 32px;
        backdrop-filter: blur(18px);
      }}
      .eyebrow {{
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.24em;
        color: var(--accent);
      }}
      h1 {{
        margin: 14px 0 0;
        font-size: clamp(2rem, 3.8vw, 3rem);
        line-height: 1.08;
        letter-spacing: -0.04em;
      }}
      .meta {{
        margin-top: 22px;
        display: grid;
        gap: 10px;
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.75;
      }}
      .actions {{
        margin-top: 30px;
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }}
      .actions a {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 46px;
        padding: 0 18px;
        border-radius: 999px;
        border: 1px solid rgba(79, 70, 229, 0.16);
        text-decoration: none;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
      }}
      .primary {{
        background: var(--accent);
        color: white;
      }}
      .secondary {{
        background: white;
        color: var(--accent);
      }}
      .section {{
        margin-top: 30px;
        border-top: 1px solid var(--line);
        padding-top: 24px;
      }}
      .section h2 {{
        margin: 0 0 12px;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--muted);
      }}
      .abstract {{
        font-size: 1rem;
        line-height: 1.9;
        color: var(--muted);
        white-space: pre-wrap;
      }}
      .notes {{
        margin-top: 24px;
        font-size: 0.9rem;
        line-height: 1.8;
        color: var(--muted);
      }}
      .mono {{
        font-family: "SFMono-Regular", "Consolas", monospace;
      }}
    </style>
  </head>
  <body>
    <main>
      <article class="card">
        <div class="eyebrow">Zotero Save</div>
        <h1>{title}</h1>
        <div class="meta">
          <div>{authors_line}</div>
          <div>{venue_line}</div>
          {affiliations_line}
          {doi_line}
          {source_line}
        </div>
        <div class="actions">
          <a class="primary" href="{canonical_url}" target="_blank" rel="noreferrer">Open Source Page</a>
          {pdf_link}
          <a class="secondary" href="{bibtex_url}">Download BibTeX</a>
          <a class="secondary" href="{ris_url}">Download RIS</a>
        </div>
        <section class="section">
          <h2>Abstract</h2>
          <div class="abstract">{abstract}</div>
        </section>
        <p class="notes">
          This page is intentionally structured for Zotero Connector and standard citation export.
          If the connector is installed in your browser, save from this page; otherwise download
          BibTeX or RIS and import into Zotero manually.
        </p>
      </article>
    </main>
  </body>
</html>
"""


def build_zotero_export_payload(settings: Settings, paper: PaperRecord, *, public_pdf_url: str | None) -> ZoteroExportPayload:
    authors, affiliations, authors_structured = load_cached_paper_authorship(settings, paper)
    return ZoteroExportPayload(
        paper_id=paper.paper_id,
        title=_clean_text(paper.title),
        authors=[author for author in (_clean_text(item) for item in authors) if author],
        authors_structured=[
            StructuredAuthor(name=_clean_text(author.name), affiliation=_clean_text(author.affiliation) or None)
            for author in authors_structured
            if _clean_text(author.name)
        ],
        affiliations=[item for item in (_clean_text(value) for value in affiliations) if item],
        abstract=_clean_text(paper.abstract),
        venue=_clean_text(paper.venue.upper()),
        year=int(paper.year),
        track=_clean_text(paper.track) or None,
        doi=_clean_text(paper.doi) or None,
        canonical_url=_clean_text(paper.url) or None,
        pdf_url=_clean_text(public_pdf_url or paper.pdf_url) or None,
        source=_clean_text(paper.source) or "acl_anthology",
    )


def render_bibtex(payload: ZoteroExportPayload) -> str:
    key_base = _slugify(payload.authors[0].split()[-1] if payload.authors else payload.paper_id)
    cite_key = f"{key_base}{payload.year}"
    fields: list[tuple[str, str]] = [
        ("title", _bibtex_escape(payload.title)),
        ("author", " and ".join(_bibtex_escape(author) for author in payload.authors) if payload.authors else ""),
        ("booktitle", _bibtex_escape(payload.venue)),
        ("year", str(payload.year)),
    ]
    if payload.abstract:
        fields.append(("abstract", _bibtex_escape(payload.abstract)))
    if payload.doi:
        fields.append(("doi", _bibtex_escape(payload.doi)))
    if payload.canonical_url:
        fields.append(("url", _bibtex_escape(payload.canonical_url)))
    if payload.pdf_url:
        fields.append(("pdf", _bibtex_escape(payload.pdf_url)))
    if payload.track:
        fields.append(("note", _bibtex_escape(f"Track: {payload.track}")))

    rendered_fields = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields if value)
    return f"@{payload.entry_type}{{{cite_key},\n{rendered_fields}\n}}\n"


def render_ris(payload: ZoteroExportPayload) -> str:
    lines = [
        "TY  - CONF",
        f"ID  - {payload.paper_id}",
        f"TI  - {payload.title}",
    ]
    for author in payload.authors:
        lines.append(f"AU  - {author}")
    lines.extend(
        [
            f"PY  - {payload.year}",
            f"T2  - {payload.venue}",
        ]
    )
    if payload.track:
        lines.append(f"N1  - Track: {payload.track}")
    if payload.abstract:
        lines.append(f"AB  - {_ris_escape(payload.abstract)}")
    if payload.doi:
        lines.append(f"DO  - {payload.doi}")
    if payload.canonical_url:
        lines.append(f"UR  - {payload.canonical_url}")
    if payload.pdf_url:
        lines.append(f"L1  - {payload.pdf_url}")
    lines.append("ER  - ")
    return "\n".join(lines) + "\n"


def render_zotero_metadata_page(
    payload: ZoteroExportPayload,
    *,
    bibtex_url: str,
    ris_url: str,
) -> str:
    author_meta = "\n".join(
        f'    <meta name="citation_author" content="{html.escape(author)}" />'
        for author in payload.authors
    )
    dc_creator_meta = "\n".join(
        f'    <meta name="DC.Creator" content="{html.escape(author)}" />'
        for author in payload.authors
    )
    doi_meta = (
        f'    <meta name="citation_doi" content="{html.escape(payload.doi)}" />'
        if payload.doi
        else ""
    )
    pdf_meta = (
        f'    <meta name="citation_pdf_url" content="{html.escape(payload.pdf_url)}" />'
        if payload.pdf_url
        else ""
    )
    public_url_meta = (
        f'    <meta name="citation_public_url" content="{html.escape(payload.canonical_url)}" />\n'
        f'    <meta name="citation_abstract_html_url" content="{html.escape(payload.canonical_url)}" />'
        if payload.canonical_url
        else ""
    )
    dc_source_meta = (
        f'    <meta name="DC.Source" content="{html.escape(payload.venue)}" />'
        if payload.venue
        else ""
    )
    dc_date_meta = f'    <meta name="DC.Date" content="{payload.year}" />'
    affiliations_line = (
        f'<div>{html.escape(" · ".join(payload.affiliations))}</div>' if payload.affiliations else ""
    )
    doi_line = f'<div class="mono">DOI: {html.escape(payload.doi)}</div>' if payload.doi else ""
    source_line = (
        f'<div class="mono">Source: {html.escape(payload.canonical_url)}</div>' if payload.canonical_url else ""
    )
    pdf_link = (
        f'<a class="secondary" href="{html.escape(payload.pdf_url)}" target="_blank" rel="noreferrer">Open PDF</a>'
        if payload.pdf_url
        else ""
    )
    venue_line = html.escape(
        f"{payload.venue} {payload.year}" + (f" · {payload.track}" if payload.track else "")
    )
    canonical_url = html.escape(payload.canonical_url or "#")
    return _HTML_TEMPLATE.format(
        title=html.escape(payload.title),
        year=payload.year,
        venue=html.escape(payload.venue),
        author_meta=author_meta,
        doi_meta=doi_meta,
        pdf_meta=pdf_meta,
        public_url_meta=public_url_meta,
        dc_creator_meta=dc_creator_meta,
        dc_source_meta=dc_source_meta,
        dc_date_meta=dc_date_meta,
        paper_id=html.escape(payload.paper_id),
        authors_line=html.escape(" · ".join(payload.authors) if payload.authors else "Authors unavailable"),
        venue_line=venue_line,
        affiliations_line=affiliations_line,
        doi_line=doi_line,
        source_line=source_line,
        canonical_url=canonical_url,
        pdf_link=pdf_link,
        bibtex_url=html.escape(bibtex_url),
        ris_url=html.escape(ris_url),
        abstract=html.escape(payload.abstract or "No abstract available."),
    )


def build_public_export_path(paper_id: str, suffix: str) -> str:
    return f"/api/papers/{quote(paper_id, safe='')}/{suffix}"


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def _slugify(value: str) -> str:
    lowered = _clean_text(value).lower()
    normalized = _NON_ALNUM_PATTERN.sub("", lowered)
    return normalized or "paper"


def _bibtex_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _ris_escape(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ")
