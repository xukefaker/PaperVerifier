from __future__ import annotations

from pathlib import Path

from chemverify.config import Settings
from chemverify.models import PaperRecord
from chemverify.zotero_export import (
    build_public_export_path,
    build_zotero_export_payload,
    render_bibtex,
    render_ris,
    render_zotero_metadata_page,
)


def test_zotero_export_renderers_include_expected_fields(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    paper = PaperRecord(
        paper_id="2024.acl-long.1",
        title="Test Paper for Zotero",
        authors=["Ada Lovelace", "Grace Hopper"],
        venue="acl",
        year=2024,
        track="long",
        abstract="We evaluate a scholarly retrieval workflow.",
        doi="10.1000/test-doi",
        url="https://aclanthology.org/2024.acl-long.1/",
        pdf_url="https://aclanthology.org/2024.acl-long.1.pdf",
    )

    payload = build_zotero_export_payload(
        settings,
        paper,
        public_pdf_url="/api/papers/2024.acl-long.1/pdf",
    )
    bibtex = render_bibtex(payload)
    ris = render_ris(payload)
    html = render_zotero_metadata_page(
        payload,
        bibtex_url=build_public_export_path(payload.paper_id, "export.bib"),
        ris_url=build_public_export_path(payload.paper_id, "export.ris"),
    )

    assert payload.title == "Test Paper for Zotero"
    assert payload.venue == "ACL"
    assert payload.pdf_url == "/api/papers/2024.acl-long.1/pdf"

    assert "@inproceedings" in bibtex
    assert "Ada Lovelace and Grace Hopper" in bibtex
    assert "booktitle = {ACL}" in bibtex
    assert "pdf = {/api/papers/2024.acl-long.1/pdf}" in bibtex

    assert "TY  - CONF" in ris
    assert "AU  - Ada Lovelace" in ris
    assert "DO  - 10.1000/test-doi" in ris
    assert "L1  - /api/papers/2024.acl-long.1/pdf" in ris

    assert 'name="citation_title"' in html
    assert 'name="citation_author" content="Ada Lovelace"' in html
    assert 'name="citation_pdf_url" content="/api/papers/2024.acl-long.1/pdf"' in html
    assert 'href="/api/papers/2024.acl-long.1/export.bib"' in html
