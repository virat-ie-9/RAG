import json
from pathlib import Path

from ragcv.ingest import (
    inspect_archive,
    load_documents,
    load_layout_headings,
    load_subset_metadata,
)


def test_inspect_archive_counts_files(tmp_path: Path):
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "a.pdf").write_text("x")
    (archive / "questions.json").write_text("[]")
    summary = inspect_archive(archive)
    assert summary["root_pdfs"] == 1
    assert summary["file_types"]["pdf"] == 1
    assert summary["file_types"]["json"] == 1


def test_load_subset_metadata(tmp_path: Path):
    archive = tmp_path / "archive"
    archive.mkdir()
    subset_data = [
        {"sha1": "doc123", "company_name": "Test Co", "cur": "USD", "major_industry": "Tech"}
    ]
    (archive / "subset.json").write_text(json.dumps(subset_data))
    meta = load_subset_metadata(archive)
    assert "doc123" in meta
    assert meta["doc123"]["company_name"] == "Test Co"


def test_load_layout_headings(tmp_path: Path):
    archive = tmp_path / "archive"
    doc_dir = archive / "EnterpriseRAG_2025_02_markdown" / "doc123"
    doc_dir.mkdir(parents=True)
    meta_content = {
        "table_of_contents": [
            {"title": "Overview", "page_id": 0},
            {"title": "Financial Report", "page_id": 2},
        ]
    }
    (doc_dir / "doc123_meta.json").write_text(json.dumps(meta_content))
    headings = load_layout_headings(archive, "doc123")
    assert headings.get(1) == "Overview"
    assert headings.get(3) == "Financial Report"


def test_pages_from_markdown_with_delimiters(tmp_path: Path):
    archive = tmp_path / "archive"
    doc_dir = archive / "EnterpriseRAG_2025_02_markdown" / "doc123"
    doc_dir.mkdir(parents=True)
    subset_data = [{"sha1": "doc123", "company_name": "Acme Corp", "cur": "USD", "major_industry": "Finance"}]
    (archive / "subset.json").write_text(json.dumps(subset_data))
    meta_content = {
        "table_of_contents": [
            {"title": "Section 1: Executive Summary", "page_id": 0},
            {"title": "Section 2: Revenue Numbers", "page_id": 1},
        ]
    }
    (doc_dir / "doc123_meta.json").write_text(json.dumps(meta_content))
    md_text = (
        "{0}------------------------------------------------\n\n"
        "# Executive Summary\n\n"
        "This is the executive summary text for Acme Corp containing sufficient words.\n\n"
        "{1}------------------------------------------------\n\n"
        "# Revenue Numbers\n\n"
        "Revenue for fiscal year 2024 increased by 25 percent to one billion dollars.\n"
    )
    (doc_dir / "doc123.md").write_text(md_text)

    pages = load_documents(archive, prefer_markdown=True)
    assert len(pages) == 2
    assert pages[0].doc_id == "doc123"
    assert pages[0].page == 1
    assert pages[0].company_name == "Acme Corp"
    assert pages[0].section_title == "Section 1: Executive Summary"
    assert pages[1].page == 2
    assert pages[1].section_title == "Section 2: Revenue Numbers"


def test_archive_remains_read_only(tmp_path: Path):
    archive = tmp_path / "archive"
    doc_dir = archive / "EnterpriseRAG_2025_02_markdown" / "doc1"
    doc_dir.mkdir(parents=True)
    (doc_dir / "doc1.md").write_text("{0}------------------------------------------------\n\nPage 1 text content that is sufficiently long.")
    
    # Snapshot files before ingestion
    before_files = set(archive.rglob("*"))
    before_mtimes = {p: p.stat().st_mtime for p in before_files if p.is_file()}

    # Run ingestion
    pages = load_documents(archive)
    assert len(pages) == 1

    # Verify no files were added, deleted, or touched
    after_files = set(archive.rglob("*"))
    after_mtimes = {p: p.stat().st_mtime for p in after_files if p.is_file()}

    assert before_files == after_files
    assert before_mtimes == after_mtimes

