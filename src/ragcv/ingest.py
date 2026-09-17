import re
from pathlib import Path

from .io_utils import read_json
from .schemas import DocumentPage

PAGE_MARKER = re.compile(r"^\{(\d+)\}-+\s*$", re.MULTILINE)


def load_subset_metadata(archive_dir: Path) -> dict[str, dict]:
    subset_path = archive_dir / "subset.json"
    if not subset_path.exists():
        return {}
    return {row["sha1"]: row for row in read_json(subset_path)}


def inspect_archive(archive_dir: Path) -> dict:
    files = [p for p in archive_dir.rglob("*") if p.is_file()]
    by_ext: dict[str, int] = {}
    for path in files:
        ext = path.suffix.lower().lstrip(".") or "no_extension"
        by_ext[ext] = by_ext.get(ext, 0) + 1
    md_root = archive_dir / "EnterpriseRAG_2025_02_markdown"
    return {
        "archive_dir": str(archive_dir),
        "total_files": len(files),
        "file_types": dict(sorted(by_ext.items())),
        "root_pdfs": len(list(archive_dir.glob("*.pdf"))),
        "markdown_documents": len(list(md_root.glob("*/*.md"))) if md_root.exists() else 0,
        "metadata_files": len(list(md_root.glob("*/*_meta.json"))) if md_root.exists() else 0,
        "image_assets": sum(1 for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
    }


def _metadata_for(doc_id: str, subset: dict[str, dict]) -> dict:
    meta = subset.get(doc_id, {})
    return {
        "company_name": meta.get("company_name"),
        "currency": meta.get("cur"),
        "industry": meta.get("major_industry"),
    }


def load_layout_headings(archive_dir: Path, doc_id: str) -> dict[int, str]:
    """Load table of contents from <doc_id>_meta.json and map page_number (1-indexed) to heading title."""
    meta_path = archive_dir / "EnterpriseRAG_2025_02_markdown" / doc_id / f"{doc_id}_meta.json"
    if not meta_path.exists():
        return {}
    try:
        data = read_json(meta_path)
        toc = data.get("table_of_contents", [])
        page_to_heading: dict[int, str] = {}
        for entry in toc:
            p_id = entry.get("page_id")
            title = entry.get("title", "")
            if p_id is not None and title:
                clean_title = re.sub(r"\s+", " ", title).strip()
                page_num = p_id + 1
                if page_num not in page_to_heading:
                    page_to_heading[page_num] = clean_title
        return page_to_heading
    except Exception:
        return {}


def _pages_from_markdown(md_path: Path, archive_dir: Path, subset: dict[str, dict]) -> list[DocumentPage]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    matches = list(PAGE_MARKER.finditer(text))
    doc_id = md_path.stem
    meta = _metadata_for(doc_id, subset)
    headings = load_layout_headings(archive_dir, doc_id)
    pages: list[DocumentPage] = []
    if not matches:
        cleaned = re.sub(r"[ \t]+", " ", text).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        if cleaned:
            pages.append(
                DocumentPage(
                    doc_id=doc_id,
                    source_path=str(md_path.relative_to(archive_dir)),
                    page=1,
                    text=cleaned,
                    section_title=headings.get(1),
                    **meta,
                )
            )
        return pages
    for i, match in enumerate(matches):
        page_number = int(match.group(1)) + 1
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        page_text = re.sub(r"[ \t]+", " ", page_text)
        page_text = re.sub(r"\n{3,}", "\n\n", page_text)
        if len(page_text) >= 40:
            pages.append(
                DocumentPage(
                    doc_id=doc_id,
                    source_path=str(md_path.relative_to(archive_dir)),
                    page=page_number,
                    text=page_text,
                    section_title=headings.get(page_number),
                    **meta,
                )
            )
    return pages


def _pages_from_pdf(pdf_path: Path, archive_dir: Path, subset: dict[str, dict]) -> list[DocumentPage]:
    import fitz

    doc_id = pdf_path.stem
    meta = _metadata_for(doc_id, subset)
    pages: list[DocumentPage] = []
    with fitz.open(pdf_path) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text("text")
            text = re.sub(r"[ \t]+", " ", text).strip()
            text = re.sub(r"\n{3,}", "\n\n", text)
            if len(text) >= 40:
                pages.append(
                    DocumentPage(
                        doc_id=doc_id,
                        source_path=str(pdf_path.relative_to(archive_dir)),
                        page=i + 1,
                        text=text,
                        **meta,
                    )
                )
    return pages


def load_documents(archive_dir: Path, prefer_markdown: bool = True) -> list[DocumentPage]:
    subset = load_subset_metadata(archive_dir)
    pages: list[DocumentPage] = []
    md_root = archive_dir / "EnterpriseRAG_2025_02_markdown"
    if prefer_markdown and md_root.exists():
        for md_path in sorted(md_root.glob("*/*.md")):
            pages.extend(_pages_from_markdown(md_path, archive_dir, subset))
    seen_docs = {p.doc_id for p in pages}
    for pdf_path in sorted(archive_dir.glob("*.pdf")):
        if pdf_path.stem not in seen_docs:
            pages.extend(_pages_from_pdf(pdf_path, archive_dir, subset))
    return pages
