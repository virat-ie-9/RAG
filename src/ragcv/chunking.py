import hashlib
import re

from .schemas import Chunk, DocumentPage


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_pages(pages: list[DocumentPage], chunk_size: int = 320, overlap: int = 60) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    for page in pages:
        words = _tokenize(normalize_text(page.text))
        if not words:
            continue
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            if len(window) < 35 and start > 0:
                continue
            text = " ".join(window)
            digest = hashlib.sha1(f"{page.doc_id}:{page.page}:{start}:{text[:80]}".encode("utf-8")).hexdigest()[:16]
            chunks.append(
                Chunk(
                    chunk_id=f"{page.doc_id}_{page.page}_{digest}",
                    doc_id=page.doc_id,
                    source_path=page.source_path,
                    page_start=page.page,
                    page_end=page.page,
                    text=text,
                    company_name=page.company_name,
                    currency=page.currency,
                    industry=page.industry,
                    section_title=page.section_title,
                )
            )
    return chunks
