import pytest

from ragcv.chunking import chunk_pages
from ragcv.schemas import DocumentPage


def test_chunk_pages_preserves_metadata():
    page = DocumentPage(
        doc_id="abc",
        source_path="abc.pdf",
        page=7,
        text=" ".join(["revenue"] * 120),
        company_name="Example Inc.",
        currency="USD",
        industry="Technology",
        section_title="Item 1A: Risk Factors",
    )
    chunks = chunk_pages([page], chunk_size=50, overlap=10)
    assert chunks
    assert chunks[0].doc_id == "abc"
    assert chunks[0].page_start == 7
    assert chunks[0].page_end == 7
    assert chunks[0].company_name == "Example Inc."
    assert chunks[0].section_title == "Item 1A: Risk Factors"
    assert chunks[0].chunk_id.startswith("abc_7_")


def test_chunk_pages_invalid_overlap():
    page = DocumentPage(doc_id="x", source_path="x.md", page=1, text="hello world")
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        chunk_pages([page], chunk_size=50, overlap=50)


def test_chunk_pages_sliding_window():
    words = [f"word{i}" for i in range(100)]
    page = DocumentPage(doc_id="x", source_path="x.md", page=1, text=" ".join(words))
    chunks = chunk_pages([page], chunk_size=40, overlap=10)
    assert len(chunks) > 1
    # Check that overlapping words appear in consecutive chunks
    c0_words = chunks[0].text.split()
    c1_words = chunks[1].text.split()
    assert c0_words[-10:] == c1_words[:10]

