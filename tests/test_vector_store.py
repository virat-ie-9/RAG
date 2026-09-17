import pytest
from pathlib import Path

from ragcv.schemas import Chunk
from ragcv.vector_store import ChromaVectorStore


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="doc1_p1_c1",
            doc_id="doc1",
            source_path="reports/doc1.md",
            page_start=1,
            page_end=1,
            text="ACRES Commercial Realty Corp announced a new share buyback program for common stock.",
            company_name="ACRES Commercial Realty Corp.",
            currency="USD",
            industry="Financial Services",
            section_title="Capital Markets",
        ),
        Chunk(
            chunk_id="doc1_p2_c2",
            doc_id="doc1",
            source_path="reports/doc1.md",
            page_start=2,
            page_end=2,
            text="Operating expenses decreased primarily due to lower legal fees and general administrative costs.",
            company_name="ACRES Commercial Realty Corp.",
            currency="USD",
            industry="Financial Services",
            section_title="Operating Results",
        ),
        Chunk(
            chunk_id="doc2_p1_c1",
            doc_id="doc2",
            source_path="reports/doc2.md",
            page_start=1,
            page_end=1,
            text="NextNav Inc specializes in next-generation 3D geolocation and navigation technology services.",
            company_name="NextNav Inc.",
            currency="USD",
            industry="Technology",
            section_title="Business Overview",
        ),
    ]


def test_vector_store_empty_search(tmp_path: Path):
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma_test",
        collection_name="test_empty",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    assert store.count() == 0
    hits = store.search("anything", top_k=5)
    assert hits == []


def test_vector_store_build_and_search(tmp_path: Path, sample_chunks: list[Chunk]):
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma_test",
        collection_name="test_reports",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    store.build(sample_chunks, reset=True)
    assert store.count() == 3

    # Semantic query about stock buyback
    hits = store.search("share repurchase program common stock", top_k=2)
    assert len(hits) == 2
    top_hit = hits[0]
    assert top_hit.chunk_id == "doc1_p1_c1"
    assert top_hit.company_name == "ACRES Commercial Realty Corp."
    assert top_hit.section_title == "Capital Markets"
    assert top_hit.score > 0.5


def test_vector_store_metadata_filtering(tmp_path: Path, sample_chunks: list[Chunk]):
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma_test",
        collection_name="test_filter",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    store.build(sample_chunks, reset=True)

    # Filter strictly for NextNav Inc.
    hits = store.search(
        "buyback program or navigation",
        top_k=5,
        where={"company_name": "NextNav Inc."},
    )
    assert len(hits) == 1
    assert hits[0].doc_id == "doc2"
    assert hits[0].company_name == "NextNav Inc."
