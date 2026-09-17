from ragcv.bm25 import BM25Index, tokenize
from ragcv.retrieval import _extract_company_name, reciprocal_rank_fusion
from ragcv.schemas import Chunk, RetrievalHit


# ─── helpers ───────────────────────────────────────────────────────────────────
def make_hit(chunk_id: str, doc_id: str = "doc", score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        text="sample text",
        score=score,
        source_path="doc.pdf",
        doc_id=doc_id,
        page_start=1,
        page_end=1,
    )


def make_chunk(
    chunk_id: str,
    text: str,
    doc_id: str = "doc",
    company: str | None = None,
    industry: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_path=f"{doc_id}.md",
        page_start=1,
        page_end=1,
        text=text,
        company_name=company,
        industry=industry,
    )


# ─── tokenizer ──────────────────────────────────────────────────────────────────
def test_tokenize_handles_financial_symbols():
    tokens = tokenize("Revenue was $1.5B, up 12.3% YoY")
    assert "$1.5b" in tokens
    assert "12.3%" in tokens
    assert "revenue" in tokens


def test_tokenize_empty_string():
    assert tokenize("") == []


# ─── RRF ────────────────────────────────────────────────────────────────────────
def test_rrf_combines_duplicate_hits():
    fused = reciprocal_rank_fusion([[make_hit("a"), make_hit("b")], [make_hit("b"), make_hit("c")]], top_k=3)
    ids = [h.chunk_id for h in fused]
    assert ids[0] == "b"          # "b" appears in both lists → highest RRF score
    assert set(ids) == {"a", "b", "c"}
    assert fused[0].retriever == "hybrid"


def test_rrf_single_list():
    fused = reciprocal_rank_fusion([[make_hit("x"), make_hit("y")]], k=60, top_k=2)
    assert [h.chunk_id for h in fused] == ["x", "y"]


def test_rrf_score_formula():
    """Verify the RRF formula: score(rank=1, k=60) = 1/(60+1) ≈ 0.01639."""
    fused = reciprocal_rank_fusion([[make_hit("only")]], k=60, top_k=1)
    expected = 1.0 / (60 + 1)
    assert abs(fused[0].score - expected) < 1e-5


def test_rrf_top_k_limiting():
    hits = [make_hit(f"h{i}") for i in range(10)]
    fused = reciprocal_rank_fusion([hits], top_k=3)
    assert len(fused) == 3


# ─── company-name extraction ────────────────────────────────────────────────────
def test_extract_company_name_exact():
    names = ["Downer EDI Limited", "ACRES Commercial Realty Corp.", "NextNav Inc."]
    q = "Did Downer EDI Limited announce a share buyback?"
    assert _extract_company_name(q, names) == "Downer EDI Limited"


def test_extract_company_name_case_insensitive():
    names = ["NextNav Inc."]
    assert _extract_company_name("nextnav inc. recent acquisitions", names) == "NextNav Inc."


def test_extract_company_name_longest_wins():
    names = ["ACRES", "ACRES Commercial Realty Corp."]
    result = _extract_company_name("what did acres commercial realty corp. report?", names)
    assert result == "ACRES Commercial Realty Corp."


def test_extract_company_name_none_found():
    names = ["Acme Inc.", "Globex Corp."]
    assert _extract_company_name("what is the CEO salary?", names) is None


# ─── BM25Index ──────────────────────────────────────────────────────────────────
def test_bm25_empty_index():
    idx = BM25Index([])
    assert idx.count() == 0
    assert idx.search("anything") == []


def test_bm25_basic_ranking():
    chunks = [
        make_chunk("c1", "The company announced a share buyback programme for common stock", company="Acme"),
        make_chunk("c2", "Revenue increased dramatically in fiscal year 2023 quarter four", company="Acme"),
        make_chunk("c3", "Buyback plan details: 1 million shares repurchased at market price", company="Acme"),
    ]
    idx = BM25Index(chunks)
    hits = idx.search("share buyback repurchase plan", top_k=3)
    assert hits, "Should return at least one hit"
    top_ids = [h.chunk_id for h in hits]
    # c1 and c3 contain buyback-related terms; at least one of them should rank first
    assert top_ids[0] in {"c1", "c3"}


def test_bm25_where_filter():
    chunks = [
        make_chunk("c1", "capital expenditure investment strategy growth", company="Alpha Inc.", industry="Tech"),
        make_chunk("c2", "capital expenditure investment strategy growth", company="Beta Corp.", industry="Finance"),
    ]
    idx = BM25Index(chunks)
    hits = idx.search("capital expenditure", top_k=5, where={"company_name": "Alpha Inc."})
    assert len(hits) == 1
    assert hits[0].chunk_id == "c1"


def test_bm25_empty_query():
    chunks = [make_chunk("c1", "some financial data here")]
    idx = BM25Index(chunks)
    assert idx.search("") == []


def test_bm25_save_and_load(tmp_path):
    chunks = [
        make_chunk("c1", "revenue profit margin ebitda", company="SaveCo"),
        make_chunk("c2", "operational expenses overhead cost", company="OtherCo"),
    ]
    idx = BM25Index(chunks)
    save_path = tmp_path / "bm25.pkl"
    idx.save(save_path)
    loaded = BM25Index.load(save_path)
    # Use where filter: ensures the right chunk is returned even in tiny corpus
    hits = loaded.search("revenue profit", top_k=1, where={"company_name": "SaveCo"})
    assert len(hits) == 1
    assert hits[0].chunk_id == "c1"

