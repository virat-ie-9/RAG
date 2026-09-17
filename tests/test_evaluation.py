"""Unit and integration tests for the evaluation module."""
from __future__ import annotations
import os, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from ragcv.evaluation import _metrics, build_eval_set, company_from_question
from ragcv.schemas import RetrievalHit

ARCHIVE_DIR = Path("archive")
LIVE_INDEX = (
    Path("data/index/bm25.pkl").exists()
    and Path("data/index/chroma").exists()
    and Path("data/processed/chunks.jsonl").exists()
)
skip_no_index = pytest.mark.skipif(not LIVE_INDEX, reason="Live index not built")
skip_no_archive = pytest.mark.skipif(not ARCHIVE_DIR.exists(), reason="Archive not mounted")


# ─── helpers ──────────────────────────────────────────────────────────────────
def _make_hit(doc_id: str, score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{doc_id}_c0", text="sample", score=score,
        source_path="doc.pdf", doc_id=doc_id,
        page_start=1, page_end=1,
    )


# ─── company_from_question ─────────────────────────────────────────────────────
def test_company_from_question_exact_match():
    names = ["Downer EDI Limited", "ACRES Commercial Realty Corp."]
    assert company_from_question("What did Downer EDI Limited report?", names) == "Downer EDI Limited"


def test_company_from_question_case_insensitive():
    names = ["NextNav Inc."]
    assert company_from_question("nextnav inc. acquisitions?", names) == "NextNav Inc."


def test_company_from_question_longest_wins():
    names = ["ACRES", "ACRES Commercial Realty Corp."]
    result = company_from_question("ACRES Commercial Realty Corp. ESG?", names)
    assert result == "ACRES Commercial Realty Corp."


def test_company_from_question_none():
    names = ["Acme Inc.", "Globex Corp."]
    assert company_from_question("What is the GDP?", names) is None


# ─── _metrics ─────────────────────────────────────────────────────────────────
def test_metrics_perfect_hit():
    eval_set = [{"doc_id": "doc_a"}, {"doc_id": "doc_b"}]
    results = [
        [_make_hit("doc_a"), _make_hit("doc_x")],
        [_make_hit("doc_b"), _make_hit("doc_y")],
    ]
    m = _metrics(results, eval_set, k=5)
    assert m["hit_rate@5"] == 1.0
    assert m["mrr@5"] == 1.0


def test_metrics_zero_hits():
    eval_set = [{"doc_id": "doc_a"}]
    results = [[_make_hit("doc_z"), _make_hit("doc_w")]]
    m = _metrics(results, eval_set, k=5)
    assert m["hit_rate@5"] == 0.0
    assert m["mrr@5"] == 0.0


def test_metrics_partial_hit():
    eval_set = [{"doc_id": "doc_a"}, {"doc_id": "doc_b"}]
    results = [
        [_make_hit("doc_a")],   # hit at rank 1  → rr = 1.0
        [_make_hit("doc_z")],   # miss           → rr = 0.0
    ]
    m = _metrics(results, eval_set, k=5)
    assert m["hit_rate@5"] == 0.5
    assert abs(m["mrr@5"] - 0.5) < 1e-9


def test_metrics_hit_at_rank_2():
    eval_set = [{"doc_id": "doc_a"}]
    results = [[_make_hit("doc_z"), _make_hit("doc_a")]]  # hit at rank 2
    m = _metrics(results, eval_set, k=5)
    assert m["hit_rate@5"] == 1.0
    assert abs(m["mrr@5"] - 0.5) < 1e-9


def test_metrics_k_cutoff():
    eval_set = [{"doc_id": "doc_a"}]
    # doc_a is at rank 3, but k=2 → miss
    results = [[_make_hit("doc_z"), _make_hit("doc_w"), _make_hit("doc_a")]]
    m = _metrics(results, eval_set, k=2)
    assert m["hit_rate@2"] == 0.0


def test_metrics_empty_eval_set():
    m = _metrics([], [], k=5)
    assert m["hit_rate@5"] == 0.0
    assert m["mrr@5"] == 0.0


# ─── build_eval_set ───────────────────────────────────────────────────────────
@skip_no_archive
def test_build_eval_set_returns_list():
    rows = build_eval_set(ARCHIVE_DIR, limit=5)
    assert isinstance(rows, list)
    assert len(rows) > 0


@skip_no_archive
def test_build_eval_set_has_required_keys():
    rows = build_eval_set(ARCHIVE_DIR, limit=5)
    for row in rows:
        assert "question" in row
        assert "company_name" in row
        assert "doc_id" in row


@skip_no_archive
def test_build_eval_set_limit():
    rows = build_eval_set(ARCHIVE_DIR, limit=3)
    assert len(rows) <= 3


@skip_no_archive
def test_build_eval_set_available_doc_ids_filter():
    """Only questions matching indexed docs are returned when available_doc_ids is set."""
    all_rows = build_eval_set(ARCHIVE_DIR, limit=50)
    if not all_rows:
        pytest.skip("No eval rows built")
    # Use only the first doc_id
    one_id = {all_rows[0]["doc_id"]}
    filtered = build_eval_set(ARCHIVE_DIR, limit=50, available_doc_ids=one_id)
    assert all(row["doc_id"] in one_id for row in filtered)


# ─── evaluate_retrieval (integration) ─────────────────────────────────────────
@skip_no_index
def test_evaluate_retrieval_returns_all_metrics():
    from ragcv.config import get_settings
    from ragcv.evaluation import evaluate_retrieval
    from ragcv.pipeline import make_retriever

    settings = get_settings()
    r = make_retriever(settings, use_reranker=False)
    results = evaluate_retrieval(r, settings, limit=4, k=5)
    assert "dense" in results
    assert "bm25" in results
    assert "hybrid" in results
    assert "hybrid_reranker" in results
    assert "hit_rate@5" in results["dense"]
    assert "mrr@5" in results["dense"]
    assert results["eval_set_size"] >= 1


@skip_no_index
def test_evaluate_retrieval_metrics_in_range():
    from ragcv.config import get_settings
    from ragcv.evaluation import evaluate_retrieval
    from ragcv.pipeline import make_retriever

    settings = get_settings()
    r = make_retriever(settings, use_reranker=False)
    results = evaluate_retrieval(r, settings, limit=4, k=5)
    for method in ("dense", "bm25", "hybrid", "hybrid_reranker"):
        hr = results[method]["hit_rate@5"]
        mrr = results[method]["mrr@5"]
        assert 0.0 <= hr <= 1.0, f"{method} hit_rate@5={hr} out of range"
        assert 0.0 <= mrr <= 1.0, f"{method} mrr@5={mrr} out of range"
