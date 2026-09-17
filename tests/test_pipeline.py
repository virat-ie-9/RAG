"""Integration tests for the complete RAG pipeline.

Requires the live 5-doc index built via: scripts/build_index.py --max-docs 5
All tests are read-only.
"""
from __future__ import annotations
import os, re, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from ragcv.llm import _detect_boolean_intent, _grounded_extractive_answer, estimate_confidence
from ragcv.pipeline import answer_question, make_retriever
from ragcv.schemas import RetrievalHit


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _hit(score: float = 1.0, rerank: float | None = None, text: str = "sample text") -> RetrievalHit:
    return RetrievalHit(
        chunk_id="x", text=text, score=score,
        source_path="doc.pdf", doc_id="abc",
        page_start=1, page_end=1, company_name="Acme Corp", rerank_score=rerank,
    )

LIVE_INDEX = (
    Path("data/index/bm25.pkl").exists()
    and Path("data/index/chroma").exists()
    and Path("data/processed/chunks.jsonl").exists()
)
skip_no_index = pytest.mark.skipif(not LIVE_INDEX, reason="Live index not built")


# ─── Unit: confidence estimation ──────────────────────────────────────────────
def test_confidence_insufficient_empty():
    assert estimate_confidence([]) == "insufficient"

def test_confidence_high_strong_rerank():
    assert estimate_confidence([_hit(rerank=8.5)]) == "high"

def test_confidence_medium_positive_rerank():
    assert estimate_confidence([_hit(rerank=2.1)]) == "medium"

def test_confidence_low_negative_rerank():
    assert estimate_confidence([_hit(rerank=-1.0)]) == "low"

def test_confidence_medium_high_dense():
    assert estimate_confidence([_hit(score=0.80)]) == "medium"

def test_confidence_low_weak_dense():
    assert estimate_confidence([_hit(score=0.60)]) == "low"


# ─── Unit: boolean intent detection ───────────────────────────────────────────
def test_boolean_intent_buyback():
    is_bool, kws = _detect_boolean_intent(
        "Did Downer EDI Limited announce a share buyback plan? If there is no mention, return False."
    )
    assert is_bool and "buyback" in kws

def test_boolean_intent_esg():
    is_bool, kws = _detect_boolean_intent(
        "Did ACRES outline any new ESG initiatives in the annual report?"
    )
    assert is_bool and any("esg" in k or "sustainability" in k for k in kws)

def test_non_boolean_factual():
    is_bool, _ = _detect_boolean_intent("What is the total revenue reported?")
    assert not is_bool


# ─── Unit: grounded extractive answer ─────────────────────────────────────────
def test_extractive_true_buyback():
    hits = [_hit(text="The company repurchased 5 million shares on-market during the year.")]
    answer = _grounded_extractive_answer(
        "Did the company announce a share buyback? If there is no mention, return False.", hits
    )
    assert answer.startswith("True") and "[1]" in answer

def test_extractive_false_no_buyback():
    hits = [_hit(text="Revenue increased by 12% in fiscal year 2023.")]
    answer = _grounded_extractive_answer(
        "Did the company announce a share buyback? If there is no mention, return False.", hits
    )
    assert answer.startswith("False")

def test_extractive_factual_has_citation():
    hits = [_hit(text="Total employees at year-end numbered 42,000 across all divisions globally.")]
    answer = _grounded_extractive_answer("How many employees does the company have?", hits)
    assert re.search(r"\[\d+\]", answer), f"No citation: {answer!r}"

def test_extractive_empty_hits():
    answer = _grounded_extractive_answer("What is the CEO salary?", [])
    assert "do not have enough evidence" in answer.lower()


# ─── Integration: live index ──────────────────────────────────────────────────
@skip_no_index
def test_crossencoder_loaded():
    r = make_retriever()
    assert r.reranker is not None

@skip_no_index
def test_rerank_sets_scores():
    r = make_retriever()
    q = "Did Downer EDI Limited announce a share buyback plan?"
    candidates = r.hybrid(q, top_k=10)
    reranked = r.rerank(q, candidates, top_k=8)
    assert all(h.rerank_score is not None for h in reranked)
    assert all(h.retriever == "hybrid+reranker" for h in reranked)

@skip_no_index
def test_rerank_changes_ordering_from_rrf():
    r = make_retriever()
    q = "Did Downer EDI Limited announce a share buyback plan?"
    hybrid_hits = r.hybrid(q, top_k=10)
    reranked = r.rerank(q, hybrid_hits, top_k=8)
    rrf_scores = [h.score for h in hybrid_hits[:8]]
    re_scores = [h.rerank_score for h in reranked]
    assert rrf_scores != re_scores

@skip_no_index
def test_answer_response_has_all_fields():
    resp = answer_question(
        "Did Downer EDI Limited announce a share buyback plan in the annual report? "
        "If there is no mention, return False."
    )
    assert resp.question
    assert resp.answer
    assert resp.confidence in {"high", "medium", "low", "insufficient"}
    assert len(resp.citations) > 0
    assert isinstance(resp.reranker_used, bool)
    assert resp.num_hits_retrieved > 0

@skip_no_index
def test_citations_have_company_and_pages():
    resp = answer_question(
        "Did Downer EDI Limited announce a share buyback plan? "
        "If there is no mention, return False."
    )
    for c in resp.citations:
        assert c.company_name
        assert c.pages

@skip_no_index
def test_boolean_question_answer_format():
    resp = answer_question(
        "Did ACRES Commercial Realty Corp. outline any new ESG initiatives "
        "in the annual report? If there is no mention, return False."
    )
    first_word = resp.answer.strip().split()[0].rstrip(".")
    assert first_word in {"True", "False"}, f"Got: {resp.answer[:80]!r}"

@skip_no_index
def test_grounded_answer_has_inline_citation():
    resp = answer_question(
        "Did Downer EDI Limited announce a share buyback plan? "
        "If there is no mention, return False."
    )
    # False answers don't have citations; True answers must
    if not resp.answer.strip().startswith("False"):
        assert re.search(r"\[\d+\]", resp.answer), f"No [N] in: {resp.answer!r}"

@skip_no_index
def test_reranker_used_flag_true():
    resp = answer_question("What is the industry of Aptevo Therapeutics Inc.?")
    assert resp.reranker_used is True

@skip_no_index
def test_section_title_in_citations():
    resp = answer_question(
        "Did Downer EDI Limited announce a share buyback plan? "
        "If there is no mention, return False."
    )
    titles = [c.section_title for c in resp.citations if c.section_title]
    assert len(titles) > 0, "Expected at least one citation with a section_title"
