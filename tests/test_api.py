"""FastAPI endpoint tests using TestClient (no real server needed).

These tests exercise every route with in-process calls — no network required.
Index-dependent tests are skipped automatically when the index is absent.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# Patch get_settings cache so tests use the project directory
from ragcv.config import get_settings
get_settings.cache_clear()

from api.main import app

client = TestClient(app, raise_server_exceptions=True)

LIVE_INDEX = (
    Path("data/index/bm25.pkl").exists()
    and Path("data/index/chroma").exists()
    and Path("data/processed/chunks.jsonl").exists()
)
skip_no_index = pytest.mark.skipif(not LIVE_INDEX, reason="Live index not built")


# ─── /health ──────────────────────────────────────────────────────────────────
def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "archive_exists" in body
    assert "chunks_exists" in body
    assert "bm25_exists" in body
    assert "chroma_dir_exists" in body


def test_health_archive_exists():
    resp = client.get("/health")
    assert resp.json()["archive_exists"] is True


# ─── /dataset ─────────────────────────────────────────────────────────────────
def test_dataset_returns_dict():
    resp = client.get("/dataset")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)


# ─── /index-stats ─────────────────────────────────────────────────────────────
@skip_no_index
def test_index_stats_returns_chunk_counts():
    resp = client.get("/index-stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "bm25_chunks" in body
    assert "chroma_chunks" in body
    assert body["bm25_chunks"] > 0
    assert body["chroma_chunks"] > 0


def test_index_stats_503_when_no_index(tmp_path, monkeypatch):
    """If BM25 index is absent, /index-stats returns 503."""
    if LIVE_INDEX:
        pytest.skip("Index exists — 503 path not testable")
    resp = client.get("/index-stats")
    assert resp.status_code == 503


# ─── /ask ─────────────────────────────────────────────────────────────────────
def test_ask_empty_question_returns_400():
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code in {400, 422}  # 422 from pydantic min_length


def test_ask_too_short_returns_422():
    resp = client.post("/ask", json={"question": "Hi"})
    assert resp.status_code == 422


def test_ask_invalid_top_k_returns_422():
    resp = client.post("/ask", json={"question": "What is the revenue?", "top_k": 0})
    assert resp.status_code == 422


def test_ask_top_k_above_max_returns_422():
    resp = client.post("/ask", json={"question": "What is the revenue?", "top_k": 99})
    assert resp.status_code == 422


@skip_no_index
def test_ask_returns_answer_response():
    resp = client.post(
        "/ask",
        json={
            "question": "Did Downer EDI Limited announce a share buyback plan? If there is no mention, return False.",
            "top_k": 4,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "confidence" in body
    assert "citations" in body
    assert "retrieval_hits" in body
    assert body["confidence"] in {"high", "medium", "low", "insufficient"}


@skip_no_index
def test_ask_response_has_new_fields():
    resp = client.post(
        "/ask",
        json={"question": "What ESG initiatives did ACRES Commercial Realty Corp. outline?", "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "reranker_used" in body
    assert "num_hits_retrieved" in body
    assert body["num_hits_retrieved"] > 0


@skip_no_index
def test_ask_citations_have_section_title_field():
    resp = client.post(
        "/ask",
        json={"question": "Did Downer EDI Limited announce a share buyback plan? If there is no mention, return False.", "top_k": 4},
    )
    body = resp.json()
    # section_title key must exist (can be null) on every citation
    for c in body["citations"]:
        assert "section_title" in c


# ─── /evaluate ────────────────────────────────────────────────────────────────
@skip_no_index
def test_evaluate_returns_metrics():
    resp = client.post("/evaluate", json={"limit": 4, "k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert "dense" in body
    assert "bm25" in body
    assert "hybrid" in body
    assert "hybrid_reranker" in body
    assert "hit_rate@5" in body["dense"]
    assert "mrr@5" in body["dense"]


def test_evaluate_503_when_no_index():
    if LIVE_INDEX:
        pytest.skip("Index exists — 503 path not testable")
    resp = client.post("/evaluate", json={"limit": 2, "k": 3})
    assert resp.status_code == 503


def test_evaluate_invalid_limit_returns_422():
    resp = client.post("/evaluate", json={"limit": 0, "k": 5})
    assert resp.status_code == 422
