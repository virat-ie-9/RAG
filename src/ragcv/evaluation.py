import json
import re
from pathlib import Path

from .config import Settings
from .ingest import load_subset_metadata
from .retrieval import HybridRetriever


def company_from_question(question: str, company_names: list[str]) -> str | None:
    lower = question.lower()
    matches = [name for name in company_names if name.lower() in lower]
    return max(matches, key=len) if matches else None


def build_eval_set(archive_dir: Path, limit: int = 30, available_doc_ids: set[str] | None = None) -> list[dict]:
    subset = load_subset_metadata(archive_dir)
    company_to_sha = {v.get("company_name"): sha for sha, v in subset.items() if v.get("company_name")}
    if available_doc_ids:
        company_to_sha = {company: sha for company, sha in company_to_sha.items() if sha in available_doc_ids}
    names = list(company_to_sha)
    questions_path = archive_dir / "questions.json"
    rows = json.loads(questions_path.read_text(encoding="utf-8")) if questions_path.exists() else []
    eval_rows: list[dict] = []
    for row in rows:
        company = company_from_question(row["text"], names)
        if company:
            eval_rows.append({"question": row["text"], "company_name": company, "doc_id": company_to_sha[company], "kind": row.get("kind")})
        if len(eval_rows) >= limit:
            break
    if eval_rows:
        return eval_rows
    for company, sha in list(company_to_sha.items())[:limit]:
        eval_rows.append({"question": f"What does the annual report say about {company}?", "company_name": company, "doc_id": sha, "kind": "metadata"})
    return eval_rows


def _metrics(results: list[list], eval_set: list[dict], k: int) -> dict:
    hits = 0
    reciprocal = 0.0
    for row, retrieved in zip(eval_set, results):
        ranks = [i + 1 for i, hit in enumerate(retrieved[:k]) if hit.doc_id == row["doc_id"]]
        if ranks:
            hits += 1
            reciprocal += 1.0 / ranks[0]
    n = max(len(eval_set), 1)
    return {f"hit_rate@{k}": hits / n, f"mrr@{k}": reciprocal / n}


def evaluate_retrieval(retriever: HybridRetriever, settings: Settings, limit: int = 30, k: int = 5) -> dict:
    available_doc_ids = {chunk.doc_id for chunk in retriever.bm25.chunks}
    eval_set = build_eval_set(settings.archive_dir, limit=limit, available_doc_ids=available_doc_ids)
    dense = [retriever.dense(row["question"], top_k=k) for row in eval_set]
    bm25 = [retriever.lexical(row["question"], top_k=k) for row in eval_set]
    hybrid = [retriever.hybrid(row["question"], top_k=k, candidate_k=30) for row in eval_set]
    reranked = [retriever.rerank(row["question"], retriever.hybrid(row["question"], top_k=20, candidate_k=40), top_k=k) for row in eval_set]
    return {
        "eval_set_size": len(eval_set),
        "note": "Retrieval metrics use the company/document mentioned in questions.json as relevance labels; answer correctness is not claimed.",
        "dense": _metrics(dense, eval_set, k),
        "bm25": _metrics(bm25, eval_set, k),
        "hybrid": _metrics(hybrid, eval_set, k),
        "hybrid_reranker": _metrics(reranked, eval_set, k),
        "sample_eval_items": eval_set[:5],
    }
