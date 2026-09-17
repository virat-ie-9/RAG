import os
import re

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from sentence_transformers import CrossEncoder

from .bm25 import BM25Index
from .schemas import RetrievalHit
from .vector_store import ChromaVectorStore


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievalHit]], k: int = 60, top_k: int = 30
) -> list[RetrievalHit]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Each hit's RRF score = sum over lists of 1 / (k + rank).
    Duplicate chunk_ids across lists are merged (highest-priority hit wins).
    """
    fused: dict[str, tuple[RetrievalHit, float]] = {}
    for results in result_sets:
        for rank, hit in enumerate(results, start=1):
            rrf_score = 1.0 / (k + rank)
            existing_hit, existing_score = fused.get(hit.chunk_id, (hit, 0.0))
            existing_hit.retriever = "hybrid"
            fused[hit.chunk_id] = (existing_hit, existing_score + rrf_score)
    ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)[:top_k]
    output: list[RetrievalHit] = []
    for hit, rrf_score in ranked:
        hit.score = round(float(rrf_score), 6)
        output.append(hit)
    return output


def _extract_company_name(query: str, known_names: list[str]) -> str | None:
    """Return the longest company name mentioned in the query (case-insensitive)."""
    q_lower = query.lower()
    matches = [name for name in known_names if name.lower() in q_lower]
    return max(matches, key=len) if matches else None


class HybridRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        bm25: BM25Index,
        reranker_model: str | None = None,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker: CrossEncoder | None = None
        if reranker_model:
            try:
                self.reranker = CrossEncoder(reranker_model, device="cpu", local_files_only=True)
            except Exception:
                self.reranker = CrossEncoder(reranker_model, device="cpu")

    # ------------------------------------------------------------------ #
    #  Known company names (populated lazily from BM25 chunk metadata)    #
    # ------------------------------------------------------------------ #
    @property
    def _known_companies(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        for chunk in self.bm25.chunks:
            name = chunk.company_name
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        return names

    # ------------------------------------------------------------------ #
    #  Individual retrieval arms                                           #
    # ------------------------------------------------------------------ #
    def dense(self, query: str, top_k: int = 20, where: dict | None = None) -> list[RetrievalHit]:
        return self.vector_store.search(query, top_k=top_k, where=where)

    def lexical(self, query: str, top_k: int = 20, where: dict | None = None) -> list[RetrievalHit]:
        return self.bm25.search(query, top_k=top_k, where=where)

    # ------------------------------------------------------------------ #
    #  Hybrid: RRF over Dense + BM25                                       #
    # ------------------------------------------------------------------ #
    def hybrid(
        self,
        query: str,
        top_k: int = 20,
        candidate_k: int = 40,
        use_company_filter: bool = True,
    ) -> list[RetrievalHit]:
        """Run Dense + BM25 retrieval and fuse with Reciprocal Rank Fusion.

        If `use_company_filter=True` (default), and a known company name is
        detected in the query, dense retrieval is pre-filtered by that company
        to reduce inter-document noise.  BM25 always searches globally so
        that exact term matches from any document can still surface.
        """
        where: dict | None = None
        if use_company_filter:
            company = _extract_company_name(query, self._known_companies)
            if company:
                where = {"company_name": company}

        dense_hits = self.dense(query, top_k=candidate_k, where=where)
        bm25_hits = self.lexical(query, top_k=candidate_k, where=where)
        return reciprocal_rank_fusion([dense_hits, bm25_hits], top_k=top_k)

    # ------------------------------------------------------------------ #
    #  Cross-Encoder re-ranking                                            #
    # ------------------------------------------------------------------ #
    def rerank(
        self, query: str, hits: list[RetrievalHit], top_k: int = 8
    ) -> list[RetrievalHit]:
        """Re-rank candidate hits using a Cross-Encoder and return top_k."""
        if not self.reranker or not hits:
            return hits[:top_k]
        pairs = [(query, hit.text) for hit in hits]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        for hit, score in zip(hits, scores):
            hit.rerank_score = round(float(score), 6)
            hit.retriever = "hybrid+reranker"
        return sorted(
            hits,
            key=lambda h: h.rerank_score if h.rerank_score is not None else h.score,
            reverse=True,
        )[:top_k]

    # ------------------------------------------------------------------ #
    #  Convenience: Dense-only or BM25-only score debug                   #
    # ------------------------------------------------------------------ #
    def compare_retrievers(
        self, query: str, top_k: int = 5
    ) -> dict[str, list[dict]]:
        """Return top-k results from each retriever side-by-side for analysis."""
        def _to_rows(hits: list[RetrievalHit]) -> list[dict]:
            return [
                {
                    "doc_id": h.doc_id[:8],
                    "company": h.company_name,
                    "page": h.page_start,
                    "section": h.section_title,
                    "score": round(h.score, 4),
                    "retriever": h.retriever,
                }
                for h in hits[:top_k]
            ]

        return {
            "dense": _to_rows(self.dense(query, top_k=top_k)),
            "bm25": _to_rows(self.lexical(query, top_k=top_k)),
            "hybrid_rrf": _to_rows(self.hybrid(query, top_k=top_k, use_company_filter=False)),
        }

