import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .schemas import Chunk, RetrievalHit


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9$%.\-]+", text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokens = [tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(self.tokens) if self.tokens else None

    def count(self) -> int:
        return len(self.chunks)

    def search(self, query: str, top_k: int = 20, where: dict | None = None) -> list[RetrievalHit]:
        if not self.index or not self.chunks:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.index.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        hits: list[RetrievalHit] = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            # Apply metadata filter first
            if where:
                if not all(getattr(chunk, k, None) == v for k, v in where.items()):
                    continue
            # When filtering is active, include results even if BM25 score is
            # non-positive (tiny corpus causes negative IDF). Without a filter
            # skip zero-scored chunks to avoid noise.
            if score <= 0 and not where:
                continue
            hits.append(RetrievalHit(**chunk.model_dump(), score=float(score), retriever="bm25"))
            if len(hits) >= top_k:
                break
        return hits

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "BM25Index":
        with path.open("rb") as f:
            return pickle.load(f)
