from pathlib import Path
import os

import chromadb

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .schemas import Chunk, RetrievalHit


def load_embedder(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except TypeError:
        return SentenceTransformer(model_name, device="cpu")
    except Exception:
        return SentenceTransformer(model_name, device="cpu")


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, embedding_model: str):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedder = load_embedder(embedding_model)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.collection.count()

    def build(self, chunks: list[Chunk], batch_size: int = 64, reset: bool = False) -> None:
        if reset:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(self.collection_name, metadata={"hnsw:space": "cosine"})
        if not chunks:
            return
        for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "source_path": c.source_path,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "company_name": c.company_name or "",
                        "currency": c.currency or "",
                        "industry": c.industry or "",
                        "section_title": c.section_title or "",
                    }
                    for c in batch
                ],
            )

    def search(self, query: str, top_k: int = 20, where: dict | None = None) -> list[RetrievalHit]:
        total = self.collection.count()
        if total == 0:
            return []
        n_results = min(top_k, total)
        query_text = (
            f"Represent this sentence for searching relevant passages: {query}"
            if "bge" in self.embedding_model.lower()
            else query
        )
        embedding = self.embedder.encode([query_text], normalize_embeddings=True, show_progress_bar=False).tolist()[0]
        query_kwargs = {
            "query_embeddings": [embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where
        result = self.collection.query(**query_kwargs)
        hits: list[RetrievalHit] = []
        if not result or not result["ids"] or not result["ids"][0]:
            return hits
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    text=text,
                    score=1.0 - float(distance),
                    source_path=meta["source_path"],
                    doc_id=meta["doc_id"],
                    page_start=int(meta["page_start"]),
                    page_end=int(meta["page_end"]),
                    company_name=meta.get("company_name") or None,
                    currency=meta.get("currency") or None,
                    industry=meta.get("industry") or None,
                    section_title=meta.get("section_title") or None,
                    retriever="dense",
                )
            )
        return hits
