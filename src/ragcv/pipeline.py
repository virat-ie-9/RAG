from __future__ import annotations

from pathlib import Path

from .bm25 import BM25Index
from .chunking import chunk_pages
from .config import Settings, get_settings
from .ingest import load_documents
from .io_utils import read_jsonl, write_jsonl
from .llm import generate_answer
from .retrieval import HybridRetriever
from .schemas import AnswerResponse, Chunk, ChatResponse, DocumentPage, SourceCitation
from .vector_store import ChromaVectorStore


def run_ingestion(
    archive_dir: Path | None = None,
    data_dir: Path | None = None,
    chunk_size: int = 320,
    overlap: int = 60,
    max_docs: int = 0,
    prefer_markdown: bool = True,
) -> dict:
    """Ingest documents from archive (strictly read-only) and produce documents.jsonl and chunks.jsonl under data/processed/."""
    settings = get_settings()
    archive_path = archive_dir or settings.archive_dir
    data_path = data_dir or settings.data_dir
    processed_dir = data_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    docs_path = processed_dir / "documents.jsonl"
    chunks_path = processed_dir / "chunks.jsonl"

    pages = load_documents(archive_path, prefer_markdown=prefer_markdown)
    if max_docs > 0:
        keep = set(sorted({p.doc_id for p in pages})[:max_docs])
        pages = [p for p in pages if p.doc_id in keep]

    write_jsonl(docs_path, pages)
    chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    write_jsonl(chunks_path, chunks)

    doc_ids = {p.doc_id for p in pages}
    return {
        "status": "success",
        "archive_dir": str(archive_path),
        "documents_ingested": len(doc_ids),
        "pages_ingested": len(pages),
        "chunks_created": len(chunks),
        "documents_path": str(docs_path),
        "chunks_path": str(chunks_path),
    }


def make_retriever(settings: Settings | None = None, use_reranker: bool = True) -> HybridRetriever:
    settings = settings or get_settings()
    chunks = read_jsonl(settings.chunks_path, Chunk)
    vector = ChromaVectorStore(settings.chroma_dir, settings.chroma_collection, settings.embedding_model)
    bm25 = BM25Index.load(settings.bm25_path) if settings.bm25_path.exists() else BM25Index(chunks)
    return HybridRetriever(vector, bm25, settings.reranker_model if use_reranker else None)


def _build_citations(hits: list, top_k: int) -> list[SourceCitation]:
    """Build SourceCitation list from reranked hits."""
    citations: list[SourceCitation] = []
    for i, hit in enumerate(hits[:top_k], start=1):
        pages = (
            str(hit.page_start)
            if hit.page_start == hit.page_end
            else f"{hit.page_start}-{hit.page_end}"
        )
        citations.append(
            SourceCitation(
                source_id=i,
                company_name=hit.company_name,
                doc_id=hit.doc_id,
                source_path=hit.source_path,
                pages=pages,
                section_title=hit.section_title,
                snippet=hit.text[:350],
            )
        )
    return citations


def answer_question(
    question: str,
    settings: Settings | None = None,
    top_k: int = 8,
    history: str = "",
) -> AnswerResponse:
    """Single-turn RAG answer (no session management).

    Pass `history` (from chat.build_history_context) for follow-up context.
    """
    settings = settings or get_settings()
    retriever = make_retriever(settings, use_reranker=True)
    reranker_available = retriever.reranker is not None

    candidates = retriever.hybrid(question, top_k=settings.candidate_k, candidate_k=settings.candidate_k)
    hits = retriever.rerank(question, candidates, top_k=settings.reranker_top_k)
    answer, confidence = generate_answer(question, hits, settings, history=history)
    citations = _build_citations(hits, top_k)

    return AnswerResponse(
        question=question,
        answer=answer,
        confidence=confidence,
        citations=citations,
        retrieval_hits=hits,
        reranker_used=reranker_available,
        num_hits_retrieved=len(candidates),
    )


def chat_answer(
    question: str,
    session_id: str | None,
    settings: Settings | None = None,
    top_k: int = 8,
) -> ChatResponse:
    """Multi-turn RAG answer with session persistence.

    Loads the session, builds history context, generates a grounded answer,
    appends the turn pair, saves the session, and returns a ChatResponse.
    """
    from .chat import (
        assistant_turn,
        build_history_context,
        load_session,
        new_session_id,
        save_session,
        user_turn,
    )

    settings = settings or get_settings()
    history_dir = settings.data_dir / "chat_history"

    # Resolve or create session
    sid = session_id or new_session_id()
    session = load_session(history_dir, sid)

    # Build history context from prior turns (last 3 Q&A pairs)
    history_ctx = build_history_context(session, n=3)

    # Retrieve + rerank + generate
    retriever = make_retriever(settings, use_reranker=True)
    reranker_available = retriever.reranker is not None

    candidates = retriever.hybrid(question, top_k=settings.candidate_k, candidate_k=settings.candidate_k)
    hits = retriever.rerank(question, candidates, top_k=settings.reranker_top_k)
    answer, confidence = generate_answer(question, hits, settings, history=history_ctx)
    citations = _build_citations(hits, top_k)

    # Detect retriever label from hits
    retriever_label = hits[0].retriever if hits else "unknown"

    # Persist the new turn pair
    session.turns.append(user_turn(question))
    session.turns.append(
        assistant_turn(answer, confidence, citations, retriever=retriever_label)
    )
    save_session(history_dir, session)

    return ChatResponse(
        session_id=sid,
        answer=answer,
        confidence=confidence,
        citations=citations,
        reranker_used=reranker_available,
        num_hits_retrieved=len(candidates),
        history=session.turns,
    )
