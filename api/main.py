"""FastAPI application for the Enterprise Reports Hybrid RAG system.

Endpoints:
  GET  /health        — liveness + index readiness check
  GET  /dataset       — archive statistics
  GET  /index-stats   — BM25 + ChromaDB chunk counts
  POST /ask           — full RAG pipeline: retrieve → rerank → generate
  POST /evaluate      — run retrieval evaluation and return metrics
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragcv.config import get_settings
from ragcv.ingest import inspect_archive


# ─── Lifespan: warm up retriever on startup ───────────────────────────────────
_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the retriever (embedder + CrossEncoder + BM25) at startup
    so the first /ask call is not slow."""
    global _retriever
    settings = get_settings()
    if settings.chunks_path.exists() and settings.bm25_path.exists():
        from ragcv.pipeline import make_retriever
        _retriever = make_retriever(settings, use_reranker=True)
    yield
    _retriever = None


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise Reports Hybrid RAG",
    version="2.0.0",
    description=(
        "CPU-friendly hybrid RAG over annual reports. "
        "Dense (BGE-small) + BM25 + RRF + CrossEncoder reranking + "
        "grounded extractive / Ollama / OpenAI / Gemini answer generation."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural-language question")
    top_k: int = Field(6, ge=1, le=20, description="Number of source citations to return")


class EvaluateRequest(BaseModel):
    limit: int = Field(10, ge=1, le=100, description="Number of eval questions to run")
    k: int = Field(5, ge=1, le=20, description="Hit-rate / MRR cutoff")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Liveness and readiness probe."""
    settings = get_settings()
    return {
        "status": "ok",
        "archive_exists": settings.archive_dir.exists(),
        "chunks_exists": settings.chunks_path.exists(),
        "bm25_exists": settings.bm25_path.exists(),
        "chroma_dir_exists": settings.chroma_dir.exists(),
        "retriever_warmed_up": _retriever is not None,
    }


@app.get("/dataset", tags=["System"])
def dataset():
    """Return archive file statistics (read-only)."""
    return inspect_archive(get_settings().archive_dir)


@app.get("/index-stats", tags=["System"])
def index_stats():
    """Return chunk counts in BM25 and ChromaDB indexes."""
    settings = get_settings()
    if not settings.bm25_path.exists():
        raise HTTPException(status_code=503, detail="Index not built — run build_index.py first.")
    from ragcv.pipeline import make_retriever
    r = _retriever or make_retriever(settings, use_reranker=False)
    return {
        "bm25_chunks": r.bm25.count(),
        "chroma_chunks": r.vector_store.count(),
        "companies": list({c.company_name for c in r.bm25.chunks if c.company_name}),
        "embedding_model": r.vector_store.embedding_model,
        "reranker_model": settings.reranker_model,
    }


@app.post("/ask", tags=["RAG"])
def ask(request: AskRequest):
    """Full RAG pipeline: hybrid retrieval → CrossEncoder reranking → grounded answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    settings = get_settings()
    if not settings.chunks_path.exists():
        raise HTTPException(status_code=503, detail="Index not built — run build_index.py first.")
    try:
        from ragcv.pipeline import answer_question
        return answer_question(request.question, settings, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Chat Endpoints ───────────────────────────────────────────────────────────

from ragcv.schemas import ChatRequest

@app.post("/chat", tags=["Chat"])
def chat(request: ChatRequest):
    """Multi-turn chat endpoint."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    settings = get_settings()
    if not settings.chunks_path.exists():
        raise HTTPException(status_code=503, detail="Index not built — run build_index.py first.")
    try:
        from ragcv.pipeline import chat_answer
        return chat_answer(
            question=request.question,
            session_id=request.session_id,
            settings=settings,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/sessions", tags=["Chat"])
def list_all_sessions():
    """List all chat sessions."""
    from ragcv.chat import list_sessions
    settings = get_settings()
    history_dir = settings.data_dir / "chat_history"
    return {"sessions": list_sessions(history_dir)}


@app.get("/chat/{session_id}", tags=["Chat"])
def get_session(session_id: str):
    """Get the full history of a specific chat session."""
    from ragcv.chat import load_session
    settings = get_settings()
    history_dir = settings.data_dir / "chat_history"
    session = load_session(history_dir, session_id)
    return session


@app.delete("/chat/{session_id}", tags=["Chat"])
def delete_chat_session(session_id: str):
    """Delete a chat session."""
    from ragcv.chat import delete_session
    settings = get_settings()
    history_dir = settings.data_dir / "chat_history"
    if delete_session(history_dir, session_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/evaluate", tags=["Evaluation"])
def evaluate(request: EvaluateRequest):
    """Run retrieval evaluation and return hit_rate@k and MRR@k metrics."""
    settings = get_settings()
    if not settings.chunks_path.exists():
        raise HTTPException(status_code=503, detail="Index not built — run build_index.py first.")
    try:
        from ragcv.evaluation import evaluate_retrieval
        from ragcv.pipeline import make_retriever
        r = _retriever or make_retriever(settings, use_reranker=True)
        return evaluate_retrieval(r, settings, limit=request.limit, k=request.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
