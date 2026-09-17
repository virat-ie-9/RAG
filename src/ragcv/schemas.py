from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    doc_id: str
    source_path: str
    page: int
    text: str
    company_name: str | None = None
    currency: str | None = None
    industry: str | None = None
    section_title: str | None = None


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    page_start: int
    page_end: int
    text: str
    company_name: str | None = None
    currency: str | None = None
    industry: str | None = None
    section_title: str | None = None


class RetrievalHit(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_path: str
    doc_id: str
    page_start: int
    page_end: int
    company_name: str | None = None
    currency: str | None = None
    industry: str | None = None
    section_title: str | None = None
    retriever: str = "unknown"
    rerank_score: float | None = None


class SourceCitation(BaseModel):
    source_id: int
    company_name: str | None
    doc_id: str
    source_path: str
    pages: str
    section_title: str | None = None     # TOC heading where this evidence came from
    snippet: str                          # first 350 chars of the chunk


class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: str = Field(description="high, medium, low, or insufficient")
    citations: list[SourceCitation]
    retrieval_hits: list[RetrievalHit]
    reranker_used: bool = False
    num_hits_retrieved: int = 0


# ── Chat history models ───────────────────────────────────────────────────────

class ChatTurn(BaseModel):
    """A single exchange turn in a chat session."""
    role: str                              # "user" or "assistant"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    # Assistant-only fields (None for user turns)
    confidence: str | None = None
    citations: list[SourceCitation] = []
    retriever: str | None = None          # e.g. "hybrid+reranker"


class ChatSession(BaseModel):
    """Persistent chat session holding all turns."""
    session_id: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    turns: list[ChatTurn] = []

    @property
    def turn_count(self) -> int:
        return len(self.turns)


class ChatRequest(BaseModel):
    """Request body for POST /chat."""
    question: str = Field(..., min_length=3, description="User message / question")
    session_id: str | None = Field(None, description="Existing session ID; omit to start a new chat")
    top_k: int = Field(6, ge=1, le=20)


class ChatResponse(BaseModel):
    """Response from POST /chat — includes the full running history."""
    session_id: str
    answer: str
    confidence: str
    citations: list[SourceCitation]
    reranker_used: bool = False
    num_hits_retrieved: int = 0
    history: list[ChatTurn] = []         # full session history including this turn
