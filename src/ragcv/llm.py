"""LLM generation layer for the RAG pipeline.

Priority order for answer backends:
  1. Ollama (local, if ollama_model is set and server is reachable)
  2. OpenAI (if openai_api_key is set)
  3. Gemini (if gemini_api_key is set)
  4. Grounded extractive fallback (always available, no network needed)

All backends receive the same structured context block with numbered citations
and are instructed to produce inline [N] references so citations are traceable.
"""
from __future__ import annotations

import re
from collections import Counter

import requests

from .config import Settings
from .schemas import RetrievalHit


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a financial document analyst. Answer ONLY using the provided context \
excerpts. Each excerpt is labelled [N].

Rules:
- Cite every claim with [N] immediately after the relevant sentence.
- For yes/no questions: answer True or False, then explain with a citation.
- If the context lacks enough evidence, say exactly:
  "I do not have enough evidence in the retrieved sources to answer that."
- Be concise (4 sentences max). Do not speculate beyond the sources.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Context builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_block(
    hits: list[RetrievalHit], max_chars: int = 7000
) -> tuple[str, list[RetrievalHit]]:
    """Format hits into a numbered context block for the LLM prompt."""
    parts: list[str] = []
    used: list[RetrievalHit] = []
    total = 0
    for i, hit in enumerate(hits, start=1):
        label = hit.company_name or hit.doc_id[:10]
        pages = (
            str(hit.page_start)
            if hit.page_start == hit.page_end
            else f"{hit.page_start}-{hit.page_end}"
        )
        section = f" § {hit.section_title}" if hit.section_title else ""
        block = f"[{i}] {label} | p{pages}{section}\n{hit.text.strip()}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        used.append(hit)
        total += len(block)
    return "\n".join(parts), used


# Keep the old name as an alias so existing callers don't break
def build_context(
    hits: list[RetrievalHit], max_chars: int = 7000
) -> tuple[str, list[RetrievalHit]]:
    return _build_context_block(hits, max_chars)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_confidence(hits: list[RetrievalHit]) -> str:
    """Map rerank/dense scores → confidence tier.

    Tiers (in order):
    - high       : top rerank_score > 5.0
    - medium     : top rerank_score > 0.0  OR dense cosine > 0.75
    - low        : reranker not used and dense score ≤ 0.75
    - insufficient: no hits
    """
    if not hits:
        return "insufficient"
    rerank_scores = [h.rerank_score for h in hits if h.rerank_score is not None]
    if rerank_scores:
        top = max(rerank_scores)
        if top > 5.0:
            return "high"
        if top > 0.0:
            return "medium"
        return "low"
    # No reranker — use dense cosine similarity
    top_dense = max(h.score for h in hits)
    return "medium" if top_dense > 0.75 else "low"


# ─────────────────────────────────────────────────────────────────────────────
# Grounded extractive fallback
# ─────────────────────────────────────────────────────────────────────────────

_BOOLEAN_EVIDENCE_MAP: dict[str, list[str]] = {
    "share buyback": ["buyback", "buy-back", "repurchase", "repurchased", "on-market"],
    "mergers or acquisitions": ["merger", "acquisition", "acquired", "divest", "disposed"],
    "dividend policy": ["dividend policy", "dividends are determined", "dividend"],
    "new product launches": ["new product", "launch", "launched", "introduced"],
    "esg initiatives": ["esg", "sustainability", "environmental", "social", "governance"],
    "new debt": ["issued", "note", "bond", "debenture", "borrowing", "credit facility"],
    "employee layoffs": ["layoff", "redundanc", "retrench", "workforce reduction"],
}


def _detect_boolean_intent(question: str) -> tuple[bool, list[str]]:
    q_lower = question.lower()
    is_boolean = (
        "if there is no mention, return false" in q_lower
        or q_lower.strip().startswith("did ")
        or q_lower.strip().startswith("does ")
        or q_lower.strip().startswith("has ")
        or q_lower.strip().startswith("is ")
    )
    keywords: list[str] = []
    for topic, terms in _BOOLEAN_EVIDENCE_MAP.items():
        if topic in q_lower:
            keywords = terms
            break
    return is_boolean, keywords


def _best_sentences(
    hits: list[RetrievalHit],
    q_terms: set[str],
    n: int = 3,
) -> list[tuple[int, str]]:
    """Return top-n (citation_id, sentence) pairs that overlap most with query terms."""
    candidates: list[tuple[int, int, str]] = []
    for idx, hit in enumerate(hits[:6], start=1):
        for sent in re.split(r"(?<=[.!?])\s+", hit.text):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            terms = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", sent.lower())
            overlap = sum((Counter(terms) & Counter(q_terms)).values())
            if overlap > 0:
                candidates.append((overlap, idx, sent))

    candidates.sort(key=lambda x: (x[0], len(x[2])), reverse=True)
    seen: set[str] = set()
    result: list[tuple[int, str]] = []
    for _, idx, sent in candidates:
        if sent not in seen:
            seen.add(sent)
            result.append((idx, sent))
        if len(result) >= n:
            break
    return result


def _grounded_extractive_answer(question: str, hits: list[RetrievalHit]) -> str:
    """Produce a grounded answer with inline [N] citations — no LLM required."""
    if not hits:
        return "I do not have enough evidence in the retrieved sources to answer that."

    q_lower = question.lower()
    is_boolean, evidence_keywords = _detect_boolean_intent(question)

    # ── Boolean path ─────────────────────────────────────────────────────────
    if is_boolean and evidence_keywords:
        found: list[tuple[int, str]] = []
        for idx, hit in enumerate(hits[:6], start=1):
            text_lower = hit.text.lower()
            for kw in evidence_keywords:
                if kw in text_lower:
                    for sent in re.split(r"(?<=[.!?])\s+", hit.text):
                        if kw in sent.lower() and len(sent.strip()) > 20:
                            found.append((idx, sent.strip()))
                            break
                    break  # one match per hit

        if found:
            evidence_str = "  ".join(f"{s} [{i}]" for i, s in found[:2])
            return f"True. {evidence_str}"
        return (
            "False. The retrieved sources do not contain evidence of "
            f"'{evidence_keywords[0]}' in the annual report."
        )

    # ── Factual path ─────────────────────────────────────────────────────────
    q_terms = {
        t.lower()
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", question)
        if len(t) > 3
    }
    best = _best_sentences(hits, q_terms, n=3)
    if not best:
        return "I do not have enough evidence in the retrieved sources to answer that."
    return " ".join(f"{sent} [{idx}]" for idx, sent in best)


# ─────────────────────────────────────────────────────────────────────────────
# LLM backends
# ─────────────────────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, settings: Settings) -> str:
    resp = requests.post(
        f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _call_openai(context: str, question: str, settings: Settings) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.openai_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            "temperature": 0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_gemini(context: str, question: str, settings: Settings) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    )
    user_text = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": user_text}]}]},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _call_groq(context: str, question: str, settings: Settings) -> str:
    """Call Groq's OpenAI-compatible chat endpoint (free tier, very fast)."""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.groq_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            "temperature": 0,
            "max_tokens": 512,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _user_content(history: str, context: str, question: str) -> str:
    """Build the user-turn content for chat-style LLM APIs."""
    parts: list[str] = []
    if history:
        parts.append(history)
        parts.append("")  # blank line separator
    parts.append(f"Current sources:\n{context}")
    parts.append(f"\nQuestion: {question}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    hits: list[RetrievalHit],
    settings: Settings,
    history: str = "",
) -> tuple[str, str]:
    """Generate a grounded answer from retrieval hits.

    Args:
        question:  The current user question.
        hits:      Retrieval + reranked hits providing evidence.
        settings:  App settings (selects backend, max_context_chars, etc.)
        history:   Optional formatted prior conversation (from chat.build_history_context).
                   Injected into the LLM prompt so follow-up questions resolve correctly.

    Returns:
        (answer_text, confidence)  where confidence ∈ {high, medium, low, insufficient}

    Backend priority: Ollama → OpenAI → Gemini → Groq → Extractive fallback
    """
    confidence = estimate_confidence(hits)
    context, used_hits = _build_context_block(hits, max_chars=settings.max_context_chars)

    if not used_hits:
        return "I do not have enough evidence in the retrieved sources to answer that.", "insufficient"

    user_msg = _user_content(history, context, question)

    # ── Ollama ────────────────────────────────────────────────────────────────
    if settings.ollama_model:
        try:
            hist_prefix = f"{history}\n\n" if history else ""
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"{hist_prefix}"
                f"Current sources:\n{context}\n\n"
                f"Question: {question}\nAnswer:"
            )
            resp = requests.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/generate",
                json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip(), confidence
        except Exception:
            pass

    # ── OpenAI ────────────────────────────────────────────────────────────────
    if settings.openai_api_key:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.openai_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip(), confidence
        except Exception:
            pass

    # ── Gemini ────────────────────────────────────────────────────────────────
    if settings.gemini_api_key:
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
            )
            full_text = f"{SYSTEM_PROMPT}\n\n{user_msg}"
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": full_text}]}]},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip(), confidence
        except Exception:
            pass

    # ── Groq (free-tier, fast llama) ─────────────────────────────────────────
    if settings.groq_api_key:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.groq_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                    "max_tokens": 512,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip(), confidence
        except Exception:
            pass

    # ── Extractive fallback ───────────────────────────────────────────────────
    answer = _grounded_extractive_answer(question, used_hits)
    if confidence == "high":
        confidence = "medium"
    return answer, confidence
