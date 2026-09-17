"""Chat session management for multi-turn conversational RAG.

Sessions are stored as JSON files under data/chat_history/<session_id>.json.
No database is required.

Public API
----------
new_session_id()                       → str
load_session(history_dir, session_id)  → ChatSession
save_session(history_dir, session)     → None
build_history_context(session, n)      → str   (for LLM prompt injection)
list_sessions(history_dir)             → list[dict]
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from .schemas import ChatSession, ChatTurn, SourceCitation


# ─── Session ID ───────────────────────────────────────────────────────────────

def new_session_id() -> str:
    """Generate a new unique session ID."""
    return str(uuid.uuid4())


# ─── Persistence ──────────────────────────────────────────────────────────────

def _session_path(history_dir: Path, session_id: str) -> Path:
    return history_dir / f"{session_id}.json"


def load_session(history_dir: Path, session_id: str) -> ChatSession:
    """Load an existing session from disk, or create a blank one if not found."""
    path = _session_path(history_dir, session_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return ChatSession.model_validate(data)
    # Brand-new session
    return ChatSession(
        session_id=session_id,
        created_at=datetime.utcnow().isoformat(),
        turns=[],
    )


def save_session(history_dir: Path, session: ChatSession) -> None:
    """Persist a session to disk (atomic write via temp file)."""
    history_dir.mkdir(parents=True, exist_ok=True)
    path = _session_path(history_dir, session.session_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def delete_session(history_dir: Path, session_id: str) -> bool:
    """Delete a session file. Returns True if it existed."""
    path = _session_path(history_dir, session_id)
    if path.exists():
        path.unlink()
        return True
    return False


def list_sessions(history_dir: Path) -> list[dict]:
    """Return a summary list of all stored sessions."""
    if not history_dir.exists():
        return []
    sessions = []
    for p in sorted(history_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            turns = data.get("turns", [])
            # Find the first user question as a preview
            preview = next(
                (t["content"][:80] for t in turns if t.get("role") == "user"), ""
            )
            sessions.append({
                "session_id": data.get("session_id", p.stem),
                "created_at": data.get("created_at", ""),
                "turn_count": len(turns),
                "preview": preview,
            })
        except Exception:
            continue
    return sessions


# ─── History context for LLM prompt ──────────────────────────────────────────

def build_history_context(session: ChatSession, n: int = 3) -> str:
    """Format the last n Q&A pairs into a text block for the LLM prompt.

    The block looks like:
        [Turn 1]
        User: Did Downer EDI announce a buyback?
        Assistant: False. No evidence found in the retrieved sources.

        [Turn 2]
        User: What about share repurchases?
        Assistant: ...

    Returns an empty string if there are no prior turns.
    """
    if not session.turns:
        return ""

    # Pair turns: user immediately followed by assistant
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(session.turns) - 1:
        if session.turns[i].role == "user" and session.turns[i + 1].role == "assistant":
            pairs.append((session.turns[i].content, session.turns[i + 1].content))
            i += 2
        else:
            i += 1

    # Take the last n pairs
    recent = pairs[-n:] if len(pairs) >= n else pairs
    if not recent:
        return ""

    lines: list[str] = ["Previous conversation:"]
    for idx, (user_msg, assistant_msg) in enumerate(recent, start=1):
        lines.append(f"\n[Turn {idx}]")
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")
    return "\n".join(lines)


# ─── Turn builders ────────────────────────────────────────────────────────────

def user_turn(content: str) -> ChatTurn:
    return ChatTurn(role="user", content=content)


def assistant_turn(
    content: str,
    confidence: str,
    citations: list[SourceCitation],
    retriever: str | None = None,
) -> ChatTurn:
    return ChatTurn(
        role="assistant",
        content=content,
        confidence=confidence,
        citations=citations,
        retriever=retriever,
    )
