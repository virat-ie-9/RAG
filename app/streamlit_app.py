"""Streamlit UI for the Enterprise Reports Hybrid RAG system.

Connects to the FastAPI backend at API_URL (configurable in sidebar).
Displays grounded answers, source citations with section headings,
and maintains full conversational history.
"""
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ── Page config MUST be the first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="Enterprise Reports RAG",
    page_icon="📋",
    layout="wide",
)

# ── Session State Init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def clear_chat():
    st.session_state.session_id = None
    st.session_state.messages = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    API_URL = st.text_input("FastAPI URL", value="http://localhost:8000")
    top_k = st.slider("Max citations", 3, 10, 6)

    st.button("➕ New Chat", on_click=clear_chat, use_container_width=True)
    st.divider()

    if st.button("🔍 Check backend health"):
        try:
            health = requests.get(f"{API_URL}/health", timeout=5).json()
            if health.get("status") == "ok":
                st.success("Backend OK")
            else:
                st.warning("Backend reachable but not fully ready")
            st.json(health)
        except Exception as exc:
            st.error(f"Cannot reach backend: {exc}")

    if st.button("📊 Index stats"):
        try:
            stats = requests.get(f"{API_URL}/index-stats", timeout=10).json()
            st.metric("BM25 chunks", stats.get("bm25_chunks", "–"))
            st.metric("Chroma chunks", stats.get("chroma_chunks", "–"))
            companies = stats.get("companies", [])
            st.caption(f"{len(companies)} companies indexed")
            if companies:
                st.write(sorted(companies))
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.caption(
        "Pipeline: BGE-small (dense) + BM25 → RRF → "
        "ms-marco CrossEncoder → grounded extractive / LLM answer"
    )

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("📋 Enterprise Reports RAG")
st.caption("Ask questions about annual reports. Answers are grounded in retrieved document chunks with source citations.")

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If it's an assistant message, show citations and metadata if present
        if msg["role"] == "assistant":
            conf = msg.get("confidence")
            conf_color = {"high": "🟢", "medium": "🟡", "low": "🟠", "insufficient": "🔴"}.get(conf, "⚪") if conf else "⚪"
            st.caption(f"{conf_color} Confidence: **{conf}**")
            
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📚 Sources ({len(citations)})"):
                    for c in citations:
                        company = c.get("company_name") or c.get("doc_id", "")[:10]
                        pages = c.get("pages", "")
                        section = c.get("section_title", "")
                        label = f"[{c['source_id']}] {company} | p{pages}"
                        if section:
                            label += f" § {section}"
                        st.markdown(f"**{label}**")
                        st.write(c.get("snippet", ""))
                        st.code(c.get("source_path", ""), language=None)

# Accept user input
if prompt := st.chat_input("Ask a question... (e.g. Did Downer EDI announce a share buyback?)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Retrieving and grounding answer…"):
            try:
                payload: dict[str, Any] = {"question": prompt, "top_k": top_k}
                if st.session_state.session_id:
                    payload["session_id"] = st.session_state.session_id
                
                response = requests.post(
                    f"{API_URL}/chat",
                    json=payload,
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
            except requests.HTTPError as exc:
                detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
                st.error(f"❌ API error: {detail}")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Request failed: {exc}")
                st.stop()

        # Update session ID if it was newly created
        st.session_state.session_id = data.get("session_id")
        
        answer_text = data.get("answer", "")
        confidence = data.get("confidence", "unknown")
        citations = data.get("citations", [])
        
        # Display the result
        message_placeholder.markdown(answer_text)
        
        conf_color = {"high": "🟢", "medium": "🟡", "low": "🟠", "insufficient": "🔴"}.get(confidence, "⚪")
        st.caption(f"{conf_color} Confidence: **{confidence}**")
        
        if citations:
            with st.expander(f"📚 Sources ({len(citations)})"):
                for c in citations:
                    company = c.get("company_name") or c.get("doc_id", "")[:10]
                    pages = c.get("pages", "")
                    section = c.get("section_title", "")
                    label = f"[{c['source_id']}] {company} | p{pages}"
                    if section:
                        label += f" § {section}"
                    st.markdown(f"**{label}**")
                    st.write(c.get("snippet", ""))
                    st.code(c.get("source_path", ""), language=None)
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "confidence": confidence,
            "citations": citations,
        })
