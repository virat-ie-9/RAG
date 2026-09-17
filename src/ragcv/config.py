from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    archive_dir: Path = Path("archive")
    data_dir: Path = Path("data")

    # ── Embedding & retrieval ─────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chroma_collection: str = "enterprise_reports"

    # ── Generation tuning ────────────────────────────────────────────────────
    max_context_chars: int = 7000   # max chars fed into the LLM context window
    reranker_top_k: int = 8         # final hits kept after reranking
    candidate_k: int = 40           # candidates sent to reranker

    # ── LLM backends (first non-empty wins) ──────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"   # free-tier fast model

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def chunks_path(self) -> Path:
        return self.processed_dir / "chunks.jsonl"

    @property
    def documents_path(self) -> Path:
        return self.processed_dir / "documents.jsonl"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.pkl"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    return settings
