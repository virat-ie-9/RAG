# Enterprise Reports Hybrid RAG

End-to-end, CPU-friendly Retrieval-Augmented Generation over a local dataset of corporate annual reports. The system ingests 100 documents (PDF + Markdown) from the read-only `archive/` folder, builds a hybrid index, and answers questions with grounded, cited responses.

---

## Architecture

```
archive/ (read-only)
  └── *.md / *.pdf / *_meta.json / subset.json / questions.json
         │
         ▼
  ┌─────────────────────────────────────────┐
  │  Ingestion & Chunking                   │
  │  • Prefers Markdown over PDF            │
  │  • TOC headings from *_meta.json        │
  │  • Word-window sliding chunks (320/60)  │
  └──────────────┬──────────────────────────┘
                 │  documents.jsonl / chunks.jsonl → data/processed/
                 ▼
  ┌─────────────────────────────────────────┐
  │  Dual Index                             │
  │  • ChromaDB  ← BGE-small-en-v1.5       │
  │  • BM25Okapi ← rank_bm25               │
  └──────────────┬──────────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────────┐
  │  Hybrid Retrieval (RRF)                 │
  │  • Dense search  (top-40)               │
  │  • BM25 lexical  (top-40)               │
  │  • Company-name auto-filter             │
  │  • Reciprocal Rank Fusion  k=60         │
  └──────────────┬──────────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────────┐
  │  CrossEncoder Reranking                 │
  │  ms-marco-MiniLM-L-6-v2 (CPU, cached)  │
  │  → top-8 hits with rerank_score         │
  └──────────────┬──────────────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────────────┐
  │  Answer Generation                      │
  │  Priority: Ollama → OpenAI → Gemini →   │
  │            Grounded extractive fallback │
  │  Always includes inline [N] citations   │
  └──────────────┬──────────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   FastAPI REST       Streamlit UI
   api/main.py     app/streamlit_app.py
```

---

## Quick Start

```bash
# 1. Clone and enter the project
cd /home/virat/Desktop/Project_for_cv/RAG

# 2. Install dependencies (CPU-only torch)
pip install -r requirements.txt

# 3. Configure (copy and edit if you have Ollama/OpenAI/Gemini)
cp .env.example .env

# 4. Build the index (5-doc smoke test takes ~5 min on CPU)
python scripts/build_index.py --reset --max-docs 5

# 5. Ask a question
python scripts/ask.py "Did Downer EDI Limited announce a share buyback plan? If there is no mention, return False." --pretty
```

---

## Full Index Build

Indexes all 100 documents (~90 min on 8-core CPU, ~1 GB RAM):

```bash
python scripts/build_index.py --reset
```

---

## Evaluation

```bash
python scripts/evaluate.py --limit 30 --k 5
```

Reports `hit_rate@k` and `MRR@k` for four retrieval strategies using `questions.json` + `subset.json` as relevance labels. Results are saved to `data/evaluation_results.json`.

Sample output (5-doc index):

| Retriever          | hit_rate@5 | MRR@5 |
|--------------------|------------|-------|
| Dense (BGE-small)  | 1.00       | 1.00  |
| BM25               | 1.00       | 0.875 |
| Hybrid RRF         | 1.00       | 1.00  |
| Hybrid + Reranker  | 1.00       | 1.00  |

---

## API Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

| Method | Endpoint       | Description                                      |
|--------|----------------|--------------------------------------------------|
| GET    | `/health`      | Liveness + index readiness                       |
| GET    | `/dataset`     | Archive file statistics (read-only)              |
| GET    | `/index-stats` | BM25 + Chroma chunk counts, company list         |
| POST   | `/ask`         | Full RAG: retrieve → rerank → generate + cite    |
| POST   | `/evaluate`    | Run retrieval evaluation, return metrics         |

Interactive docs at `http://localhost:8000/docs`.

### POST /ask

```json
{
  "question": "Did ACRES Commercial Realty Corp. outline any new ESG initiatives?",
  "top_k": 6
}
```

Response fields:

| Field                | Type          | Description                               |
|----------------------|---------------|-------------------------------------------|
| `answer`             | string        | Grounded answer with inline [N] citations |
| `confidence`         | string        | `high / medium / low / insufficient`      |
| `citations`          | list          | Source documents with page + section      |
| `retrieval_hits`     | list          | All reranked hits with scores             |
| `reranker_used`      | bool          | Whether CrossEncoder ran                  |
| `num_hits_retrieved` | int           | Pre-rerank candidate count                |

---

## Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`. Enter the FastAPI URL in the sidebar, then type a question. Displays:
- Grounded answer with confidence badge (🟢 high / 🟡 medium / 🟠 low / 🔴 insufficient)
- Source citations with TOC section headings
- Retrieval details table (RRF score, rerank score, retriever label)
- Index stats panel

---

## Tests

```bash
pytest                    # run all 70+ tests
pytest -v tests/test_api.py          # FastAPI endpoint tests
pytest -v tests/test_evaluation.py   # evaluation unit + integration
pytest -v tests/test_pipeline.py     # end-to-end RAG pipeline
pytest -v tests/test_retrieval_utils.py  # BM25, RRF, company extractor
pytest -v tests/test_vector_store.py     # ChromaDB wrapper
pytest -v tests/test_ingest.py           # ingestion + archive read-only
pytest -v tests/test_chunking.py         # chunking mechanics
```

Tests that require the live index are automatically skipped if the index is not built.

---

## LLM Backends

Configure via `.env`. Priority order (first non-empty setting wins):

| Backend         | Setting                        | Notes                              |
|-----------------|--------------------------------|------------------------------------|
| Ollama (local)  | `OLLAMA_MODEL=mistral`         | Free, private, no GPU needed       |
| OpenAI          | `OPENAI_API_KEY=sk-...`        | Uses `OPENAI_MODEL` (default gpt-4o-mini) |
| Gemini          | `GEMINI_API_KEY=AI...`         | Uses `GEMINI_MODEL` (default gemini-2.0-flash) |
| Extractive      | *(no key set)*                 | Always available, cites evidence inline |

---

## Project Structure

```
RAG/
├── archive/               # Read-only source data (PDFs, markdown, metadata)
├── api/
│   └── main.py            # FastAPI application (lifespan warmup, all endpoints)
├── app/
│   └── streamlit_app.py   # Streamlit chat UI
├── data/                  # Generated (gitignored)
│   ├── processed/         # documents.jsonl, chunks.jsonl
│   └── index/             # chroma/, bm25.pkl
├── scripts/
│   ├── ask.py             # CLI: answer a question (--pretty flag)
│   ├── build_index.py     # Build/reset Chroma + BM25 index
│   ├── evaluate.py        # Retrieval evaluation CLI
│   ├── inspect_dataset.py # Archive statistics
│   └── preprocess.py      # Preprocessing-only CLI
├── src/ragcv/
│   ├── bm25.py            # BM25Index with where-filter and save/load
│   ├── chunking.py        # Word-window sliding chunk
│   ├── config.py          # Pydantic settings (all tunable via .env)
│   ├── evaluation.py      # hit_rate@k, MRR@k metrics
│   ├── ingest.py          # PDF/MD ingestion, TOC heading extraction
│   ├── io_utils.py        # JSONL read/write helpers
│   ├── llm.py             # Generation layer (confidence, citations, backends)
│   ├── pipeline.py        # run_ingestion(), make_retriever(), answer_question()
│   ├── retrieval.py       # HybridRetriever (dense + BM25 + RRF + CrossEncoder)
│   ├── schemas.py         # Pydantic models (DocumentPage, Chunk, RetrievalHit, ...)
│   └── vector_store.py    # ChromaVectorStore wrapper
├── tests/
│   ├── test_api.py           # FastAPI TestClient tests
│   ├── test_chunking.py
│   ├── test_evaluation.py
│   ├── test_ingest.py
│   ├── test_pipeline.py      # Full end-to-end integration
│   ├── test_retrieval_utils.py
│   └── test_vector_store.py
├── .env.example           # Template for all settings
├── pytest.ini
└── requirements.txt
```

---

## Notes

- `archive/` is **never modified** — all outputs go to `data/`.
- ChromaDB telemetry warnings (`capture() takes 1 positional argument`) are harmless and can be ignored.
- `USE_TF=0` and `TRANSFORMERS_NO_TF=1` are set automatically in code to avoid TensorFlow import conflicts with `sentence-transformers`.
- Confidence tiers: `high` (CrossEncoder > 5.0), `medium` (> 0.0), `low` (dense only), `insufficient` (no hits).
