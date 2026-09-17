import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragcv.bm25 import BM25Index
from ragcv.chunking import chunk_pages
from ragcv.config import get_settings
from ragcv.ingest import inspect_archive, load_documents
from ragcv.io_utils import write_jsonl
from ragcv.vector_store import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=320)
    parser.add_argument("--overlap", type=int, default=60)
    parser.add_argument("--max-docs", type=int, default=0, help="Optional smoke-test limit by document count.")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true", help="Only run document loading and chunking without building vector index.")
    args = parser.parse_args()

    settings = get_settings()
    print(json.dumps(inspect_archive(settings.archive_dir), indent=2))
    pages = load_documents(settings.archive_dir)
    if args.max_docs:
        keep = {doc_id for doc_id in sorted({p.doc_id for p in pages})[: args.max_docs]}
        pages = [p for p in pages if p.doc_id in keep]
    doc_count = len({p.doc_id for p in pages})
    print(f"Loaded {len(pages)} pages from {doc_count} documents")
    write_jsonl(settings.documents_path, pages)
    chunks = chunk_pages(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    write_jsonl(settings.chunks_path, chunks)
    print(f"Created {len(chunks)} chunks")
    BM25Index(chunks).save(settings.bm25_path)
    print(f"Saved BM25 index to {settings.bm25_path}")
    if args.preprocess_only:
        print("Preprocessing and BM25 index complete (--preprocess-only specified).")
        return
    vector = ChromaVectorStore(settings.chroma_dir, settings.chroma_collection, settings.embedding_model)
    vector.build(chunks, reset=args.reset)
    print(f"Saved Chroma collection '{settings.chroma_collection}' to {settings.chroma_dir}")


if __name__ == "__main__":
    main()
