import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragcv.config import get_settings
from ragcv.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and preprocess enterprise reports into chunks (archive is read-only).")
    parser.add_argument("--chunk-size", type=int, default=320, help="Word chunk size (default: 320)")
    parser.add_argument("--overlap", type=int, default=60, help="Word overlap (default: 60)")
    parser.add_argument("--max-docs", type=int, default=0, help="Optional document limit for smoke testing (0 for all)")
    args = parser.parse_args()

    settings = get_settings()
    print(f"Starting ingestion from '{settings.archive_dir}' (read-only)...")
    start = time.perf_counter()
    summary = run_ingestion(
        archive_dir=settings.archive_dir,
        data_dir=settings.data_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_docs=args.max_docs,
    )
    elapsed = time.perf_counter() - start
    summary["elapsed_seconds"] = round(elapsed, 2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
