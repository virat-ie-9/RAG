import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragcv.config import get_settings
from ragcv.evaluation import evaluate_retrieval
from ragcv.pipeline import make_retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    retriever = make_retriever(settings, use_reranker=not args.no_reranker)
    results = evaluate_retrieval(retriever, settings, limit=args.limit, k=args.k)
    out_path = settings.data_dir / "evaluation_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
