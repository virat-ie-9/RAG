import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragcv.config import get_settings
from ragcv.pipeline import answer_question


def _pretty_print(resp) -> None:
    print(f"\n{'─'*70}")
    print(f"Q: {resp.question}")
    print(f"{'─'*70}")
    print(f"\nAnswer ({resp.confidence} confidence):\n  {resp.answer}\n")
    reranker = "yes" if resp.reranker_used else "no"
    print(f"Retrieved {resp.num_hits_retrieved} candidates → reranked: {reranker} → {len(resp.citations)} citations\n")
    if resp.citations:
        print("Citations:")
        for c in resp.citations:
            section = f" § {c.section_title}" if c.section_title else ""
            print(f"  [{c.source_id}] {c.company_name or 'Unknown'} | p{c.pages}{section}")
            print(f"      {c.snippet[:120].strip()}...")
    print(f"{'─'*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the RAG index")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=6, help="Final citations to show")
    parser.add_argument("--pretty", action="store_true", help="Human-readable output instead of JSON")
    args = parser.parse_args()

    response = answer_question(args.question, get_settings(), top_k=args.top_k)

    if args.pretty:
        _pretty_print(response)
    else:
        print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    main()
