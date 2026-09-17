import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragcv.config import get_settings
from ragcv.ingest import inspect_archive, load_documents


def main() -> None:
    settings = get_settings()
    summary = inspect_archive(settings.archive_dir)
    pages = load_documents(settings.archive_dir)
    summary["ingested_pages_with_text"] = len(pages)
    summary["documents_with_text"] = len({p.doc_id for p in pages})
    summary["sample_pages"] = [p.model_dump(exclude={"text"}) | {"chars": len(p.text)} for p in pages[:5]]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
