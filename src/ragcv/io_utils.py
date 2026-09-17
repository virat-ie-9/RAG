import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")
            count += 1
    return count


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    rows: list[T] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(model.model_validate_json(line))
    return rows


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
