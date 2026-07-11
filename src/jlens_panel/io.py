"""Small, crash-safe JSON and JSONL helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects, rejecting blank or non-object records."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {source}:{line_number}")
            yield value


def append_jsonl(path: str | Path, value: dict[str, Any]) -> None:
    """Append and fsync one complete JSONL record for resumable runs."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with output.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def completed_keys(path: str | Path, fields: Iterable[str]) -> set[tuple[Any, ...]]:
    """Read composite resume keys from an existing JSONL output."""

    output = Path(path)
    if not output.exists():
        return set()
    names = tuple(fields)
    return {tuple(record[name] for name in names) for record in read_jsonl(output)}
