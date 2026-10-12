"""JSONL helpers shared by benchmark launchers and analysis."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("recorded_at", utc_now_iso())
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {source}:{line_number}") from error
    return records


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("cannot write a CSV without rows")
    fieldnames = sorted({key for row in materialized for key in row})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
