"""JSON/JSONL persistence helpers with atomic writes."""

from __future__ import annotations

import json
import os
import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping


def safe_artifact_name(value: str, limit: int = 96) -> str:
    """Make an untrusted dataset id safe for use as one path component."""
    raw = str(value or "")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    if not cleaned:
        cleaned = "item"
    if cleaned != raw or raw in {".", ".."} or len(cleaned) > limit:
        suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
        cleaned = f"{cleaned[: max(1, limit - 11)]}_{suffix}"
    return cleaned


def read_records(path: str | Path) -> Iterator[Dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{source}:{line_number} is not a JSON object")
                yield value
        return
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("data") or value.get("records") or [value]
    if not isinstance(value, list):
        raise ValueError(f"{source} must contain a JSON list or JSONL objects")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{source}[{index}] is not a JSON object")
        yield item


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: str | Path, value: Any) -> None:
    _atomic_text(Path(path), json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    _atomic_text(Path(path), content)


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, separators=(",", ":")) + "\n")
