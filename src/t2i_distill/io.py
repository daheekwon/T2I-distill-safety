from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(value)
    return rows


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key, "")) for key in fieldnames})
            count += 1
    return count


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stable_hash(*parts: Any, length: int = 16) -> str:
    joined = "\u241f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


def slugify(value: str, max_length: int = 80) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9가-힣._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_.")
    if not value:
        value = "item"
    return value[:max_length]


def parse_literal_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, set):
        return sorted(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return sorted(str(item).strip() for item in parsed if str(item).strip())
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def serialize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def load_cell_json(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
