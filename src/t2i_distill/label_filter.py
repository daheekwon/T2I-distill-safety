from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from .execution import required_signal_columns
from .io import write_csv, write_json


def filter_labels_by_existing_images(
    labels_path: Path,
    output_path: Path,
    *,
    summary_path: Path | None = None,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    """Write labels whose image_path currently exists on disk."""

    rows_out: list[dict[str, Any]] = []
    by_model: Counter[str] = Counter()
    missing_by_model: Counter[str] = Counter()
    empty_by_model: Counter[str] = Counter()
    rows_seen = 0
    fieldnames: list[str] = []

    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows_seen += 1
            model_id = str(row.get("model_id", ""))
            image_path = Path(str(row.get("image_path", "")))
            if not image_path.exists():
                missing_by_model[model_id] += 1
                continue
            if require_nonempty and image_path.stat().st_size <= 0:
                empty_by_model[model_id] += 1
                continue
            rows_out.append(row)
            by_model[model_id] += 1

    if not fieldnames:
        raise ValueError(f"{labels_path} has no CSV header")
    written = write_csv(output_path, rows_out, fieldnames)
    summary = {
        "labels_path": str(labels_path),
        "output_path": str(output_path),
        "rows_seen": rows_seen,
        "rows_written": written,
        "rows_missing_image": sum(missing_by_model.values()),
        "rows_empty_image": sum(empty_by_model.values()),
        "require_nonempty": require_nonempty,
        "rows_written_by_model": dict(sorted(by_model.items())),
        "rows_missing_by_model": dict(sorted(missing_by_model.items())),
        "rows_empty_by_model": dict(sorted(empty_by_model.items())),
    }
    if summary_path:
        write_json(summary_path, summary)
    return summary

def filter_labels_by_required_signals(
    labels_path: Path,
    output_path: Path,
    *,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Write label rows whose benchmark-specific required signal columns are all filled."""

    rows_out: list[dict[str, Any]] = []
    by_model: Counter[str] = Counter()
    by_benchmark: Counter[str] = Counter()
    removed_by_benchmark: Counter[str] = Counter()
    removed_by_condition: Counter[str] = Counter()
    missing_by_column: Counter[str] = Counter()
    rows_seen = 0
    fieldnames: list[str] = []

    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            rows_seen += 1
            required = required_signal_columns(row)
            missing = [column for column in required if str(row.get(column, "")).strip() == ""]
            if missing:
                removed_by_benchmark[str(row.get("benchmark", ""))] += 1
                removed_by_condition[str(row.get("condition_type", ""))] += 1
                for column in missing:
                    missing_by_column[column] += 1
                continue
            rows_out.append(row)
            by_model[str(row.get("model_id", ""))] += 1
            by_benchmark[str(row.get("benchmark", ""))] += 1

    if not fieldnames:
        raise ValueError(f"{labels_path} has no CSV header")
    written = write_csv(output_path, rows_out, fieldnames)
    summary = {
        "labels_path": str(labels_path),
        "output_path": str(output_path),
        "rows_seen": rows_seen,
        "rows_written": written,
        "rows_removed_missing_required": rows_seen - written,
        "rows_written_by_model": dict(sorted(by_model.items())),
        "rows_written_by_benchmark": dict(sorted(by_benchmark.items())),
        "rows_removed_by_benchmark": dict(sorted(removed_by_benchmark.items())),
        "rows_removed_by_condition": dict(sorted(removed_by_condition.items())),
        "missing_required_by_column": dict(sorted(missing_by_column.items())),
    }
    if summary_path:
        write_json(summary_path, summary)
    return summary

