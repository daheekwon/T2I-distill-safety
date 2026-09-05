#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate filled label CSVs and audit duplicate label_uids.")
    parser.add_argument("labels", nargs="+", help="Filled label CSV files to concatenate.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    sources: list[dict[str, object]] = []

    for label_path_text in args.labels:
        label_path = Path(label_path_text)
        with label_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            local_rows = 0
            for row in reader:
                row["_combined_source"] = str(label_path)
                rows.append(row)
                local_rows += 1
            sources.append({"path": str(label_path), "rows": local_rows})

    if "_combined_source" not in fieldnames:
        fieldnames.append("_combined_source")

    seen: set[str] = set()
    duplicate_label_uids: list[str] = []
    rows_by_model: dict[str, int] = {}
    rows_by_benchmark: dict[str, int] = {}
    rows_by_condition: dict[str, int] = {}
    rows_by_source: dict[str, int] = {}

    for row in rows:
        label_uid = row.get("label_uid", "")
        if label_uid in seen:
            duplicate_label_uids.append(label_uid)
        else:
            seen.add(label_uid)
        model_id = row.get("model_id", "")
        benchmark = row.get("benchmark", "")
        condition_type = row.get("condition_type", "")
        source = row.get("_combined_source", "")
        rows_by_model[model_id] = rows_by_model.get(model_id, 0) + 1
        rows_by_benchmark[benchmark] = rows_by_benchmark.get(benchmark, 0) + 1
        rows_by_condition[condition_type] = rows_by_condition.get(condition_type, 0) + 1
        rows_by_source[source] = rows_by_source.get(source, 0) + 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_path": str(output_path),
        "rows_written": len(rows),
        "sources": sources,
        "duplicate_label_uids": len(duplicate_label_uids),
        "duplicate_label_uid_preview": duplicate_label_uids[:10],
        "rows_by_source": dict(sorted(rows_by_source.items())),
        "rows_by_model": dict(sorted(rows_by_model.items())),
        "rows_by_benchmark": dict(sorted(rows_by_benchmark.items())),
        "rows_by_condition": dict(sorted(rows_by_condition.items())),
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
