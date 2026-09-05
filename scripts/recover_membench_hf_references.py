#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.image_proxies import MEMBENCH_REFERENCE_FIELDS
from t2i_distill.io import ensure_parent, write_csv, write_json


DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover MemBench references from a public HF dataset only when the source URL matches exactly."
    )
    parser.add_argument("--input-index-json", default="data/mats20/membench_reference_index.json")
    parser.add_argument("--output-index-json", default="data/mats20/membench_reference_index_recovered.json")
    parser.add_argument("--output-index-csv", default="data/mats20/membench_reference_index_recovered.csv")
    parser.add_argument("--output-dir", default="benchmarks/MemBench/references_mats20_hf")
    parser.add_argument("--dataset", default="gdhanuka/memorization_data2")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout-s", type=float, default=45.0)
    parser.add_argument("--sleep-s", type=float, default=0.15)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_index_json).read_text(encoding="utf-8"))
    records = [record for record in payload.get("records", []) if isinstance(record, dict)]
    unresolved = {
        str(record["url"]): record
        for record in records
        if str(record.get("status")) == "failed"
        and record.get("url")
        and not _eval_uid_has_ok_reference(records, str(record.get("eval_uid", "")))
    }
    rows, fetch_errors = _load_hf_rows(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        page_size=args.page_size,
        timeout_s=args.timeout_s,
        sleep_s=args.sleep_s,
    )
    by_url = {str(row.get("url", "")): row for row in rows if row.get("url")}

    recovered: list[dict[str, Any]] = []
    for url, source_record in sorted(unresolved.items()):
        hf_row = by_url.get(url)
        if hf_row is None:
            recovered.append(_failed_recovery_record(source_record, "hf_url_not_found", args.dataset))
            continue
        image_info = hf_row.get("ground_truth")
        src = image_info.get("src") if isinstance(image_info, dict) else ""
        if not src:
            recovered.append(_failed_recovery_record(source_record, "hf_ground_truth_missing", args.dataset))
            continue
        recovered.append(
            _download_recovered_reference(
                source_record,
                src,
                output_dir=Path(args.output_dir),
                timeout_s=args.timeout_s,
                source_dataset=args.dataset,
                hf_row=hf_row,
            )
        )

    combined_records = records + recovered
    summary = dict(payload.get("summary", {}))
    summary.update(
        {
            "recovery_source": args.dataset,
            "recovery_rows_scanned": len(rows),
            "recovery_fetch_errors": fetch_errors,
            "recovery_candidates": len(unresolved),
            "recovery_records": len(recovered),
            "recovery_ok_records": sum(1 for record in recovered if str(record.get("status", "")).startswith("ok")),
            "recovered_eval_uids_with_reference": len(
                {
                    str(record.get("eval_uid", ""))
                    for record in combined_records
                    if str(record.get("status", "")).startswith("ok") and record.get("path")
                }
            ),
        }
    )
    out = {"summary": summary, "records": combined_records}
    write_json(Path(args.output_index_json), out)
    write_csv(Path(args.output_index_csv), combined_records, MEMBENCH_REFERENCE_FIELDS)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _eval_uid_has_ok_reference(records: list[dict[str, Any]], eval_uid: str) -> bool:
    return any(str(record.get("eval_uid", "")) == eval_uid and str(record.get("status", "")).startswith("ok") for record in records)


def _load_hf_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    page_size: int,
    timeout_s: float,
    sleep_s: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        url = f"{DATASET_ROWS_URL}?{query}"
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                page = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"offset={offset}:{type(exc).__name__}:{exc}")
            break
        total = int(page.get("num_rows_total", 0))
        page_rows = [entry.get("row", {}) for entry in page.get("rows", []) if isinstance(entry, dict)]
        rows.extend([row for row in page_rows if isinstance(row, dict)])
        if not page_rows:
            break
        offset += len(page_rows)
        if sleep_s > 0:
            time.sleep(sleep_s)
    return rows, errors


def _download_recovered_reference(
    source_record: dict[str, Any],
    src: str,
    *,
    output_dir: Path,
    timeout_s: float,
    source_dataset: str,
    hf_row: dict[str, Any],
) -> dict[str, Any]:
    target = output_dir / str(source_record.get("eval_uid", "unknown")).replace(":", "_") / "hf_ground_truth.jpg"
    try:
        request = urllib.request.Request(src, headers={"User-Agent": "t2i-distill-mats20/1.0"})
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = response.read()
        ensure_parent(target)
        target.write_bytes(data)
        with Image.open(target) as image:
            image.verify()
        with Image.open(target) as image:
            width, height = image.size
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return _failed_recovery_record(source_record, f"hf_download:{type(exc).__name__}:{exc}", source_dataset)
    except Exception as exc:
        return _failed_recovery_record(source_record, f"hf_invalid_image:{type(exc).__name__}:{exc}", source_dataset)
    record = dict(source_record)
    record.update(
        {
            "url_index": f"{source_record.get('url_index', '')}:hf_ground_truth",
            "status": "ok_hf_ground_truth",
            "path": str(target),
            "bytes": target.stat().st_size,
            "width": width,
            "height": height,
            "error": f"source_dataset={source_dataset};hf_index={hf_row.get('index','')}",
        }
    )
    return record


def _failed_recovery_record(source_record: dict[str, Any], error: str, source_dataset: str) -> dict[str, Any]:
    record = dict(source_record)
    record.update(
        {
            "url_index": f"{source_record.get('url_index', '')}:hf_ground_truth",
            "status": "failed_hf_ground_truth",
            "path": "",
            "bytes": 0,
            "width": 0,
            "height": 0,
            "error": f"{error};source_dataset={source_dataset}",
        }
    )
    return record


if __name__ == "__main__":
    main()
