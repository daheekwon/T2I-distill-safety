from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import iter_jsonl, read_json, write_csv, write_json

PROGRESS_FIELDS = [
    "shard_id",
    "path",
    "jobs",
    "success_log_rows",
    "failure_log_rows",
    "existing_outputs",
    "missing_outputs",
    "completion_fraction",
    "failure_fraction",
    "status",
]


def summarize_generation_progress(
    shard_index_path: Path,
    *,
    logs_dir: Path = Path("results/generation"),
    check_images: bool = False,
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    index = read_json(shard_index_path)
    rows = []
    totals = Counter()
    for shard in index.get("shards", []):
        row = _shard_progress(shard, logs_dir=logs_dir, check_images=check_images)
        rows.append(row)
        totals["jobs"] += int(row["jobs"])
        totals["success_log_rows"] += int(row["success_log_rows"])
        totals["failure_log_rows"] += int(row["failure_log_rows"])
        totals["existing_outputs"] += int(row["existing_outputs"])
        totals["missing_outputs"] += int(row["missing_outputs"])
    summary = {
        "shard_index_path": str(shard_index_path),
        "logs_dir": str(logs_dir),
        "check_images": check_images,
        "shard_count": len(rows),
        "jobs": totals["jobs"],
        "success_log_rows": totals["success_log_rows"],
        "failure_log_rows": totals["failure_log_rows"],
        "existing_outputs": totals["existing_outputs"],
        "missing_outputs": totals["missing_outputs"],
        "completion_fraction_from_logs": totals["success_log_rows"] / totals["jobs"] if totals["jobs"] else 0.0,
        "completion_fraction_from_images": totals["existing_outputs"] / totals["jobs"] if check_images and totals["jobs"] else None,
        "failure_fraction_from_logs": totals["failure_log_rows"] / totals["jobs"] if totals["jobs"] else 0.0,
        "shards_done_by_logs": sum(row["status"] == "done_by_logs" for row in rows),
        "shards_with_failures": sum(int(row["failure_log_rows"]) > 0 for row in rows),
    }
    if output_csv is not None:
        write_csv(output_csv, rows, PROGRESS_FIELDS)
    if output_json is not None:
        write_json(output_json, {"summary": summary, "shards": rows})
    return {"summary": summary, "shards": rows}


def _shard_progress(shard: dict[str, Any], *, logs_dir: Path, check_images: bool) -> dict[str, Any]:
    shard_id = str(shard["shard_id"])
    jobs = int(shard["jobs"])
    success_count = _count_jsonl(logs_dir / f"{shard_id}_success.jsonl")
    failure_count = _count_jsonl(logs_dir / f"{shard_id}_failures.jsonl")
    existing_outputs = 0
    missing_outputs = 0
    if check_images:
        for job in iter_jsonl(Path(shard["path"])):
            path = Path(str(job.get("output_path", "")))
            if path.exists() and path.stat().st_size > 0:
                existing_outputs += 1
            else:
                missing_outputs += 1
    status = "pending"
    if success_count >= jobs:
        status = "done_by_logs"
    elif success_count > 0 or failure_count > 0:
        status = "partial"
    if check_images and existing_outputs >= jobs:
        status = "done_by_images"
    return {
        "shard_id": shard_id,
        "path": shard["path"],
        "jobs": jobs,
        "success_log_rows": success_count,
        "failure_log_rows": failure_count,
        "existing_outputs": existing_outputs,
        "missing_outputs": missing_outputs,
        "completion_fraction": success_count / jobs if jobs else 0.0,
        "failure_fraction": failure_count / jobs if jobs else 0.0,
        "status": status,
    }


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count
