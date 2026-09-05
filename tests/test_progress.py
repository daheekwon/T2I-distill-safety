from __future__ import annotations

from pathlib import Path

from t2i_distill.io import write_json, write_jsonl
from t2i_distill.progress import summarize_generation_progress


def test_summarize_generation_progress_from_logs(tmp_path: Path) -> None:
    shard = tmp_path / "shard_00000.jsonl"
    write_jsonl(
        shard,
        [
            {"job_id": "a", "output_path": str(tmp_path / "a.png")},
            {"job_id": "b", "output_path": str(tmp_path / "b.png")},
        ],
    )
    index = tmp_path / "index.json"
    write_json(index, {"shards": [{"shard_id": "shard_00000", "path": str(shard), "jobs": 2}]})
    logs = tmp_path / "logs"
    write_jsonl(logs / "shard_00000_success.jsonl", [{"job_id": "a"}])
    result = summarize_generation_progress(index, logs_dir=logs, output_csv=tmp_path / "progress.csv", output_json=tmp_path / "progress.json")
    assert result["summary"]["jobs"] == 2
    assert result["summary"]["success_log_rows"] == 1
    assert result["shards"][0]["status"] == "partial"
