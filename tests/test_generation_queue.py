from __future__ import annotations

from pathlib import Path

import pytest

from t2i_distill.generation_queue import build_generation_command, plan_generation_queue, plan_multi_gpu_generation_queue, run_generation_queue, run_multi_gpu_generation_queue, select_shards_to_run
from t2i_distill.io import read_json, write_json, write_jsonl


def test_select_shards_skips_done_and_filters_model(tmp_path: Path) -> None:
    index = _write_index(tmp_path)
    logs = tmp_path / "logs"
    write_jsonl(logs / "shard_00000_success.jsonl", [{"job_id": "a"}, {"job_id": "b"}])

    selected = select_shards_to_run(index, logs_dir=logs, max_shards=0, model_ids={"m1"})

    assert [row["shard_id"] for row in selected] == ["shard_00001"]


def test_plan_generation_queue_builds_per_shard_logs(tmp_path: Path) -> None:
    index = _write_index(tmp_path)
    plan = plan_generation_queue(index, logs_dir=tmp_path / "logs", max_shards=1, gpu="2")

    assert plan["selected_shard_count"] == 1
    command = plan["selected_shards"][0]["command"]
    assert "--device" in command
    assert "cuda:2" in command
    assert str(tmp_path / "logs" / "shard_00000_success.jsonl") in command


def test_build_generation_command_can_disable_resume_flags(tmp_path: Path) -> None:
    command = build_generation_command(
        {"shard_id": "shard_00003", "path": "shard.jsonl"},
        logs_dir=tmp_path,
        skip_existing=False,
        continue_on_error=False,
    )

    assert "--skip-existing" not in command
    assert "--continue-on-error" not in command
    assert str(tmp_path / "shard_00003_failures.jsonl") in command


def test_run_generation_queue_respects_existing_lock_on_real_runs(tmp_path: Path) -> None:
    lock = tmp_path / "queue.lock"
    write_json(lock, {"pid": 123})
    plan = {"gpu": "2", "selected_shards": [], "selected_shard_count": 0, "selected_jobs": 0}

    with pytest.raises(RuntimeError):
        run_generation_queue(plan, dry_run=False, lock_path=lock)

    assert lock.exists()


def test_run_generation_queue_reports_gpu_wait_timeout_and_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_wait(*args: object, **kwargs: object) -> None:
        raise TimeoutError("GPU busy")

    monkeypatch.setattr("t2i_distill.generation_queue.wait_for_gpu", fail_wait)
    lock = tmp_path / "queue.lock"
    plan = {
        "gpu": "2",
        "selected_shards": [{"shard_id": "shard_00000", "command": ["false"]}],
        "selected_shard_count": 1,
        "selected_jobs": 1,
    }

    run_plan = tmp_path / "run_plan.json"
    result = run_generation_queue(plan, dry_run=False, lock_path=lock, max_wait_seconds=1, run_plan_output=run_plan)

    expected_failure = {"shard_id": "shard_00000", "returncode": None, "stage": "wait_for_gpu", "error": "GPU busy"}
    assert result["failed_shards"] == [expected_failure]
    assert result["status"] == "failed"
    assert read_json(run_plan)["failed_shards"] == [expected_failure]
    assert not lock.exists()


def test_plan_multi_gpu_generation_queue_assigns_unique_shards_round_robin(tmp_path: Path) -> None:
    index = _write_index(tmp_path)

    plan = plan_multi_gpu_generation_queue(index, gpus=["0", "cuda:1"], logs_dir=tmp_path / "logs", max_shards=0)

    assert plan["gpus"] == ["0", "1"]
    assert [row["shard_id"] for row in plan["selected_shards"]] == ["shard_00000", "shard_00001"]
    assert [row["gpu"] for row in plan["selected_shards"]] == ["0", "1"]
    assert "cuda:0" in plan["assignments"]["0"][0]["command"]
    assert "cuda:1" in plan["assignments"]["1"][0]["command"]


def test_run_multi_gpu_generation_queue_dry_run_writes_plan(tmp_path: Path) -> None:
    index = _write_index(tmp_path)
    plan = plan_multi_gpu_generation_queue(index, gpus=["0", "1"], logs_dir=tmp_path / "logs", max_shards=2)
    output = tmp_path / "multi_plan.json"

    result = run_multi_gpu_generation_queue(plan, dry_run=True, run_plan_output=output, lock_path=tmp_path / "queue.lock")

    assert result["status"] == "dry_run"
    assert read_json(output)["gpus"] == ["0", "1"]
    assert read_json(output)["selected_shard_count"] == 2


def test_run_multi_gpu_generation_queue_reports_gpu_wait_timeout_and_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_wait(*args: object, **kwargs: object) -> None:
        raise TimeoutError("GPU busy")

    monkeypatch.setattr("t2i_distill.generation_queue.wait_for_gpu", fail_wait)
    lock = tmp_path / "multi.lock"
    plan = {
        "gpus": ["0", "1"],
        "selected_shards": [],
        "assignments": {
            "0": [{"shard_id": "shard_00000", "command": ["false"]}],
            "1": [{"shard_id": "shard_00001", "command": ["false"]}],
        },
        "selected_shard_count": 2,
        "selected_jobs": 2,
    }

    result = run_multi_gpu_generation_queue(plan, dry_run=False, lock_path=lock, max_wait_seconds=1)

    assert result["status"] == "failed"
    assert len(result["failed_shards"]) >= 1
    assert not lock.exists()


def _write_index(tmp_path: Path) -> Path:
    shard0 = tmp_path / "shard_00000.jsonl"
    shard1 = tmp_path / "shard_00001.jsonl"
    write_jsonl(shard0, [{"job_id": "a"}, {"job_id": "b"}])
    write_jsonl(shard1, [{"job_id": "c"}])
    index = tmp_path / "index.json"
    write_json(
        index,
        {
            "shards": [
                {"shard_id": "shard_00000", "path": str(shard0), "jobs": 2, "jobs_by_model": {"m0": 2}, "jobs_by_benchmark": {"grade": 2}},
                {"shard_id": "shard_00001", "path": str(shard1), "jobs": 1, "jobs_by_model": {"m1": 1}, "jobs_by_benchmark": {"dimcim": 1}},
            ]
        },
    )
    return index
