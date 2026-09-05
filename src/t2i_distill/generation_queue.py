from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .io import read_json, write_json
from .progress import summarize_generation_progress


def select_shards_to_run(
    shard_index_path: Path,
    *,
    logs_dir: Path = Path("results/generation"),
    max_shards: int = 1,
    statuses: set[str] | None = None,
    model_ids: set[str] | None = None,
    benchmarks: set[str] | None = None,
) -> list[dict[str, Any]]:
    if max_shards < 0:
        raise ValueError("max_shards must be non-negative; use 0 for all matching shards")
    statuses = statuses or {"pending", "partial"}
    index = read_json(shard_index_path)
    progress = summarize_generation_progress(shard_index_path, logs_dir=logs_dir)["shards"]
    progress_by_id = {row["shard_id"]: row for row in progress}
    selected = []
    for shard in index.get("shards", []):
        shard_id = str(shard["shard_id"])
        progress_row = progress_by_id.get(shard_id, {})
        if str(progress_row.get("status", "pending")) not in statuses:
            continue
        if model_ids and not (set(shard.get("jobs_by_model", {})) & model_ids):
            continue
        if benchmarks and not (set(shard.get("jobs_by_benchmark", {})) & benchmarks):
            continue
        selected.append(
            {
                "shard_id": shard_id,
                "path": shard["path"],
                "jobs": int(shard.get("jobs", 0)),
                "jobs_by_model": shard.get("jobs_by_model", {}),
                "jobs_by_benchmark": shard.get("jobs_by_benchmark", {}),
                "progress": progress_row,
            }
        )
        if max_shards and len(selected) >= max_shards:
            break
    return selected


def build_generation_command(
    shard: dict[str, Any],
    *,
    generation_script: str = "scripts/generate_diffusers.py",
    gpu: str = "2",
    logs_dir: Path = Path("results/generation"),
    skip_existing: bool = True,
    continue_on_error: bool = True,
    max_loaded_pipelines: int = 1,
    status_every: int = 25,
    max_failures: int = 0,
) -> list[str]:
    shard_id = str(shard["shard_id"])
    gpu = _normalize_gpu(gpu)
    command = [
        sys.executable,
        generation_script,
        "--manifest",
        str(shard["path"]),
        "--device",
        _device_arg(gpu),
        "--success-log",
        str(logs_dir / f"{shard_id}_success.jsonl"),
        "--failure-log",
        str(logs_dir / f"{shard_id}_failures.jsonl"),
        "--max-loaded-pipelines",
        str(max_loaded_pipelines),
        "--status-every",
        str(status_every),
    ]
    if skip_existing:
        command.append("--skip-existing")
    if continue_on_error:
        command.append("--continue-on-error")
    if max_failures:
        command.extend(["--max-failures", str(max_failures)])
    return command


def _normalize_gpu(gpu: str) -> str:
    value = str(gpu).strip()
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    if not value:
        raise ValueError("gpu must not be empty")
    return value


def _device_arg(gpu: str) -> str:
    return gpu if str(gpu).startswith("cuda:") else f"cuda:{gpu}"


def plan_generation_queue(
    shard_index_path: Path,
    *,
    logs_dir: Path = Path("results/generation"),
    max_shards: int = 1,
    statuses: set[str] | None = None,
    model_ids: set[str] | None = None,
    benchmarks: set[str] | None = None,
    gpu: str = "2",
    generation_script: str = "scripts/generate_diffusers.py",
    skip_existing: bool = True,
    continue_on_error: bool = True,
    max_loaded_pipelines: int = 1,
    status_every: int = 25,
    max_failures: int = 0,
) -> dict[str, Any]:
    gpu = _normalize_gpu(gpu)
    shards = select_shards_to_run(
        shard_index_path,
        logs_dir=logs_dir,
        max_shards=max_shards,
        statuses=statuses,
        model_ids=model_ids,
        benchmarks=benchmarks,
    )
    planned = []
    for shard in shards:
        planned.append(
            {
                **shard,
                "command": build_generation_command(
                    shard,
                    generation_script=generation_script,
                    gpu=gpu,
                    logs_dir=logs_dir,
                    skip_existing=skip_existing,
                    continue_on_error=continue_on_error,
                    max_loaded_pipelines=max_loaded_pipelines,
                    status_every=status_every,
                    max_failures=max_failures,
                ),
            }
        )
    return {
        "shard_index_path": str(shard_index_path),
        "logs_dir": str(logs_dir),
        "gpu": gpu,
        "max_shards": max_shards,
        "statuses": sorted(statuses or {"pending", "partial"}),
        "model_ids": sorted(model_ids or []),
        "benchmarks": sorted(benchmarks or []),
        "selected_shards": planned,
        "selected_shard_count": len(planned),
        "selected_jobs": sum(int(shard["jobs"]) for shard in planned),
    }



def plan_multi_gpu_generation_queue(
    shard_index_path: Path,
    *,
    gpus: list[str],
    logs_dir: Path = Path("results/generation"),
    max_shards: int = 1,
    statuses: set[str] | None = None,
    model_ids: set[str] | None = None,
    benchmarks: set[str] | None = None,
    generation_script: str = "scripts/generate_diffusers.py",
    skip_existing: bool = True,
    continue_on_error: bool = True,
    max_loaded_pipelines: int = 1,
    status_every: int = 25,
    max_failures: int = 0,
) -> dict[str, Any]:
    normalized_gpus = [_normalize_gpu(gpu) for gpu in gpus]
    if not normalized_gpus:
        raise ValueError("at least one GPU must be provided")
    shards = select_shards_to_run(
        shard_index_path,
        logs_dir=logs_dir,
        max_shards=max_shards,
        statuses=statuses,
        model_ids=model_ids,
        benchmarks=benchmarks,
    )
    assignments: dict[str, list[dict[str, Any]]] = {gpu: [] for gpu in normalized_gpus}
    planned = []
    for idx, shard in enumerate(shards):
        gpu = normalized_gpus[idx % len(normalized_gpus)]
        planned_shard = {
            **shard,
            "gpu": gpu,
            "command": build_generation_command(
                shard,
                generation_script=generation_script,
                gpu=gpu,
                logs_dir=logs_dir,
                skip_existing=skip_existing,
                continue_on_error=continue_on_error,
                max_loaded_pipelines=max_loaded_pipelines,
                status_every=status_every,
                max_failures=max_failures,
            ),
        }
        assignments[gpu].append(planned_shard)
        planned.append(planned_shard)
    return {
        "shard_index_path": str(shard_index_path),
        "logs_dir": str(logs_dir),
        "gpus": normalized_gpus,
        "max_shards": max_shards,
        "statuses": sorted(statuses or {"pending", "partial"}),
        "model_ids": sorted(model_ids or []),
        "benchmarks": sorted(benchmarks or []),
        "selected_shards": planned,
        "assignments": assignments,
        "selected_shard_count": len(planned),
        "selected_jobs": sum(int(shard["jobs"]) for shard in planned),
    }


def run_generation_queue(
    plan: dict[str, Any],
    *,
    max_memory_mib: int = 8000,
    max_utilization: int = 15,
    poll_seconds: int = 60,
    max_wait_seconds: int = 0,
    dry_run: bool = False,
    run_plan_output: Path | None = None,
    lock_path: Path | None = None,
    force_lock: bool = False,
) -> dict[str, Any]:
    if run_plan_output is not None:
        write_json(run_plan_output, {**plan, "dry_run": dry_run, "completed_shards": [], "failed_shards": [], "status": "planned"})
    if dry_run:
        result = {**plan, "dry_run": True, "completed_shards": [], "failed_shards": [], "status": "dry_run"}
        if run_plan_output is not None:
            write_json(run_plan_output, result)
        return result
    completed = []
    failed = []
    acquired_lock = False
    lock_path = lock_path or Path("results/generation/run_generation_queue.lock")
    try:
        _acquire_lock(lock_path, plan, force=force_lock)
        acquired_lock = True
        started = time.time()
        for shard in plan["selected_shards"]:
            try:
                wait_for_gpu(
                    str(plan["gpu"]),
                    max_memory_mib=max_memory_mib,
                    max_utilization=max_utilization,
                    poll_seconds=poll_seconds,
                    max_wait_seconds=max_wait_seconds,
                    started_at=started,
                )
            except TimeoutError as exc:
                failed.append({"shard_id": shard["shard_id"], "returncode": None, "stage": "wait_for_gpu", "error": str(exc)})
                break
            result = subprocess.run(shard["command"], check=False)
            row = {"shard_id": shard["shard_id"], "returncode": result.returncode}
            if result.returncode == 0:
                completed.append(row)
            else:
                failed.append(row)
                break
        result = {
            **plan,
            "dry_run": False,
            "completed_shards": completed,
            "failed_shards": failed,
            "status": "failed" if failed else "completed",
        }
        if run_plan_output is not None:
            write_json(run_plan_output, result)
        return result
    finally:
        if acquired_lock:
            _release_lock(lock_path)



def run_multi_gpu_generation_queue(
    plan: dict[str, Any],
    *,
    max_memory_mib: int = 8000,
    max_utilization: int = 15,
    poll_seconds: int = 60,
    max_wait_seconds: int = 0,
    dry_run: bool = False,
    run_plan_output: Path | None = None,
    lock_path: Path | None = None,
    force_lock: bool = False,
) -> dict[str, Any]:
    base_result = {**plan, "dry_run": dry_run, "completed_shards": [], "failed_shards": [], "status": "planned"}
    if run_plan_output is not None:
        write_json(run_plan_output, base_result)
    if dry_run:
        result = {**base_result, "status": "dry_run"}
        if run_plan_output is not None:
            write_json(run_plan_output, result)
        return result

    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    result_lock = threading.Lock()
    stop_event = threading.Event()
    lock_path = lock_path or Path("results/generation/run_generation_queue.lock")
    acquired_lock = False

    def snapshot(status: str) -> dict[str, Any]:
        return {**plan, "dry_run": False, "completed_shards": list(completed), "failed_shards": list(failed), "status": status}

    def persist(status: str) -> None:
        if run_plan_output is not None:
            write_json(run_plan_output, snapshot(status))

    def worker(gpu: str, shards: list[dict[str, Any]], started: float) -> None:
        for shard in shards:
            if stop_event.is_set():
                return
            try:
                wait_for_gpu(
                    gpu,
                    max_memory_mib=max_memory_mib,
                    max_utilization=max_utilization,
                    poll_seconds=poll_seconds,
                    max_wait_seconds=max_wait_seconds,
                    started_at=started,
                )
            except TimeoutError as exc:
                with result_lock:
                    failed.append({"gpu": gpu, "shard_id": shard["shard_id"], "returncode": None, "stage": "wait_for_gpu", "error": str(exc)})
                    stop_event.set()
                    persist("failed")
                return
            result = subprocess.run(shard["command"], check=False)
            row = {"gpu": gpu, "shard_id": shard["shard_id"], "returncode": result.returncode}
            with result_lock:
                if result.returncode == 0:
                    completed.append(row)
                    persist("running")
                else:
                    failed.append(row)
                    stop_event.set()
                    persist("failed")
                    return

    try:
        _acquire_lock(lock_path, plan, force=force_lock)
        acquired_lock = True
        started = time.time()
        threads = []
        for gpu, shards in (plan.get("assignments") or {}).items():
            if not shards:
                continue
            thread = threading.Thread(target=worker, args=(str(gpu), shards, started), daemon=False)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        final_status = "failed" if failed else "completed"
        result = snapshot(final_status)
        if run_plan_output is not None:
            write_json(run_plan_output, result)
        return result
    finally:
        if acquired_lock:
            _release_lock(lock_path)


def wait_for_gpu(
    gpu: str,
    *,
    max_memory_mib: int,
    max_utilization: int,
    poll_seconds: int,
    max_wait_seconds: int,
    started_at: float | None = None,
) -> None:
    started_at = time.time() if started_at is None else started_at
    while True:
        state = gpu_state(gpu)
        if state["memory_used_mib"] <= max_memory_mib and state["utilization_gpu"] <= max_utilization:
            return
        if max_wait_seconds and time.time() - started_at >= max_wait_seconds:
            raise TimeoutError(f"GPU {gpu} did not become available before max wait time")
        time.sleep(poll_seconds)


def gpu_state(gpu: str) -> dict[str, int]:
    query = "memory.used,utilization.gpu"
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    mem, util = [int(part.strip()) for part in result.stdout.strip().split(",")]
    return {"memory_used_mib": mem, "utilization_gpu": util}


def _acquire_lock(lock_path: Path, plan: dict[str, Any], *, force: bool) -> None:
    if lock_path.exists() and not force:
        raise RuntimeError(f"generation queue lock already exists: {lock_path}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        lock_path,
        {
            "pid": os.getpid(),
            "created_at_unix": time.time(),
            "gpu": plan.get("gpu", ""),
            "selected_shard_count": plan.get("selected_shard_count", 0),
            "selected_jobs": plan.get("selected_jobs", 0),
        },
    )


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
