#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.generation_queue import plan_generation_queue, plan_multi_gpu_generation_queue, run_generation_queue, run_multi_gpu_generation_queue


def _split(value: str) -> set[str] | None:
    values = {item.strip() for item in value.split(",") if item.strip()}
    return values or None


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable generation shards sequentially once the requested GPU is available.")
    parser.add_argument("--shard-index", default="data/manifests/shards_primary/index.json")
    parser.add_argument("--logs-dir", default="results/generation")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--gpus", default="", help="Comma-separated GPU indices for one parent-managed multi-GPU queue, e.g. 0,1.")
    parser.add_argument("--max-shards", type=int, default=1, help="Number of matching shards to run; 0 means all matching shards.")
    parser.add_argument("--statuses", default="pending,partial", help="Comma-separated progress statuses to run.")
    parser.add_argument("--model-ids", default="")
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--generation-script", default="scripts/generate_diffusers.py")
    parser.add_argument("--max-memory-mib", type=int, default=8000)
    parser.add_argument("--max-utilization", type=int, default=15)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-seconds", type=int, default=0, help="0 means wait forever.")
    parser.add_argument("--status-every", type=int, default=25)
    parser.add_argument("--max-loaded-pipelines", type=int, default=1)
    parser.add_argument("--max-failures", type=int, default=0)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-plan-output", default="results/generation/run_plan.json")
    parser.add_argument("--lock-path", default="results/generation/run_generation_queue.lock")
    parser.add_argument("--force-lock", action="store_true")
    args = parser.parse_args()

    gpus = _split_list(args.gpus)
    if gpus:
        plan = plan_multi_gpu_generation_queue(
            Path(args.shard_index),
            gpus=gpus,
            logs_dir=Path(args.logs_dir),
            max_shards=args.max_shards,
            statuses=_split(args.statuses),
            model_ids=_split(args.model_ids),
            benchmarks=_split(args.benchmarks),
            generation_script=args.generation_script,
            skip_existing=not args.no_skip_existing,
            continue_on_error=not args.stop_on_error,
            max_loaded_pipelines=args.max_loaded_pipelines,
            status_every=args.status_every,
            max_failures=args.max_failures,
        )
        result = run_multi_gpu_generation_queue(
            plan,
            max_memory_mib=args.max_memory_mib,
            max_utilization=args.max_utilization,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
            dry_run=args.dry_run,
            run_plan_output=Path(args.run_plan_output) if args.run_plan_output else None,
            lock_path=Path(args.lock_path) if args.lock_path else None,
            force_lock=args.force_lock,
        )
    else:
        plan = plan_generation_queue(
            Path(args.shard_index),
            logs_dir=Path(args.logs_dir),
            max_shards=args.max_shards,
            statuses=_split(args.statuses),
            model_ids=_split(args.model_ids),
            benchmarks=_split(args.benchmarks),
            gpu=args.gpu,
            generation_script=args.generation_script,
            skip_existing=not args.no_skip_existing,
            continue_on_error=not args.stop_on_error,
            max_loaded_pipelines=args.max_loaded_pipelines,
            status_every=args.status_every,
            max_failures=args.max_failures,
        )
        result = run_generation_queue(
            plan,
            max_memory_mib=args.max_memory_mib,
            max_utilization=args.max_utilization,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
            dry_run=args.dry_run,
            run_plan_output=Path(args.run_plan_output) if args.run_plan_output else None,
            lock_path=Path(args.lock_path) if args.lock_path else None,
            force_lock=args.force_lock,
        )
    print(
        json.dumps(
            {
                "dry_run": result["dry_run"],
                "selected_shard_count": result["selected_shard_count"],
                "selected_jobs": result["selected_jobs"],
                "completed_shards": result["completed_shards"],
                "failed_shards": result["failed_shards"],
                "gpus": result.get("gpus", [result.get("gpu")]),
                "status": result.get("status", "unknown"),
                "run_plan_output": args.run_plan_output,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if result["failed_shards"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
