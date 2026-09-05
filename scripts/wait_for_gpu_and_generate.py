#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait until a GPU is below memory/utilization thresholds, then run a generation manifest.")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--max-memory-mib", type=int, default=8000)
    parser.add_argument("--max-utilization", type=int, default=15)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-wait-seconds", type=int, default=0, help="0 means wait forever.")
    parser.add_argument("--manifest", default="data/preflight/generation_manifest.jsonl")
    parser.add_argument("--success-log", default="results/generation/preflight_success.jsonl")
    parser.add_argument("--failure-log", default="results/generation/preflight_failures.jsonl")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", help="Report the command that would run after GPU availability.")
    args = parser.parse_args()

    started = time.time()
    while True:
        state = _gpu_state(args.gpu)
        ready = state["memory_used_mib"] <= args.max_memory_mib and state["utilization_gpu"] <= args.max_utilization
        print(json.dumps({"gpu": args.gpu, "ready": ready, **state}, ensure_ascii=False), flush=True)
        if ready:
            command = [
                sys.executable,
                "scripts/generate_diffusers.py",
                "--manifest",
                args.manifest,
                "--device",
                f"cuda:{args.gpu}",
                "--success-log",
                args.success_log,
                "--failure-log",
                args.failure_log,
            ]
            if args.skip_existing:
                command.append("--skip-existing")
            if args.continue_on_error:
                command.append("--continue-on-error")
            if args.dry_run:
                print(json.dumps({"would_run": command}, ensure_ascii=False, indent=2))
                return
            raise SystemExit(subprocess.run(command, check=False).returncode)
        if args.max_wait_seconds and time.time() - started >= args.max_wait_seconds:
            raise SystemExit("GPU did not become available before max wait time")
        time.sleep(args.poll_seconds)


def _gpu_state(gpu: str) -> dict[str, int]:
    query = "memory.used,utilization.gpu"
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    mem, util = [int(part.strip()) for part in result.stdout.strip().split(",")]
    return {"memory_used_mib": mem, "utilization_gpu": util}


if __name__ == "__main__":
    main()
