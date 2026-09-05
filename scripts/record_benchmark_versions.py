#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Record benchmark source revisions and dirty-state snapshots.")
    parser.add_argument("--benchmarks-root", default="benchmarks")
    parser.add_argument("--output", default="results/audit/benchmark_versions.json")
    args = parser.parse_args()

    root = Path(args.benchmarks_root)
    rows = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        rows.append(_snapshot_repo(child))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"benchmarks_root": str(root), "sources": rows}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"sources": len(rows), "output": str(output)}, ensure_ascii=False, indent=2, sort_keys=True))


def _snapshot_repo(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"name": path.name, "path": str(path), "exists": path.exists(), "is_git": (path / ".git").exists()}
    if not record["is_git"]:
        record["file_count"] = sum(1 for item in path.rglob("*") if item.is_file())
        return record
    record["remote_url"] = _run_git(path, ["remote", "get-url", "origin"])
    record["head_commit"] = _run_git(path, ["rev-parse", "HEAD"])
    record["current_branch"] = _run_git(path, ["branch", "--show-current"])
    status = _run_git(path, ["status", "--short"])
    record["dirty"] = bool(status.strip())
    record["status_short"] = status.splitlines()
    return record


def _run_git(path: Path, args: list[str]) -> str:
    result = subprocess.run(["git", "-C", str(path), *args], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return result.stderr.strip()
    return result.stdout.strip()


if __name__ == "__main__":
    main()
