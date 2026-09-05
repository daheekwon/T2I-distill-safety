#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.preflight import build_preflight_package


def _seeds(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small full-schema preflight package from the full prompt bank and manifest.")
    parser.add_argument("--prompt-bank", default="data/prompts/inheritance_prompt_bank.jsonl")
    parser.add_argument("--manifest", default="data/manifests/generation_manifest.jsonl")
    parser.add_argument("--output-dir", default="data/preflight")
    parser.add_argument("--prompts-per-condition", type=int, default=1)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--jobs-per-shard", type=int, default=200)
    args = parser.parse_args()

    summary = build_preflight_package(
        Path(args.prompt_bank),
        Path(args.manifest),
        Path(args.output_dir),
        prompts_per_condition=args.prompts_per_condition,
        seeds=_seeds(args.seeds),
        jobs_per_shard=args.jobs_per_shard,
    )
    printable = dict(summary)
    printable.pop("selected_by_condition", None)
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
