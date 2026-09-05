#!/usr/bin/env python
"""Build the MATS20 safety/memorization pilot package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from t2i_distill.mats20 import build_mats20_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt-bank",
        default="data/prompts/inheritance_prompt_bank.jsonl",
        help="Source prompt bank JSONL.",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/generation_manifest.jsonl",
        help="Full canonical generation manifest JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/mats20",
        help="Directory for the MATS20 package.",
    )
    parser.add_argument(
        "--jobs-per-shard",
        type=int,
        default=200,
        help="Maximum jobs per generation/evaluator shard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_mats20_package(
        Path(args.prompt_bank),
        Path(args.manifest),
        Path(args.output_dir),
        jobs_per_shard=args.jobs_per_shard,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
