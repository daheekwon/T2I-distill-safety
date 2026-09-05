#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.execution import split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a generation manifest into resumable JSONL shards.")
    parser.add_argument("--manifest", default="data/manifests/generation_manifest.jsonl")
    parser.add_argument("--output-dir", default="data/manifests/shards_primary")
    parser.add_argument("--jobs-per-shard", type=int, default=10000)
    parser.add_argument("--index", default="")
    parser.add_argument("--respect-model-boundaries", action="store_true")
    args = parser.parse_args()

    index_path = Path(args.index) if args.index else None
    summary = split_manifest(
        Path(args.manifest),
        Path(args.output_dir),
        jobs_per_shard=args.jobs_per_shard,
        index_path=index_path,
        respect_model_boundaries=args.respect_model_boundaries,
    )
    printable = dict(summary)
    printable["shards"] = printable["shards"][:3]
    printable["shards_preview_count"] = len(printable["shards"])
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
