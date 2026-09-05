#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.progress import summarize_generation_progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize generation progress across manifest shards.")
    parser.add_argument("--shard-index", default="data/manifests/shards_primary/index.json")
    parser.add_argument("--logs-dir", default="results/generation")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--output-csv", default="results/generation/progress_by_shard.csv")
    parser.add_argument("--output-json", default="results/generation/progress_summary.json")
    args = parser.parse_args()

    result = summarize_generation_progress(
        Path(args.shard_index),
        logs_dir=Path(args.logs_dir),
        check_images=args.check_images,
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
    )
    summary = result["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
