#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.config import load_config
from t2i_distill.prompt_bank import build_prompt_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized all-benchmark prompt bank.")
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--benchmarks-root", default="benchmarks")
    parser.add_argument("--output", default="data/prompts/inheritance_prompt_bank.jsonl")
    parser.add_argument("--benchmarks", default="", help="Comma-separated benchmark ids. Empty means all non-gated benchmarks.")
    parser.add_argument("--include-gated", action="store_true", help="Include HUB/EMMA valid-lineage unlearning prompts.")
    args = parser.parse_args()

    selected = {item.strip() for item in args.benchmarks.split(",") if item.strip()} or None
    summary = build_prompt_bank(
        load_config(Path(args.config)),
        Path(args.benchmarks_root),
        Path(args.output),
        benchmarks=selected,
        include_gated=args.include_gated,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
