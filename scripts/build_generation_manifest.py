#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.config import load_config
from t2i_distill.manifest import build_generation_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model x prompt x seed generation jobs.")
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--prompt-bank", default="data/prompts/inheritance_prompt_bank.jsonl")
    parser.add_argument("--output", default="data/manifests/generation_manifest.jsonl")
    parser.add_argument("--model-scope", choices=["primary", "secondary", "all"], default="primary")
    parser.add_argument("--benchmarks", default="", help="Comma-separated benchmark ids. Empty means all prompt-bank benchmarks.")
    parser.add_argument("--include-gated", action="store_true", help="Include valid-lineage-only unlearning prompts.")
    parser.add_argument("--limit-prompts-per-benchmark", type=int, default=None)
    parser.add_argument("--job-order", choices=["model_major", "prompt_major"], default="model_major")
    args = parser.parse_args()

    benchmark_filter = {item.strip() for item in args.benchmarks.split(",") if item.strip()} or None
    summary = build_generation_manifest(
        load_config(Path(args.config)),
        Path(args.prompt_bank),
        Path(args.output),
        model_scope=args.model_scope,
        include_gated=args.include_gated,
        benchmark_filter=benchmark_filter,
        limit_prompts_per_benchmark=args.limit_prompts_per_benchmark,
        job_order=args.job_order,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
