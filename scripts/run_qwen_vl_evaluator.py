#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.qwen_vl_evaluator import (  # noqa: E402
    DEFAULT_QWEN_EVALUATOR_KINDS,
    DEFAULT_QWEN_MODEL_PATH,
    parse_filter,
    run_qwen_vl_evaluator,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run blinded Qwen3-VL-8B labels for generated-image evaluator tasks.")
    parser.add_argument("--labels", default="data/mats20/label_template_existing_images.csv")
    parser.add_argument("--output-updates", default="data/mats20/qwen3vl_safety_updates.csv")
    parser.add_argument("--raw-jsonl", default="results/mats20/qwen3vl/safety_raw.jsonl")
    parser.add_argument("--summary-json", default="results/mats20/qwen3vl/safety_summary.json")
    parser.add_argument("--model-path", default=DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument(
        "--evaluator-kinds",
        default=",".join(sorted(DEFAULT_QWEN_EVALUATOR_KINDS)),
        help="Comma-separated evaluator kinds. Default: safety_image_audit,safety_overrefusal_audit.",
    )
    parser.add_argument("--benchmarks", default=None, help="Comma-separated benchmark filter.")
    parser.add_argument("--conditions", default=None, help="Comma-separated condition_type filter.")
    parser.add_argument("--models", default=None, help="Comma-separated model_id filter.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--include-memorization", action="store_true", help="Allow Qwen quality-only rows for MemBench; memorization_score remains SSCD/reference-based.")
    parser.add_argument("--dry-run-prompts", action="store_true", help="Write blinded prompts to raw-jsonl without loading Qwen.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=786432)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    summary = run_qwen_vl_evaluator(
        labels_path=Path(args.labels),
        output_updates_path=Path(args.output_updates),
        raw_jsonl_path=Path(args.raw_jsonl),
        summary_json_path=Path(args.summary_json),
        model_path=args.model_path,
        evaluator_kinds=parse_filter(args.evaluator_kinds),
        benchmarks=parse_filter(args.benchmarks),
        conditions=parse_filter(args.conditions),
        model_ids=parse_filter(args.models),
        start_index=args.start_index,
        limit=args.limit,
        only_existing_images=not args.allow_missing_images,
        resume=not args.no_resume,
        include_memorization=args.include_memorization,
        dry_run_prompts=args.dry_run_prompts,
        device_map=args.device_map,
        load_in_4bit=not args.no_4bit,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
