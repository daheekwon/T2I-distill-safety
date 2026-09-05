#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.evaluation import export_evaluator_batch


def _split(value: str) -> set[str] | None:
    values = {item.strip() for item in value.split(",") if item.strip()}
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export filtered evaluator batches from the label template as JSONL.")
    parser.add_argument("--labels", default="data/labels/label_template.csv")
    parser.add_argument("--output", default="data/evaluation/batches/evaluator_batch.jsonl")
    parser.add_argument("--evaluator-kinds", default="", help="Comma-separated evaluator kinds.")
    parser.add_argument("--benchmarks", default="", help="Comma-separated benchmark ids.")
    parser.add_argument("--conditions", default="", help="Comma-separated condition types.")
    parser.add_argument("--model-ids", default="", help="Comma-separated model ids.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--only-existing-images", action="store_true")
    parser.add_argument("--prompt-mode", choices=["needed", "always", "never"], default="needed")
    parser.add_argument("--allow-safety-prompt-text", action="store_true")
    args = parser.parse_args()

    summary = export_evaluator_batch(
        Path(args.labels),
        Path(args.output),
        evaluator_kinds=_split(args.evaluator_kinds),
        benchmarks=_split(args.benchmarks),
        conditions=_split(args.conditions),
        model_ids=_split(args.model_ids),
        limit=args.limit,
        start_index=args.start_index,
        only_existing_images=args.only_existing_images,
        prompt_mode=args.prompt_mode,
        redact_safety_prompts=not args.allow_safety_prompt_text,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
