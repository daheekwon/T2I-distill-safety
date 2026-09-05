#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.evaluation import plan_evaluator_batches


def _split(value: str) -> set[str] | None:
    values = {item.strip() for item in value.split(",") if item.strip()}
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or export sharded evaluator batches by evaluator kind, benchmark, condition, and optional model.")
    parser.add_argument("--labels", default="data/labels/label_template.csv")
    parser.add_argument("--output-dir", default="data/evaluation/batches_full")
    parser.add_argument("--batch-index", default="data/evaluation/batch_index.csv")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--split-fields", default="evaluator_kind,benchmark,condition_type", help="Comma-separated fields. Common choices: evaluator_kind,benchmark,condition_type,model_id.")
    parser.add_argument("--evaluator-kinds", default="")
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--conditions", default="")
    parser.add_argument("--model-ids", default="")
    parser.add_argument("--only-existing-images", action="store_true")
    parser.add_argument("--prompt-mode", choices=["needed", "always", "never"], default="needed")
    parser.add_argument("--allow-safety-prompt-text", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write only the batch index, not JSONL evaluator input files.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path. Defaults to <batch-index>.summary.json.")
    args = parser.parse_args()

    split_fields = tuple(item.strip() for item in args.split_fields.split(",") if item.strip())
    batch_index_path = Path(args.batch_index)
    summary = plan_evaluator_batches(
        Path(args.labels),
        Path(args.output_dir),
        batch_index_path,
        batch_size=args.batch_size,
        split_fields=split_fields,
        evaluator_kinds=_split(args.evaluator_kinds),
        benchmarks=_split(args.benchmarks),
        conditions=_split(args.conditions),
        model_ids=_split(args.model_ids),
        only_existing_images=args.only_existing_images,
        prompt_mode=args.prompt_mode,
        redact_safety_prompts=not args.allow_safety_prompt_text,
        dry_run=args.dry_run,
    )
    summary_json_path = Path(args.summary_json) if args.summary_json else batch_index_path.with_suffix(".summary.json")
    summary["summary_json"] = str(summary_json_path)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
