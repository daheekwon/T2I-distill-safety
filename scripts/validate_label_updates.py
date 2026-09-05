#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.evaluation import validate_label_updates, write_update_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate evaluator label-update JSONL/CSV before merging into a label CSV.")
    parser.add_argument("updates", nargs="+", help="Evaluator update files, JSONL or CSV.")
    parser.add_argument("--output-dir", default="results/evaluation_validation")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for update in args.updates:
        path = Path(update)
        summary = validate_label_updates(path)
        summaries.append(summary)
        report = output_dir / f"{path.stem}_validation.json"
        write_update_validation_report(summary, report)
    combined = {
        "files": len(summaries),
        "valid": all(item["valid"] for item in summaries),
        "error_count": sum(int(item["error_count"]) for item in summaries),
        "summaries": summaries,
    }
    print(json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True))
    if args.fail_on_invalid and not combined["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
