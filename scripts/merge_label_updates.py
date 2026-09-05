#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.evaluation import merge_label_updates, validate_label_updates


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge evaluator update files into a filled label CSV.")
    parser.add_argument("updates", nargs="+", help="Evaluator update files, JSONL or CSV.")
    parser.add_argument("--labels", default="data/labels/label_template.csv")
    parser.add_argument("--output", default="data/labels/filled_labels.csv")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    if not args.skip_validation:
        invalid = [path for path in args.updates if not validate_label_updates(Path(path))["valid"]]
        if invalid:
            raise SystemExit(f"invalid update files: {invalid}")
    summary = merge_label_updates(
        Path(args.labels),
        [Path(path) for path in args.updates],
        Path(args.output),
        strict=args.strict,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
