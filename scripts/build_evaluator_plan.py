#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.evaluation import build_evaluator_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluator schemas and a compact task index from a label template.")
    parser.add_argument("--labels", default="data/labels/label_template.csv")
    parser.add_argument("--task-index", default="data/evaluation/evaluator_task_index.csv")
    parser.add_argument("--schema", default="data/evaluation/evaluator_schema.json")
    parser.add_argument("--summary", default="data/evaluation/evaluator_summary.csv")
    args = parser.parse_args()

    summary = build_evaluator_plan(
        Path(args.labels),
        Path(args.task_index),
        Path(args.schema),
        Path(args.summary),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
