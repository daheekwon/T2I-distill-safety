#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.evaluator_audit import audit_evaluator_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit evaluator schema, task index, optional batch index, and batch instruction/redaction quality.")
    parser.add_argument("--labels", default="data/labels/label_template.csv")
    parser.add_argument("--schema", default="data/evaluation/evaluator_schema.json")
    parser.add_argument("--task-index", default="data/evaluation/evaluator_task_index.csv")
    parser.add_argument("--batch-index", default="data/evaluation/batch_index.csv")
    parser.add_argument("--batch-dir", default="")
    parser.add_argument("--output-json", default="results/audit/evaluator_plan.json")
    parser.add_argument("--output-md", default="results/audit/evaluator_plan.md")
    parser.add_argument("--output-csv", default="results/audit/evaluator_plan_by_group.csv")
    args = parser.parse_args()

    audit = audit_evaluator_plan(
        labels_path=Path(args.labels),
        schema_path=Path(args.schema),
        task_index_path=Path(args.task_index) if args.task_index else None,
        batch_index_path=Path(args.batch_index) if args.batch_index else None,
        batch_dir=Path(args.batch_dir) if args.batch_dir else None,
        output_json=Path(args.output_json) if args.output_json else None,
        output_markdown=Path(args.output_md) if args.output_md else None,
        output_csv=Path(args.output_csv) if args.output_csv else None,
    )
    print(
        json.dumps(
            {
                "overall_pass": audit["overall_pass"],
                "failed_gates": [name for name, gate in audit["gates"].items() if not gate["pass"]],
                "label_rows": audit["label_summary"]["rows"],
                "task_index_rows": audit["task_index_summary"].get("rows", 0),
                "batch_rows": audit["batch_summary"].get("rows", 0),
                "inspected_batch_rows": audit["batch_summary"].get("inspected_rows", 0),
                "output_json": args.output_json,
                "output_md": args.output_md,
                "output_csv": args.output_csv,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
