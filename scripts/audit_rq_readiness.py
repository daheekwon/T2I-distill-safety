#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.config import load_config
from t2i_distill.readiness import audit_rq_readiness, write_rq_readiness_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit which research questions are ready to interpret from a label CSV.")
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--labels", default="data/labels/label_template.csv")
    parser.add_argument("--model-scope", choices=["primary", "secondary", "all"], default="primary")
    parser.add_argument("--allow-partial", action="store_true", help="Mark RQs ready when they have any matched ready cells, instead of requiring full completion.")
    parser.add_argument("--output-json", default="results/audit/rq_readiness.json")
    parser.add_argument("--output-md", default="results/audit/rq_readiness.md")
    args = parser.parse_args()

    readiness = audit_rq_readiness(
        Path(args.labels),
        load_config(Path(args.config)),
        model_scope=args.model_scope,
        strict_complete=not args.allow_partial,
    )
    write_rq_readiness_reports(readiness, Path(args.output_json), Path(args.output_md) if args.output_md else None)
    compact = {
        "labels": args.labels,
        "overall_ready": readiness["overall_ready"],
        "rq_summary": {
            rq: {
                "ready": summary["ready"],
                "matched_ready_cells": summary["matched_ready_cells"],
                "expected_matched_cells": summary["expected_matched_cells"],
                "status_counts": summary["status_counts"],
            }
            for rq, summary in readiness["rq_summary"].items()
        },
        "output_json": args.output_json,
        "output_md": args.output_md,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
