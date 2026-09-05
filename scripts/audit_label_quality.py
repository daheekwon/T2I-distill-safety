#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.config import load_config
from t2i_distill.label_quality import audit_label_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final label completeness, value validity, and optional multi-evaluator update reliability.")
    parser.add_argument("updates", nargs="*", help="Optional evaluator update files to check for overlap agreement/conflicts.")
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--labels", default="data/labels/label_template.csv")
    parser.add_argument("--output-json", default="results/audit/label_quality.json")
    parser.add_argument("--output-md", default="results/audit/label_quality.md")
    parser.add_argument("--output-csv", default="results/audit/label_quality_by_group.csv")
    parser.add_argument("--min-semantic-confidence", type=float, default=0.60)
    parser.add_argument("--numeric-disagreement-threshold", type=float, default=0.25)
    parser.add_argument("--quality-disagreement-threshold", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    min_quality = float(config.get("analysis", {}).get("quality_controls", {}).get("minimum_quality_score", 3.0))
    audit = audit_label_quality(
        Path(args.labels),
        output_json=Path(args.output_json),
        output_markdown=Path(args.output_md) if args.output_md else None,
        output_csv=Path(args.output_csv) if args.output_csv else None,
        update_paths=[Path(path) for path in args.updates],
        min_quality=min_quality,
        min_semantic_confidence=args.min_semantic_confidence,
        numeric_disagreement_threshold=args.numeric_disagreement_threshold,
        quality_disagreement_threshold=args.quality_disagreement_threshold,
    )
    print(
        json.dumps(
            {
                "overall_pass": audit["overall_pass"],
                "failed_or_waiting_gates": [name for name, gate in audit["gates"].items() if not gate["pass"]],
                "rows": audit["final_label_quality"]["rows"],
                "filled_rows": audit["final_label_quality"]["filled_rows"],
                "overlap_label_uids": audit["update_reliability"]["overlap_label_uids"],
                "conflicting_update_cells": audit["update_reliability"]["conflicting_update_cells"],
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
