#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.power import audit_statistical_power_from_path, write_power_audit_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate benchmark-level design resolution and conservative MDE from design_matrix.json.")
    parser.add_argument("--design-audit", default="results/audit/design_matrix.json")
    parser.add_argument("--output-json", default="results/audit/statistical_power.json")
    parser.add_argument("--output-md", default="results/audit/statistical_power.md")
    parser.add_argument("--output-csv", default="results/audit/statistical_power.csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.80)
    parser.add_argument("--baseline-rate", type=float, default=0.50)
    parser.add_argument("--min-matched-cells", type=int, default=100)
    parser.add_argument("--included-rqs", default="", help="Comma-separated RQs to include in the power gate. Default includes every non-unlearning design RQ.")
    args = parser.parse_args()

    audit = audit_statistical_power_from_path(
        Path(args.design_audit),
        alpha=args.alpha,
        target_power=args.target_power,
        baseline_rate=args.baseline_rate,
        min_matched_cells=args.min_matched_cells,
        included_rqs={part.strip() for part in args.included_rqs.split(",") if part.strip()} or None,
    )
    write_power_audit_reports(audit, Path(args.output_json), Path(args.output_md) if args.output_md else None, Path(args.output_csv) if args.output_csv else None)
    print(
        json.dumps(
            {
                "overall_pass": audit["overall_pass"],
                "failed_gates": [name for name, gate in audit["gates"].items() if not gate["pass"]],
                "benchmark_test_count": audit["benchmark_test_count"],
                "bonferroni_alpha": audit["bonferroni_alpha"],
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
