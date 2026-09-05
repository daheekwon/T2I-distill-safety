#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.config import load_config
from t2i_distill.reporting import build_claim_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a claim-level research report from post-label audit and analysis outputs.")
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--result-dir", default="results/post_label_pipeline")
    parser.add_argument("--output-json", default="", help="Defaults to <result-dir>/claim_report.json")
    parser.add_argument("--output-md", default="", help="Defaults to <result-dir>/claim_report.md")
    parser.add_argument("--model-scope", choices=["primary", "secondary", "all"], default="primary")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_json = Path(args.output_json) if args.output_json else result_dir / "claim_report.json"
    output_md = Path(args.output_md) if args.output_md else result_dir / "claim_report.md"
    report = build_claim_report(
        result_dir=result_dir,
        config=load_config(Path(args.config)),
        output_json=output_json,
        output_markdown=output_md,
        model_scope=args.model_scope,
    )
    print(
        json.dumps(
            {
                "overall_status": report["overall_status"],
                "output_json": str(output_json),
                "output_md": str(output_md),
                "action_items": report["action_items"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
