#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.report_grids import make_report_grids


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic random/example image grids for the MATS20 report.")
    parser.add_argument("--labels", default="data/mats20/filled_labels_qwen_proxy_sscd_recovered_claim_ready.csv")
    parser.add_argument("--config", default="config/mats20_experiment.json")
    parser.add_argument("--copy-scores", default="results/mats20/auto_proxy_recovered/membench_copy_scores.csv")
    parser.add_argument("--output-dir", default="results/mats20/qwen_proxy_sscd_recovered_claim_ready/grids")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--model-scope", choices=["primary", "secondary", "all"], default="primary")
    args = parser.parse_args()

    summary = make_report_grids(
        labels_path=Path(args.labels),
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        copy_scores_path=Path(args.copy_scores) if args.copy_scores else None,
        seed=args.seed,
        model_scope=args.model_scope,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
