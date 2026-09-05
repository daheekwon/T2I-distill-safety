#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.analysis import run_full_analysis
from t2i_distill.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run behavioral inheritance analyses from label CSV.")
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", default="results/inheritance_analysis")
    parser.add_argument("--model-scope", choices=["primary", "secondary", "all"], default="primary")
    args = parser.parse_args()
    summary = run_full_analysis(
        Path(args.labels),
        load_config(Path(args.config)),
        Path(args.output_dir),
        model_scope=args.model_scope,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
