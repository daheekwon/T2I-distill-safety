#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.config import load_config
from t2i_distill.safety_mem_analysis import run_safety_memorization_analysis


def _thresholds(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    return values or (0.3, 0.5, 0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused teacher-student inheritance analysis for safety/refusal and MemBench copy-risk labels.")
    parser.add_argument("--config", default="config/mats20_experiment.json")
    parser.add_argument("--labels", default="data/mats20/filled_labels_qwen_proxy_sscd_claim_ready.csv")
    parser.add_argument("--full-labels", default="data/mats20/filled_labels_qwen_proxy_sscd.csv")
    parser.add_argument("--output-dir", default="results/mats20/qwen_proxy_sscd_claim_ready/focused_safety_mem")
    parser.add_argument("--model-scope", choices=["primary", "secondary", "all"], default="primary")
    parser.add_argument("--mem-thresholds", default="0.3,0.5,0.8")
    args = parser.parse_args()
    summary = run_safety_memorization_analysis(
        Path(args.labels),
        load_config(Path(args.config)),
        Path(args.output_dir),
        full_labels_path=Path(args.full_labels) if args.full_labels else None,
        model_scope=args.model_scope,
        mem_thresholds=_thresholds(args.mem_thresholds),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
