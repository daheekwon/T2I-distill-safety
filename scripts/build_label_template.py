#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.manifest import build_label_template


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSV template for VLM/manual labels.")
    parser.add_argument("--prompt-bank", default="data/prompts/inheritance_prompt_bank.jsonl")
    parser.add_argument("--manifest", default="data/manifests/generation_manifest.jsonl")
    parser.add_argument("--output", default="data/labels/label_template.csv")
    parser.add_argument("--include-gated", action="store_true")
    args = parser.parse_args()
    summary = build_label_template(
        Path(args.prompt_bank),
        Path(args.manifest),
        Path(args.output),
        include_gated=args.include_gated,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
