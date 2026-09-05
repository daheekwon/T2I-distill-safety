#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.label_filter import filter_labels_by_required_signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only label rows whose benchmark-specific required signal columns are filled.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()
    summary = filter_labels_by_required_signals(
        Path(args.labels),
        Path(args.output),
        summary_path=Path(args.summary_json) if args.summary_json else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
