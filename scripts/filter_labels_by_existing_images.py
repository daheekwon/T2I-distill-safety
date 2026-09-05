#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.label_filter import filter_labels_by_existing_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only label rows whose image_path exists.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--allow-empty-images", action="store_true")
    args = parser.parse_args()

    summary = filter_labels_by_existing_images(
        Path(args.labels),
        Path(args.output),
        summary_path=Path(args.summary_json) if args.summary_json else None,
        require_nonempty=not args.allow_empty_images,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
