#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.image_proxies import download_membench_references


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate selected MemBench reference images for copy-detection scoring.")
    parser.add_argument("--prompt-bank", default="data/mats20/prompt_bank.jsonl")
    parser.add_argument("--output-dir", default="benchmarks/MemBench/references_mats20")
    parser.add_argument("--index-json", default="data/mats20/membench_reference_index.json")
    parser.add_argument("--index-csv", default="data/mats20/membench_reference_index.csv")
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--max-bytes", type=int, default=20_000_000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    result = download_membench_references(
        prompt_bank_path=Path(args.prompt_bank),
        output_dir=Path(args.output_dir),
        index_json_path=Path(args.index_json),
        index_csv_path=Path(args.index_csv) if args.index_csv else None,
        timeout_s=args.timeout_s,
        max_bytes=args.max_bytes,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
