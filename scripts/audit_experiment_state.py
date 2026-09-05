#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.config import load_config
from t2i_distill.execution import audit_experiment_state, write_audit_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit prompt, manifest, image, and label readiness for the inheritance experiment.")
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--prompt-bank", default="data/prompts/inheritance_prompt_bank.jsonl")
    parser.add_argument("--manifest", default="data/manifests/generation_manifest.jsonl")
    parser.add_argument("--labels", default="data/labels/label_template.csv")
    parser.add_argument("--output-json", default="results/audit/experiment_state.json")
    parser.add_argument("--output-md", default="results/audit/experiment_state.md")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--image-limit", type=int, default=None)
    args = parser.parse_args()

    labels = Path(args.labels) if args.labels else None
    audit = audit_experiment_state(
        config=load_config(Path(args.config)),
        prompt_bank_path=Path(args.prompt_bank),
        manifest_path=Path(args.manifest),
        labels_path=labels,
        check_images=args.check_images,
        image_verify=args.verify_images,
        image_limit=args.image_limit,
    )
    write_audit_reports(audit, Path(args.output_json), Path(args.output_md) if args.output_md else None)
    compact = {
        "prompt_records": audit["prompt_bank"]["records"],
        "manifest_jobs": audit["manifest"]["jobs"],
        "labels_rows": audit["labels"]["rows"] if audit.get("labels") else None,
        "gates": {key: value["pass"] for key, value in audit["gates"].items()},
        "output_json": args.output_json,
        "output_md": args.output_md,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
