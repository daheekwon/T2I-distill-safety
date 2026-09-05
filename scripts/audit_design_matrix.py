#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.config import load_config
from t2i_distill.design_audit import audit_design_matrix, write_design_audit_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether the prompt/manifest/label design matrix supports every planned RQ before expensive generation.")
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--prompt-bank", default="data/prompts/inheritance_prompt_bank.jsonl")
    parser.add_argument("--manifest", default="data/manifests/generation_manifest.jsonl")
    parser.add_argument("--labels", default="data/labels/label_template.csv")
    parser.add_argument("--shard-index", default="data/manifests/shards_primary/index.json")
    parser.add_argument("--output-json", default="results/audit/design_matrix.json")
    parser.add_argument("--output-md", default="results/audit/design_matrix.md")
    parser.add_argument("--model-scope", choices=["primary", "secondary", "all"], default="primary")
    parser.add_argument("--include-gated", action="store_true")
    args = parser.parse_args()

    audit = audit_design_matrix(
        config=load_config(Path(args.config)),
        prompt_bank_path=Path(args.prompt_bank),
        manifest_path=Path(args.manifest),
        labels_path=Path(args.labels),
        shard_index_path=Path(args.shard_index) if args.shard_index else None,
        model_scope=args.model_scope,
        include_gated=args.include_gated,
    )
    write_design_audit_reports(audit, Path(args.output_json), Path(args.output_md) if args.output_md else None)
    print(
        json.dumps(
            {
                "overall_pass": audit["overall_pass"],
                "failed_gates": [name for name, gate in audit["gates"].items() if not gate["pass"]],
                "prompt_records": audit["prompt_design"]["records"],
                "generation_jobs": audit["manifest_design"]["jobs"],
                "label_rows": audit["label_design"]["rows"],
                "rq3_axis_target_bridges": audit["rq3_dissociation_bridge"]["axis_target_bridge_count"],
                "output_json": args.output_json,
                "output_md": args.output_md,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
