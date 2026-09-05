#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.analysis import run_full_analysis
from t2i_distill.config import load_config
from t2i_distill.design_audit import audit_design_matrix, write_design_audit_reports
from t2i_distill.evaluation import merge_label_updates, validate_label_updates
from t2i_distill.label_quality import audit_label_quality
from t2i_distill.execution import audit_experiment_state, write_audit_reports
from t2i_distill.power import audit_statistical_power, write_power_audit_reports
from t2i_distill.readiness import audit_rq_readiness, write_rq_readiness_reports
from t2i_distill.reporting import build_claim_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate evaluator updates, merge labels, audit readiness, and run analysis.")
    parser.add_argument("updates", nargs="*", help="Evaluator update files, JSONL or CSV. Omit when --labels already points to a filled CSV.")
    parser.add_argument("--config", default="config/experiment.json")
    parser.add_argument("--prompt-bank", default="data/prompts/inheritance_prompt_bank.jsonl")
    parser.add_argument("--manifest", default="data/manifests/generation_manifest.jsonl")
    parser.add_argument("--label-template", default="data/labels/label_template.csv")
    parser.add_argument("--labels", default="", help="Existing filled labels. If empty and updates are provided, writes --merged-labels.")
    parser.add_argument("--design-labels", default="", help="Optional full-coverage label CSV for design/shard audit when --labels is a claim-ready subset.")
    parser.add_argument("--merged-labels", default="data/labels/filled_labels.csv")
    parser.add_argument("--output-dir", default="results/post_label_pipeline")
    parser.add_argument("--shard-index", default="data/manifests/shards_primary/index.json")
    parser.add_argument("--model-scope", choices=["primary", "secondary", "all"], default="primary")
    parser.add_argument("--allow-partial-rq", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--skip-claim-report", action="store_true")
    parser.add_argument("--skip-design-audit", action="store_true")
    parser.add_argument("--skip-power-audit", action="store_true")
    parser.add_argument("--skip-label-quality-audit", action="store_true")
    parser.add_argument("--strict-design", action="store_true")
    parser.add_argument("--strict-power", action="store_true")
    parser.add_argument("--power-min-matched-cells", type=int, default=100)
    parser.add_argument("--strict-label-quality", action="store_true")
    parser.add_argument("--min-semantic-confidence", type=float, default=0.60)
    parser.add_argument("--skip-update-validation", action="store_true")
    parser.add_argument("--strict-merge", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))

    update_summaries = []
    update_paths = [Path(path) for path in args.updates]
    if update_paths and not args.skip_update_validation:
        for path in update_paths:
            summary = validate_label_updates(path)
            update_summaries.append(summary)
            if not summary["valid"]:
                (output_dir / "invalid_updates.json").write_text(json.dumps(update_summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                raise SystemExit(f"invalid update file: {path}")

    if args.labels:
        labels_path = Path(args.labels)
        merge_summary: dict[str, Any] | None = None
    elif update_paths:
        merge_summary = merge_label_updates(Path(args.label_template), update_paths, Path(args.merged_labels), strict=args.strict_merge)
        labels_path = Path(args.merged_labels)
    else:
        labels_path = Path(args.label_template)
        merge_summary = None

    design_audit = None
    if not args.skip_design_audit:
        shard_index_path = Path(args.shard_index) if args.shard_index else None
        design_labels_path = Path(args.design_labels) if args.design_labels else labels_path
        design_audit = audit_design_matrix(
            config=config,
            prompt_bank_path=Path(args.prompt_bank),
            manifest_path=Path(args.manifest),
            labels_path=design_labels_path,
            shard_index_path=shard_index_path if shard_index_path and shard_index_path.exists() else None,
            model_scope=args.model_scope,
        )
        design_audit["source_path"] = str(output_dir / "design_matrix.json")
        write_design_audit_reports(design_audit, output_dir / "design_matrix.json", output_dir / "design_matrix.md")
        if args.strict_design and not design_audit["overall_pass"]:
            failed = [name for name, gate in design_audit["gates"].items() if not gate["pass"]]
            raise SystemExit(f"design audit failed: {failed}")

    power_audit = None
    if design_audit and not args.skip_power_audit:
        included_rqs = set(config.get("analysis", {}).get("required_primary_rqs") or []) or None
        power_audit = audit_statistical_power(design_audit, min_matched_cells=args.power_min_matched_cells, included_rqs=included_rqs)
        write_power_audit_reports(
            power_audit,
            output_dir / "statistical_power.json",
            output_dir / "statistical_power.md",
            output_dir / "statistical_power.csv",
        )
        if args.strict_power and not power_audit["overall_pass"]:
            failed = [name for name, gate in power_audit["gates"].items() if not gate["pass"]]
            raise SystemExit(f"statistical power audit failed: {failed}")

    label_quality_audit = None
    if not args.skip_label_quality_audit:
        min_quality = float(config.get("analysis", {}).get("quality_controls", {}).get("minimum_quality_score", 3.0))
        label_quality_audit = audit_label_quality(
            labels_path,
            output_json=output_dir / "label_quality.json",
            output_markdown=output_dir / "label_quality.md",
            output_csv=output_dir / "label_quality_by_group.csv",
            update_paths=update_paths,
            min_quality=min_quality,
            min_semantic_confidence=args.min_semantic_confidence,
        )
        if args.strict_label_quality and not label_quality_audit["overall_pass"]:
            failed = [name for name, gate in label_quality_audit["gates"].items() if not gate["pass"]]
            raise SystemExit(f"label quality audit failed: {failed}")

    experiment_audit = audit_experiment_state(
        config=config,
        prompt_bank_path=Path(args.prompt_bank),
        manifest_path=Path(args.manifest),
        labels_path=labels_path,
    )
    write_audit_reports(experiment_audit, output_dir / "experiment_state.json", output_dir / "experiment_state.md")

    rq_readiness = audit_rq_readiness(labels_path, config, model_scope=args.model_scope, strict_complete=not args.allow_partial_rq)
    write_rq_readiness_reports(rq_readiness, output_dir / "rq_readiness.json", output_dir / "rq_readiness.md")

    analysis_summary = None
    if not args.skip_analysis:
        analysis_summary = run_full_analysis(labels_path, config, output_dir / "analysis", model_scope=args.model_scope)

    claim_report_summary = None

    summary = {
        "labels_path": str(labels_path),
        "design_labels_path": str(Path(args.design_labels)) if args.design_labels else str(labels_path),
        "update_files": [str(path) for path in update_paths],
        "update_validation": update_summaries,
        "merge": merge_summary,
        "design_overall_pass": design_audit["overall_pass"] if design_audit else None,
        "design_failed_gates": [name for name, gate in design_audit["gates"].items() if not gate["pass"]] if design_audit else [],
        "power_overall_pass": power_audit["overall_pass"] if power_audit else None,
        "power_min_matched_cells": args.power_min_matched_cells,
        "power_failed_gates": [name for name, gate in power_audit["gates"].items() if not gate["pass"]] if power_audit else [],
        "label_quality_overall_pass": label_quality_audit["overall_pass"] if label_quality_audit else None,
        "label_quality_failed_gates": [name for name, gate in label_quality_audit["gates"].items() if not gate["pass"]] if label_quality_audit else [],
        "experiment_gates": {key: value["pass"] for key, value in experiment_audit["gates"].items()},
        "rq_overall_ready": rq_readiness["overall_ready"],
        "rq_summary": {
            rq: {
                "ready": item["ready"],
                "matched_ready_cells": item["matched_ready_cells"],
                "expected_matched_cells": item["expected_matched_cells"],
                "status_counts": item["status_counts"],
            }
            for rq, item in rq_readiness["rq_summary"].items()
        },
        "analysis_summary": analysis_summary,
        "output_dir": str(output_dir),
    }
    (output_dir / "post_label_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.skip_claim_report:
        claim_report = build_claim_report(
            result_dir=output_dir,
            config=config,
            output_json=output_dir / "claim_report.json",
            output_markdown=output_dir / "claim_report.md",
            model_scope=args.model_scope,
        )
        claim_report_summary = {
            "overall_status": claim_report["overall_status"],
            "output_json": str(output_dir / "claim_report.json"),
            "output_md": str(output_dir / "claim_report.md"),
        }
        summary["claim_report"] = claim_report_summary
        (output_dir / "post_label_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    printable = {
        "labels_path": summary["labels_path"],
        "update_files": summary["update_files"],
        "design_labels_path": summary["design_labels_path"],
        "power_min_matched_cells": summary["power_min_matched_cells"],
        "design_overall_pass": summary["design_overall_pass"],
        "design_failed_gates": summary["design_failed_gates"],
        "power_overall_pass": summary["power_overall_pass"],
        "power_failed_gates": summary["power_failed_gates"],
        "label_quality_overall_pass": summary["label_quality_overall_pass"],
        "label_quality_failed_gates": summary["label_quality_failed_gates"],
        "experiment_gates": summary["experiment_gates"],
        "rq_overall_ready": summary["rq_overall_ready"],
        "rq_ready": {rq: item["ready"] for rq, item in summary["rq_summary"].items()},
        "analysis_usable_label_rows": analysis_summary["usable_label_rows"] if analysis_summary else None,
        "claim_report_status": claim_report_summary["overall_status"] if claim_report_summary else None,
        "output_dir": summary["output_dir"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
