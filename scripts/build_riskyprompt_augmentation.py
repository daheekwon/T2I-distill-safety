#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t2i_distill.config import load_config
from t2i_distill.io import iter_jsonl, stable_hash, write_csv, write_json, write_jsonl
from t2i_distill.manifest import build_generation_manifest, build_label_template, summarize_jobs
from t2i_distill.mats20 import (
    SELECTION_FIELDS,
    _add_selection,
    _candidate_groups,
    _record_category,
    _representative_group,
    _round_robin_select,
    _safe_for_mats20,
)
from t2i_distill.prompt_bank import iter_prompt_records


FAMILY_MANIFESTS = {
    "gpu0_sdxl": {
        "device": "cuda:0",
        "models": {"sdxl_base", "sdxl_lightning_4step"},
    },
    "gpu3_sd35": {
        "device": "cuda:3",
        "models": {"sd35_large", "sd35_large_turbo"},
    },
    "gpu2_flux": {
        "device": "cuda:2",
        "models": {"flux2_klein_base_4b", "flux2_klein_4b"},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a one-hour T2I-RiskyPrompt augmentation package.")
    parser.add_argument("--config", default="config/mats20_experiment.json")
    parser.add_argument("--benchmarks-root", default="benchmarks")
    parser.add_argument("--base-prompt-bank", default="data/mats20/prompt_bank.jsonl")
    parser.add_argument("--output-dir", default="data/mats20_aug/riskyprompt_1h")
    parser.add_argument("--image-output-root", default="outputs/mats20_aug/riskyprompt_1h/images")
    parser.add_argument("--n-prompts", type=int, default=24)
    args = parser.parse_args()

    config = load_config(args.config)
    config = dict(config)
    config["image_output_root"] = args.image_output_root

    existing_uids = {str(row.get("image_prompt_uid", "")) for row in iter_jsonl(Path(args.base_prompt_bank))}
    all_records = [
        row
        for row in iter_prompt_records(config, Path(args.benchmarks_root), benchmarks={"t2i_riskyprompt"})
        if row.get("condition_type") == "risky_safety"
        and str(row.get("image_prompt_uid", "")) not in existing_uids
        and _safe_for_mats20(row)
    ]
    candidates = _candidate_groups(all_records, "t2i_riskyprompt", "risky_safety")
    chosen = _round_robin_select(candidates, args.n_prompts, group_fn=_representative_group)

    selected: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for uid, rows in chosen:
        _add_selection(selected, audit_rows, uid, rows, "one-hour T2I-RiskyPrompt safety augmentation")

    selected_uid_set = set(selected)
    selected_prompt_rows = [row for row in all_records if row.get("image_prompt_uid") in selected_uid_set]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_bank = output_dir / "prompt_bank.jsonl"
    manifest = output_dir / "generation_manifest.jsonl"
    label_template = output_dir / "label_template.csv"
    write_jsonl(prompt_bank, selected_prompt_rows)
    write_csv(output_dir / "selection_index.csv", audit_rows, SELECTION_FIELDS)
    manifest_summary = build_generation_manifest(
        config,
        prompt_bank,
        manifest,
        model_scope="primary",
        include_gated=False,
        benchmark_filter={"t2i_riskyprompt"},
        job_order="model_major",
    )
    label_summary = build_label_template(prompt_bank, manifest, label_template)
    family_outputs = _write_family_manifests(manifest, output_dir)
    summary = {
        "output_dir": str(output_dir),
        "base_prompt_bank": args.base_prompt_bank,
        "image_output_root": args.image_output_root,
        "selected_unique_image_prompts": len(selected_uid_set),
        "selected_prompt_bank_rows": len(selected_prompt_rows),
        "selection_by_axis": dict(sorted(Counter(row["axis_id"] for row in audit_rows).items())),
        "selection_by_category": dict(sorted(Counter(_record_category(row) for row in selected_prompt_rows).items())),
        "prompt_bank": str(prompt_bank),
        "manifest": str(manifest),
        "label_template": str(label_template),
        "manifest_summary": manifest_summary,
        "label_summary": label_summary,
        "family_manifests": family_outputs,
    }
    write_json(output_dir / "augmentation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _write_family_manifests(manifest: Path, output_dir: Path) -> dict[str, Any]:
    jobs = list(iter_jsonl(manifest))
    out: dict[str, Any] = {}
    for name, spec in FAMILY_MANIFESTS.items():
        models = set(spec["models"])
        device = str(spec["device"])
        family_jobs = [{**job, "device": device} for job in jobs if job.get("model_id") in models]
        path = output_dir / f"generation_manifest_{name}.jsonl"
        write_jsonl(path, family_jobs)
        out[name] = {
            "path": str(path),
            "device": device,
            "models": sorted(models),
            "jobs": len(family_jobs),
            "summary": summarize_jobs(family_jobs),
            "run_id": stable_hash(path, len(family_jobs), device),
        }
    return out


if __name__ == "__main__":
    main()
