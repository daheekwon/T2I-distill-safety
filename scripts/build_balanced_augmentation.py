#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
    _round_robin_select,
    _safe_for_mats20,
)
from t2i_distill.prompt_bank import iter_prompt_records, make_record


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

CONTROL_PROMPT_RULES = (
    ("pillow", "a generic product photo of a square throw pillow on a neutral sofa"),
    ("backpack", "a generic product photo of a hiking backpack shown from the front"),
    ("pack", "a generic product photo of a hiking backpack shown from the front"),
    ("rug", "a generic product photo of a patterned area rug on a plain floor"),
    ("canvas", "a generic abstract canvas print hanging on a white wall"),
    ("blanket", "a generic product photo of a soft throw blanket folded on a sofa"),
    ("flag", "a generic product photo of a small garden flag on a plain background"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a one-hour balanced safety/memorization augmentation package.")
    parser.add_argument("--config", default="config/mats20_experiment.json")
    parser.add_argument("--benchmarks-root", default="benchmarks")
    parser.add_argument(
        "--exclude-prompt-banks",
        default="data/mats20/prompt_bank.jsonl,data/mats20_aug/riskyprompt_1h/prompt_bank.jsonl",
    )
    parser.add_argument("--reference-index", default="data/mats20/membench_reference_index_recovered.json")
    parser.add_argument("--output-dir", default="data/mats20_aug/balanced_1h")
    parser.add_argument("--image-output-root", default="outputs/mats20_aug/balanced_1h/images")
    parser.add_argument("--risk-prompts", type=int, default=6)
    parser.add_argument("--overt-benign-prompts", type=int, default=3)
    parser.add_argument("--overt-risky-prompts", type=int, default=3)
    parser.add_argument("--t2isafety-prompts", type=int, default=4)
    parser.add_argument("--membench-items", type=int, default=4)
    parser.add_argument("--membench-seed", type=int, default=2)
    args = parser.parse_args()

    base_config = load_config(args.config)
    config = copy.deepcopy(base_config)
    config["image_output_root"] = args.image_output_root
    config["seed_sets"]["single_full"] = [0]
    config["seed_sets"]["matched_2"] = [args.membench_seed]

    exclude_uids = _load_excluded_uids(args.exclude_prompt_banks)
    all_records = [
        row
        for row in iter_prompt_records(
            config,
            Path(args.benchmarks_root),
            benchmarks={"t2i_riskyprompt", "overt", "t2isafety", "membench"},
        )
        if _safe_for_mats20(row)
    ]

    selected: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    _select_standard_prompts(
        all_records,
        selected,
        audit_rows,
        exclude_uids=exclude_uids,
        benchmark="t2i_riskyprompt",
        condition_type="risky_safety",
        n_prompts=args.risk_prompts,
        reason="balanced augmentation: unsafe behavior inheritance",
    )
    _select_standard_prompts(
        all_records,
        selected,
        audit_rows,
        exclude_uids=exclude_uids,
        benchmark="overt",
        condition_type="benign_overrefusal",
        n_prompts=args.overt_benign_prompts,
        reason="balanced augmentation: OVERT benign over-refusal",
    )
    _select_standard_prompts(
        all_records,
        selected,
        audit_rows,
        exclude_uids=exclude_uids,
        benchmark="overt",
        condition_type="unsafe_safety",
        n_prompts=args.overt_risky_prompts,
        reason="balanced augmentation: OVERT risky under-refusal",
    )
    _select_standard_prompts(
        all_records,
        selected,
        audit_rows,
        exclude_uids=exclude_uids,
        benchmark="t2isafety",
        condition_type="safety_native",
        n_prompts=args.t2isafety_prompts,
        reason="balanced augmentation: T2ISafety native safety",
    )

    mem_records, mem_audit = _select_membench_pairs(
        all_records,
        Path(args.reference_index),
        n_items=args.membench_items,
    )
    for record in mem_records:
        selected.setdefault(str(record["image_prompt_uid"]), record)
    audit_rows.extend(mem_audit)

    selected_uid_set = set(selected)
    selected_prompt_rows = [row for row in all_records if row.get("image_prompt_uid") in selected_uid_set]
    selected_prompt_rows.extend(row for row in mem_records if row["condition_type"] == "memorization_control")
    selected_prompt_rows = _dedupe_prompt_rows(selected_prompt_rows)

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
        benchmark_filter={"t2i_riskyprompt", "overt", "t2isafety", "membench"},
        job_order="model_major",
    )
    label_summary = build_label_template(prompt_bank, manifest, label_template)
    family_outputs = _write_family_manifests(manifest, output_dir)

    summary = {
        "output_dir": str(output_dir),
        "image_output_root": args.image_output_root,
        "exclude_prompt_banks": [str(path) for path in _split_paths(args.exclude_prompt_banks)],
        "reference_index": args.reference_index,
        "selected_unique_image_prompts": len({row["image_prompt_uid"] for row in selected_prompt_rows}),
        "selected_prompt_bank_rows": len(selected_prompt_rows),
        "selection_by_condition": dict(sorted(Counter(f"{row['benchmark']}::{row['condition_type']}" for row in audit_rows).items())),
        "selection_by_category": dict(sorted(Counter(f"{row['benchmark']}::{row['condition_type']}::{row['category']}" for row in audit_rows).items())),
        "membench_seed": args.membench_seed,
        "prompt_bank": str(prompt_bank),
        "manifest": str(manifest),
        "label_template": str(label_template),
        "manifest_summary": manifest_summary,
        "label_summary": label_summary,
        "family_manifests": family_outputs,
    }
    write_json(output_dir / "augmentation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _select_standard_prompts(
    records: list[dict[str, Any]],
    selected: dict[str, dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    *,
    exclude_uids: set[str],
    benchmark: str,
    condition_type: str,
    n_prompts: int,
    reason: str,
) -> None:
    candidates = {
        uid: rows
        for uid, rows in _candidate_groups(records, benchmark, condition_type).items()
        if uid not in exclude_uids and uid not in selected
    }
    chosen = _round_robin_select(candidates, n_prompts, group_fn=_representative_group)
    if len(chosen) < n_prompts:
        raise SystemExit(f"not enough candidates for {benchmark}/{condition_type}: {len(chosen)} < {n_prompts}")
    for uid, rows in chosen:
        _add_selection(selected, audit_rows, uid, rows, reason)


def _select_membench_pairs(
    records: list[dict[str, Any]],
    reference_index: Path,
    *,
    n_items: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    referenced_eval_uids = _reference_eval_uids(reference_index)
    triggers_by_eval = {
        str(row["eval_uid"]): row
        for row in records
        if row.get("benchmark") == "membench"
        and row.get("condition_type") == "memorization_trigger"
        and str(row.get("eval_uid")) in referenced_eval_uids
        and _control_prompt_for(str(row.get("prompt_text", "")))
    }
    candidates = [(stable_hash("balanced_membench", eval_uid), eval_uid, row) for eval_uid, row in triggers_by_eval.items()]
    candidates.sort()
    chosen = candidates[:n_items]
    if len(chosen) < n_items:
        raise SystemExit(f"not enough MemBench reference-backed trigger items: {len(chosen)} < {n_items}")

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for _, eval_uid, trigger in chosen:
        trigger = dict(trigger)
        rows.append(trigger)
        audit_rows.append(_audit_row(trigger, "balanced augmentation: MemBench trigger with recovered reference"))
        control_prompt = _control_prompt_for(str(trigger["prompt_text"]))
        control = make_record(
            benchmark="membench",
            source_path=Path(str(trigger["source_path"])),
            source_row_id=f"{trigger['source_row_id']}:balanced_control",
            image_source_id=f"{trigger['source_row_id']}:balanced_control",
            prompt_text=control_prompt,
            axis_id="membench/paired_control",
            axis_name="paired_control",
            condition_type="memorization_control",
            analysis_layers=list(trigger.get("analysis_layers") or ["memorization_retention"]),
            complexity_level="memorization",
            source="balanced_membench_paired_control",
            target_behavior="not_memorized_reference",
            evaluator="SSCD paired-reference copy-detection similarity",
            metadata={
                "control_for_eval_uid": eval_uid,
                "control_for_image_prompt_uid": trigger["image_prompt_uid"],
                "original_trigger_prompt": trigger["prompt_text"],
            },
        )
        control["eval_uid"] = eval_uid
        rows.append(control)
        audit_rows.append(_audit_row(control, "balanced augmentation: MemBench paired neutral control"))
    return rows, audit_rows


def _reference_eval_uids(reference_index: Path) -> set[str]:
    data = json.loads(reference_index.read_text(encoding="utf-8"))
    out = set()
    for record in data.get("records", []):
        if str(record.get("status", "")).startswith("ok") and record.get("path"):
            out.add(str(record.get("eval_uid", "")))
    return out


def _control_prompt_for(prompt: str) -> str:
    text = prompt.lower()
    for needle, control in CONTROL_PROMPT_RULES:
        if needle in text:
            return control
    return ""


def _audit_row(row: dict[str, Any], reason: str) -> dict[str, str]:
    return {
        "image_prompt_uid": str(row.get("image_prompt_uid", "")),
        "benchmark": str(row.get("benchmark", "")),
        "condition_type": str(row.get("condition_type", "")),
        "axis_id": str(row.get("axis_id", "")),
        "category": _record_category(row),
        "reason": reason,
    }


def _representative_group(rows: list[dict[str, Any]]) -> str:
    categories = sorted({_record_category(row) for row in rows if _record_category(row)})
    if categories:
        return categories[0]
    axes = sorted({str(row.get("axis_id", "")) for row in rows if row.get("axis_id")})
    return axes[0] if axes else "ungrouped"


def _dedupe_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("eval_uid", "")), str(row.get("image_prompt_uid", "")), str(row.get("condition_type", "")))
        out.setdefault(key, row)
    return list(out.values())


def _load_excluded_uids(value: str) -> set[str]:
    out: set[str] = set()
    for path in _split_paths(value):
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            uid = str(row.get("image_prompt_uid", ""))
            if uid:
                out.add(uid)
    return out


def _split_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


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
