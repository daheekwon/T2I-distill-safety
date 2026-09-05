from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .config import selected_pairs, seeds_for_condition, unique_models
from .execution import LABEL_REQUIRED_FIELDS, MEMORIZATION_CONDITIONS, SAFETY_CONDITIONS
from .io import iter_jsonl, read_json, stable_hash, write_json
from .readiness import (
    BROAD_DIAGNOSTIC_CONDITIONS,
    EXPLICIT_DIAGNOSTIC_CONDITIONS,
    RQ_DESCRIPTIONS,
    RQ_ORDER,
    applicable_rqs,
)

REQUIRED_PRIMARY_RQS = [
    "rq1_sharpening_fit",
    "rq1_sharpening_transfer",
    "rq2_residual_transport",
    "rq3_preference_capability",
    "rq4_survival_principle",
    "safety_inheritance",
    "memorization_retention",
]

REQUIRED_SAFETY_CONDITIONS = {
    "benign_overrefusal",
    "risky_safety",
    "safety_counterfactual_benign",
    "safety_native",
    "social_bias_default",
    "unsafe_safety",
}
REQUIRED_MEMORIZATION_CONDITIONS = {"memorization_control", "memorization_trigger"}
OLD_PROMPT_MARKERS = ("h1_prompt", "h2_prompt", "h1/h2", "diversity_h1", "diversity_h2")


def audit_design_matrix(
    *,
    config: dict[str, Any],
    prompt_bank_path: Path,
    manifest_path: Path,
    labels_path: Path,
    shard_index_path: Path | None = None,
    model_scope: str = "primary",
    include_gated: bool = False,
) -> dict[str, Any]:
    expected_benchmarks = expected_enabled_benchmarks(config, include_gated=include_gated)
    expected_models = [model["model_id"] for model in unique_models(config, model_scope=model_scope)]
    prompt_summary = _audit_prompt_bank_design(prompt_bank_path, config, expected_benchmarks)
    manifest_summary, job_index = _audit_manifest_design(manifest_path, config, expected_models)
    label_summary, rq_summary, rq_pair_rows, rq3_bridge = _audit_label_design(
        labels_path,
        config,
        job_index,
        model_scope=model_scope,
    )
    shard_summary = _audit_shard_index(shard_index_path) if shard_index_path else None
    gates = _design_gates(
        config=config,
        expected_benchmarks=expected_benchmarks,
        expected_models=expected_models,
        prompt_summary=prompt_summary,
        manifest_summary=manifest_summary,
        label_summary=label_summary,
        rq_summary=rq_summary,
        rq3_bridge=rq3_bridge,
        shard_summary=shard_summary,
    )
    return {
        "prompt_bank_path": str(prompt_bank_path),
        "manifest_path": str(manifest_path),
        "labels_path": str(labels_path),
        "shard_index_path": str(shard_index_path) if shard_index_path else "",
        "model_scope": model_scope,
        "include_gated": include_gated,
        "expected_benchmarks": expected_benchmarks,
        "expected_models": expected_models,
        "overall_pass": all(gate["pass"] for gate in gates.values()),
        "gates": gates,
        "prompt_design": prompt_summary,
        "manifest_design": manifest_summary,
        "label_design": label_summary,
        "rq_design_summary": rq_summary,
        "rq_pair_design_rows": rq_pair_rows,
        "rq3_dissociation_bridge": rq3_bridge,
        "shard_design": shard_summary,
    }


def write_design_audit_reports(audit: dict[str, Any], output_json: Path, output_markdown: Path | None = None) -> None:
    write_json(output_json, audit)
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_design_audit_markdown(audit), encoding="utf-8")


def render_design_audit_markdown(audit: dict[str, Any]) -> str:
    lines = ["# Design Matrix Audit", ""]
    lines.append(f"- prompt bank: `{audit['prompt_bank_path']}`")
    lines.append(f"- manifest: `{audit['manifest_path']}`")
    lines.append(f"- labels: `{audit['labels_path']}`")
    lines.append(f"- model scope: `{audit['model_scope']}`")
    lines.append(f"- overall_pass: {audit['overall_pass']}")
    lines.append("")
    lines.append("## Gates")
    for name, gate in sorted(audit["gates"].items()):
        status = "PASS" if gate["pass"] else "FAIL"
        lines.append(f"- {status} `{name}`: {gate['detail']}")
    lines.append("")
    lines.append("## Scale")
    prompt = audit["prompt_design"]
    manifest = audit["manifest_design"]
    labels = audit["label_design"]
    lines.append(f"- prompt records: {prompt['records']}")
    lines.append(f"- unique prompt images: {prompt['unique_image_prompts']}")
    lines.append(f"- generation jobs: {manifest['jobs']}")
    lines.append(f"- expected label rows from manifest eval_uids: {manifest['expected_label_rows_from_manifest']}")
    lines.append(f"- label rows: {labels['rows']}")
    lines.append(f"- RQ3 bridge targets: {audit['rq3_dissociation_bridge']['axis_target_bridge_count']}")
    lines.append("")
    lines.append("## RQ Pair Coverage")
    lines.append("| rq | comparison_id | planned matched cells | benchmarks |")
    lines.append("| --- | --- | ---: | --- |")
    for row in audit["rq_pair_design_rows"]:
        by_benchmark = ", ".join(f"{key}:{value}" for key, value in row["planned_matched_cells_by_benchmark"].items())
        lines.append(f"| `{row['rq']}` | `{row['comparison_id']}` | {row['planned_matched_cells']} | {by_benchmark} |")
    lines.append("")
    return "\n".join(lines)


def expected_enabled_benchmarks(config: dict[str, Any], *, include_gated: bool = False) -> list[str]:
    names = []
    for benchmark, spec in sorted(config.get("benchmark_plan", {}).items()):
        if spec.get("enabled") is False:
            continue
        if spec.get("requires_valid_lineage") and not include_gated:
            continue
        names.append(benchmark)
    return names


def _audit_prompt_bank_design(path: Path, config: dict[str, Any], expected_benchmarks: list[str]) -> dict[str, Any]:
    by_benchmark: Counter[str] = Counter()
    by_condition: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    conditions_by_benchmark: dict[str, set[str]] = defaultdict(set)
    source_paths_seen: set[str] = set()
    image_uids: set[str] = set()
    eval_uids: set[str] = set()
    duplicate_eval_uids = 0
    old_prompt_records = 0
    gated_records = 0
    rows = 0
    for row in iter_jsonl(path):
        rows += 1
        benchmark = str(row.get("benchmark", ""))
        condition = str(row.get("condition_type", ""))
        source = str(row.get("source", ""))
        source_path = str(row.get("source_path", ""))
        by_benchmark[benchmark] += 1
        by_condition[condition] += 1
        by_source[source] += 1
        conditions_by_benchmark[benchmark].add(condition)
        if source_path:
            source_paths_seen.add(source_path)
        image_uids.add(str(row.get("image_prompt_uid", "")))
        eval_uid = str(row.get("eval_uid", ""))
        if eval_uid in eval_uids:
            duplicate_eval_uids += 1
        eval_uids.add(eval_uid)
        if row.get("requires_valid_lineage"):
            gated_records += 1
        marker_text = f"{source} {source_path}".lower()
        if any(marker in marker_text for marker in OLD_PROMPT_MARKERS):
            old_prompt_records += 1

    configured_source_paths = _configured_source_paths(config, expected_benchmarks)
    source_path_status = {
        source_path: Path(source_path).exists()
        for source_path in configured_source_paths
    }
    return {
        "records": rows,
        "unique_image_prompts": len(image_uids),
        "eval_uids": len(eval_uids),
        "duplicate_eval_uids": duplicate_eval_uids,
        "records_by_benchmark": dict(sorted(by_benchmark.items())),
        "records_by_condition": dict(sorted(by_condition.items())),
        "records_by_source": dict(sorted(by_source.items())),
        "conditions_by_benchmark": {key: sorted(value) for key, value in sorted(conditions_by_benchmark.items())},
        "source_paths_seen": sorted(source_paths_seen),
        "configured_source_path_status": source_path_status,
        "geneval2_records": by_benchmark.get("geneval2", 0),
        "requires_valid_lineage_records": gated_records,
        "old_diversity_prompt_records": old_prompt_records,
    }


def _audit_manifest_design(
    path: Path,
    config: dict[str, Any],
    expected_models: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    expected_device = str(config.get("default_device", ""))
    by_model: Counter[str] = Counter()
    by_benchmark: Counter[str] = Counter()
    by_condition: Counter[str] = Counter()
    by_benchmark_condition: Counter[str] = Counter()
    by_device: Counter[str] = Counter()
    unexpected_devices: Counter[str] = Counter()
    job_ids: set[str] = set()
    duplicate_job_ids = 0
    output_paths: set[str] = set()
    duplicate_output_paths = 0
    job_index: dict[str, dict[str, Any]] = {}
    seed_sets_by_model_prompt: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    model_switches = 0
    model_order: list[str] = []
    last_model = ""
    rows = 0
    expected_label_rows = 0

    for row in iter_jsonl(path):
        rows += 1
        model_id = str(row.get("model_id", ""))
        benchmark = str(row.get("benchmark", ""))
        condition = str(row.get("condition_type", ""))
        image_uid = str(row.get("image_prompt_uid", ""))
        seed = str(row.get("seed", ""))
        device = str(row.get("device", ""))
        eval_uids = [str(uid) for uid in row.get("eval_uids", [])]
        by_model[model_id] += 1
        by_benchmark[benchmark] += 1
        by_condition[condition] += 1
        by_benchmark_condition[f"{benchmark}/{condition}"] += 1
        by_device[device] += 1
        if expected_device and device != expected_device:
            unexpected_devices[device] += 1
        job_id = str(row.get("job_id", ""))
        if job_id in job_ids:
            duplicate_job_ids += 1
        job_ids.add(job_id)
        output_path = str(row.get("output_path", ""))
        if output_path in output_paths:
            duplicate_output_paths += 1
        output_paths.add(output_path)
        expected_label_rows += len(eval_uids)
        seed_sets_by_model_prompt[(model_id, benchmark, condition, image_uid)].add(seed)
        if model_id != last_model:
            model_order.append(model_id)
            if last_model:
                model_switches += 1
            last_model = model_id
        job_index[job_id] = {
            "model_id": model_id,
            "benchmark": benchmark,
            "condition_type": condition,
            "image_prompt_uid": image_uid,
            "seed": seed,
            "output_path": output_path,
            "eval_uids": set(eval_uids),
        }

    seed_policy = _summarize_seed_policy(seed_sets_by_model_prompt, config)
    return (
        {
            "jobs": rows,
            "jobs_by_model": dict(sorted(by_model.items())),
            "jobs_by_benchmark": dict(sorted(by_benchmark.items())),
            "jobs_by_condition": dict(sorted(by_condition.items())),
            "jobs_by_benchmark_condition": dict(sorted(by_benchmark_condition.items())),
            "jobs_by_device": dict(sorted(by_device.items())),
            "unexpected_devices": dict(sorted(unexpected_devices.items())),
            "duplicate_job_ids": duplicate_job_ids,
            "duplicate_output_paths": duplicate_output_paths,
            "expected_label_rows_from_manifest": expected_label_rows,
            "geneval2_jobs": by_benchmark.get("geneval2", 0),
            "model_switches": model_switches,
            "model_order": model_order,
            "model_major_expected_max_switches": max(len(expected_models) - 1, 0),
            "seed_policy": seed_policy,
        },
        job_index,
    )


def _audit_label_design(
    labels_path: Path,
    config: dict[str, Any],
    job_index: dict[str, dict[str, Any]],
    *,
    model_scope: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    by_model: Counter[str] = Counter()
    by_benchmark: Counter[str] = Counter()
    by_condition: Counter[str] = Counter()
    by_benchmark_condition: Counter[str] = Counter()
    required_signal_columns: Counter[str] = Counter()
    required_signal_by_condition: Counter[str] = Counter()
    label_uids: set[str] = set()
    duplicate_label_uids = 0
    missing_job_references = 0
    field_mismatches: Counter[str] = Counter()
    label_uid_mismatches = 0
    eval_uid_not_in_manifest_job = 0
    rows = 0
    rq_state: dict[tuple[str, str], dict[str, Any]] = defaultdict(_empty_rq_pair_state)
    pair_models = {
        pair["comparison_id"]: {
            "teacher": pair["teacher"]["model_id"],
            "student": pair["student"]["model_id"],
            "family": pair["family"],
        }
        for pair in selected_pairs(config, model_scope=model_scope)
    }
    broad_axes: set[tuple[str, str]] = set()
    explicit_targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    header_missing: list[str] = []

    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header_missing = [field for field in LABEL_REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
        for row in reader:
            rows += 1
            benchmark = str(row.get("benchmark", ""))
            condition = str(row.get("condition_type", ""))
            model_id = str(row.get("model_id", ""))
            axis_id = str(row.get("axis_id", ""))
            label_uid = str(row.get("label_uid", ""))
            job_id = str(row.get("job_id", ""))
            eval_uid = str(row.get("eval_uid", ""))
            by_model[model_id] += 1
            by_benchmark[benchmark] += 1
            by_condition[condition] += 1
            by_benchmark_condition[f"{benchmark}/{condition}"] += 1
            if label_uid in label_uids:
                duplicate_label_uids += 1
            label_uids.add(label_uid)
            for column in _required_signal_columns_for_design(row):
                required_signal_columns[column] += 1
                required_signal_by_condition[f"{condition}/{column}"] += 1
            manifest_job = job_index.get(job_id)
            if manifest_job is None:
                missing_job_references += 1
            else:
                for label_field, job_field in (
                    ("model_id", "model_id"),
                    ("benchmark", "benchmark"),
                    ("condition_type", "condition_type"),
                    ("image_prompt_uid", "image_prompt_uid"),
                    ("seed", "seed"),
                ):
                    if str(row.get(label_field, "")) != str(manifest_job[job_field]):
                        field_mismatches[label_field] += 1
                if str(row.get("image_path", "")) != str(manifest_job["output_path"]):
                    field_mismatches["image_path"] += 1
                if eval_uid not in manifest_job["eval_uids"]:
                    eval_uid_not_in_manifest_job += 1
            if label_uid != stable_hash(job_id, eval_uid):
                label_uid_mismatches += 1
            if condition in BROAD_DIAGNOSTIC_CONDITIONS:
                broad_axes.add((benchmark, axis_id))
            if condition in EXPLICIT_DIAGNOSTIC_CONDITIONS and str(row.get("target_behavior", "")).strip():
                explicit_targets[(benchmark, axis_id)].add(str(row.get("target_behavior", "")).strip())
            applicable = applicable_rqs(row, config)
            if not applicable:
                continue
            cell = _cell_key(row)
            for comparison_id, models in pair_models.items():
                role = "teacher" if model_id == models["teacher"] else "student" if model_id == models["student"] else ""
                if not role:
                    continue
                for rq in applicable:
                    bucket = rq_state[(rq, comparison_id)]
                    bucket["family"] = models["family"]
                    bucket["planned_cells"][cell].add(role)
                    bucket["planned_by_benchmark"][benchmark][cell].add(role)
                    bucket["row_count"] += 1

    rq_summary, rq_pair_rows = _summarize_rq_design(rq_state, config, model_scope=model_scope)
    rq3_bridge = _summarize_rq3_bridge(broad_axes, explicit_targets)
    return (
        {
            "rows": rows,
            "rows_by_model": dict(sorted(by_model.items())),
            "rows_by_benchmark": dict(sorted(by_benchmark.items())),
            "rows_by_condition": dict(sorted(by_condition.items())),
            "rows_by_benchmark_condition": dict(sorted(by_benchmark_condition.items())),
            "missing_required_header_fields": header_missing,
            "duplicate_label_uids": duplicate_label_uids,
            "missing_job_references": missing_job_references,
            "manifest_field_mismatches": dict(sorted(field_mismatches.items())),
            "label_uid_mismatches": label_uid_mismatches,
            "eval_uid_not_in_manifest_job": eval_uid_not_in_manifest_job,
            "required_signal_cells_by_column": dict(sorted(required_signal_columns.items())),
            "required_signal_cells_by_condition_column": dict(sorted(required_signal_by_condition.items())),
            "geneval2_label_rows": by_benchmark.get("geneval2", 0),
        },
        rq_summary,
        rq_pair_rows,
        rq3_bridge,
    )


def _summarize_rq_design(
    rq_state: dict[tuple[str, str], dict[str, Any]],
    config: dict[str, Any],
    *,
    model_scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    summary: dict[str, Any] = {}
    for rq in RQ_ORDER:
        pair_rows = []
        for pair in selected_pairs(config, model_scope=model_scope):
            pair_id = pair["comparison_id"]
            bucket = rq_state[(rq, pair_id)]
            planned = _count_matched_cells(bucket["planned_cells"])
            row = {
                "rq": rq,
                "description": RQ_DESCRIPTIONS[rq],
                "comparison_id": pair_id,
                "family": bucket.get("family", pair.get("family", "")),
                "row_count": bucket["row_count"],
                "planned_matched_cells": planned,
                "planned_matched_cells_by_benchmark": _matched_by_benchmark(bucket["planned_by_benchmark"]),
                "status": "planned" if planned > 0 else ("not_applicable" if rq == "unlearning_optional" else "missing_design"),
            }
            pair_rows.append(row)
            rows.append(row)
        summary[rq] = {
            "description": RQ_DESCRIPTIONS[rq],
            "planned": all(row["planned_matched_cells"] > 0 for row in pair_rows) if rq != "unlearning_optional" else True,
            "planned_matched_cells": sum(row["planned_matched_cells"] for row in pair_rows),
            "status_counts": dict(Counter(row["status"] for row in pair_rows)),
        }
    return summary, rows


def _summarize_rq3_bridge(
    broad_axes: set[tuple[str, str]],
    explicit_targets: dict[tuple[str, str], set[str]],
) -> dict[str, Any]:
    bridged = {
        key: targets
        for key, targets in explicit_targets.items()
        if key in broad_axes and targets
    }
    by_benchmark: Counter[str] = Counter()
    for benchmark, _axis_id in bridged:
        by_benchmark[benchmark] += len(bridged[(benchmark, _axis_id)])
    return {
        "axis_bridge_count": len(bridged),
        "axis_target_bridge_count": sum(len(targets) for targets in bridged.values()),
        "axis_target_bridge_count_by_benchmark": dict(sorted(by_benchmark.items())),
        "example_bridges": [
            {"benchmark": benchmark, "axis_id": axis_id, "targets": sorted(targets)[:5]}
            for (benchmark, axis_id), targets in sorted(bridged.items())[:20]
        ],
    }


def _summarize_seed_policy(
    seed_sets_by_model_prompt: dict[tuple[str, str, str, str], set[str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_benchmark_condition: dict[str, dict[str, int]] = defaultdict(lambda: {"groups": 0, "compliant_groups": 0, "noncompliant_groups": 0})
    examples = []
    unknown_policy_groups = 0
    for (model_id, benchmark, condition, image_uid), observed in seed_sets_by_model_prompt.items():
        key = f"{benchmark}/{condition}"
        by_benchmark_condition[key]["groups"] += 1
        try:
            expected = {str(seed) for seed in seeds_for_condition(config, benchmark, condition)}
        except KeyError:
            expected = set()
            unknown_policy_groups += 1
        if observed == expected:
            by_benchmark_condition[key]["compliant_groups"] += 1
        else:
            by_benchmark_condition[key]["noncompliant_groups"] += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "model_id": model_id,
                        "benchmark": benchmark,
                        "condition_type": condition,
                        "image_prompt_uid": image_uid,
                        "observed_seeds": sorted(observed, key=_seed_sort_key),
                        "expected_seeds": sorted(expected, key=_seed_sort_key),
                    }
                )
    return {
        "groups": len(seed_sets_by_model_prompt),
        "unknown_policy_groups": unknown_policy_groups,
        "noncompliant_groups": sum(item["noncompliant_groups"] for item in by_benchmark_condition.values()),
        "by_benchmark_condition": {key: value for key, value in sorted(by_benchmark_condition.items())},
        "examples": examples,
    }


def _audit_shard_index(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": str(path) if path else "", "exists": False}
    index = read_json(path)
    mixed = []
    total_jobs = 0
    for shard in index.get("shards", []):
        total_jobs += int(shard.get("jobs", 0))
        models = shard.get("jobs_by_model", {})
        if len(models) > 1:
            mixed.append({"shard_id": shard.get("shard_id", ""), "jobs_by_model": models})
    return {
        "path": str(path),
        "exists": True,
        "respect_model_boundaries": bool(index.get("respect_model_boundaries")),
        "shard_count": int(index.get("shard_count", 0)),
        "total_jobs": total_jobs,
        "index_total_jobs": int(index.get("total_jobs", 0)),
        "mixed_model_shards": len(mixed),
        "mixed_model_examples": mixed[:20],
    }


def _design_gates(
    *,
    config: dict[str, Any],
    expected_benchmarks: list[str],
    expected_models: list[str],
    prompt_summary: dict[str, Any],
    manifest_summary: dict[str, Any],
    label_summary: dict[str, Any],
    rq_summary: dict[str, Any],
    rq3_bridge: dict[str, Any],
    shard_summary: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    expected_benchmark_set = set(expected_benchmarks)
    prompt_benchmarks = set(prompt_summary["records_by_benchmark"])
    manifest_benchmarks = set(manifest_summary["jobs_by_benchmark"])
    label_benchmarks = set(label_summary["rows_by_benchmark"])
    manifest_models = set(manifest_summary["jobs_by_model"])
    label_models = set(label_summary["rows_by_model"])
    expected_model_set = set(expected_models)
    model_job_counts = [manifest_summary["jobs_by_model"].get(model, 0) for model in expected_models]
    model_label_counts = [label_summary["rows_by_model"].get(model, 0) for model in expected_models]
    safety_conditions = set(label_summary["rows_by_condition"]) & REQUIRED_SAFETY_CONDITIONS
    memorization_conditions = set(label_summary["rows_by_condition"]) & REQUIRED_MEMORIZATION_CONDITIONS
    expected_label_rows = manifest_summary["expected_label_rows_from_manifest"]
    required_primary_rqs = _required_primary_rqs(config)
    rq3_bridge_required = _require_rq3_dissociation_bridge(config)
    shard_pass = True
    shard_detail = "No shard index supplied."
    if shard_summary is not None:
        shard_pass = (
            shard_summary.get("exists")
            and shard_summary.get("respect_model_boundaries")
            and shard_summary.get("mixed_model_shards") == 0
            and shard_summary.get("total_jobs") == manifest_summary["jobs"]
        )
        shard_detail = f"mixed_model_shards={shard_summary.get('mixed_model_shards')}, total_jobs={shard_summary.get('total_jobs')}"
    return {
        "benchmark_sources_available": {
            "pass": all(prompt_summary["configured_source_path_status"].values()),
            "detail": "All configured benchmark source paths for enabled primary benchmarks exist under ./benchmarks.",
        },
        "geneval2_excluded": {
            "pass": prompt_summary["geneval2_records"] == 0 and manifest_summary["geneval2_jobs"] == 0 and label_summary["geneval2_label_rows"] == 0,
            "detail": "GenEval2 is excluded from prompt bank, manifest, and label template for the current primary study.",
        },
        "required_benchmarks_present": {
            "pass": expected_benchmark_set <= prompt_benchmarks and expected_benchmark_set <= manifest_benchmarks and expected_benchmark_set <= label_benchmarks,
            "detail": f"expected={sorted(expected_benchmark_set)}, prompt_missing={sorted(expected_benchmark_set - prompt_benchmarks)}, manifest_missing={sorted(expected_benchmark_set - manifest_benchmarks)}, label_missing={sorted(expected_benchmark_set - label_benchmarks)}",
        },
        "no_unexpected_benchmarks": {
            "pass": (prompt_benchmarks | manifest_benchmarks | label_benchmarks) <= expected_benchmark_set,
            "detail": f"unexpected={sorted((prompt_benchmarks | manifest_benchmarks | label_benchmarks) - expected_benchmark_set)}",
        },
        "prompt_sources_independent": {
            "pass": prompt_summary["old_diversity_prompt_records"] == 0,
            "detail": "No prompt-bank source/source_path records match old H1/H2 diversity prompt markers.",
        },
        "primary_models_present": {
            "pass": manifest_models == expected_model_set and label_models == expected_model_set,
            "detail": f"expected={sorted(expected_model_set)}, manifest={sorted(manifest_models)}, labels={sorted(label_models)}",
        },
        "primary_model_job_balance": {
            "pass": len(set(model_job_counts)) == 1 and all(count > 0 for count in model_job_counts),
            "detail": f"jobs_by_expected_model={dict(zip(expected_models, model_job_counts, strict=True))}",
        },
        "primary_label_row_balance": {
            "pass": len(set(model_label_counts)) == 1 and all(count > 0 for count in model_label_counts),
            "detail": f"label_rows_by_expected_model={dict(zip(expected_models, model_label_counts, strict=True))}",
        },
        "manifest_device": {
            "pass": manifest_summary["unexpected_devices"] == {} and set(manifest_summary["jobs_by_device"]) == {str(config.get("default_device", ""))},
            "detail": f"jobs_by_device={manifest_summary['jobs_by_device']}",
        },
        "model_major_manifest": {
            "pass": manifest_summary["model_switches"] <= manifest_summary["model_major_expected_max_switches"],
            "detail": f"model_switches={manifest_summary['model_switches']}, expected_max={manifest_summary['model_major_expected_max_switches']}",
        },
        "seed_policy": {
            "pass": manifest_summary["seed_policy"]["noncompliant_groups"] == 0 and manifest_summary["seed_policy"]["unknown_policy_groups"] == 0,
            "detail": f"noncompliant_groups={manifest_summary['seed_policy']['noncompliant_groups']}, unknown_policy_groups={manifest_summary['seed_policy']['unknown_policy_groups']}",
        },
        "manifest_label_join": {
            "pass": label_summary["rows"] == expected_label_rows and label_summary["missing_job_references"] == 0 and label_summary["manifest_field_mismatches"] == {} and label_summary["eval_uid_not_in_manifest_job"] == 0,
            "detail": f"label_rows={label_summary['rows']}, expected_from_manifest={expected_label_rows}, missing_job_refs={label_summary['missing_job_references']}, field_mismatches={label_summary['manifest_field_mismatches']}",
        },
        "label_uid_integrity": {
            "pass": label_summary["duplicate_label_uids"] == 0 and label_summary["label_uid_mismatches"] == 0,
            "detail": f"duplicates={label_summary['duplicate_label_uids']}, mismatches={label_summary['label_uid_mismatches']}",
        },
        "label_schema": {
            "pass": label_summary["missing_required_header_fields"] == [],
            "detail": f"missing_header_fields={label_summary['missing_required_header_fields']}",
        },
        "rq_pair_design": {
            "pass": all(rq_summary[rq]["planned"] for rq in required_primary_rqs),
            "detail": ", ".join(f"{rq}:{rq_summary[rq]['planned_matched_cells']}" for rq in required_primary_rqs),
        },
        "rq3_dissociation_bridge": {
            "pass": (not rq3_bridge_required) or rq3_bridge["axis_target_bridge_count"] > 0,
            "detail": f"required={rq3_bridge_required}, axis_bridges={rq3_bridge['axis_bridge_count']}, axis_target_bridges={rq3_bridge['axis_target_bridge_count']}, by_benchmark={rq3_bridge['axis_target_bridge_count_by_benchmark']}",
        },
        "safety_memorization_coverage": {
            "pass": REQUIRED_SAFETY_CONDITIONS <= safety_conditions and REQUIRED_MEMORIZATION_CONDITIONS <= memorization_conditions,
            "detail": f"safety_conditions={sorted(safety_conditions)}, memorization_conditions={sorted(memorization_conditions)}",
        },
        "single_model_shards": {
            "pass": shard_pass,
            "detail": shard_detail,
        },
    }


def _required_primary_rqs(config: dict[str, Any]) -> list[str]:
    configured = config.get("analysis", {}).get("required_primary_rqs")
    if configured is None:
        return list(REQUIRED_PRIMARY_RQS)
    return [str(rq) for rq in configured if str(rq) in RQ_DESCRIPTIONS]


def _require_rq3_dissociation_bridge(config: dict[str, Any]) -> bool:
    return bool(config.get("analysis", {}).get("require_rq3_dissociation_bridge", True))


def _configured_source_paths(config: dict[str, Any], expected_benchmarks: list[str]) -> list[str]:
    out: set[str] = set()
    for benchmark in expected_benchmarks:
        for source_path in config.get("benchmark_plan", {}).get(benchmark, {}).get("source_paths", []):
            out.add(str(source_path))
    return sorted(out)


def _required_signal_columns_for_design(row: dict[str, Any]) -> list[str]:
    # Imported lazily through execution to keep design audit in sync with label readiness rules.
    from .execution import required_signal_columns

    return required_signal_columns(row)


def _empty_rq_pair_state() -> dict[str, Any]:
    return {
        "family": "",
        "row_count": 0,
        "planned_cells": defaultdict(set),
        "planned_by_benchmark": defaultdict(lambda: defaultdict(set)),
    }


def _count_matched_cells(cells: dict[tuple[str, ...], set[str]]) -> int:
    return sum(roles == {"teacher", "student"} for roles in cells.values())


def _matched_by_benchmark(grouped: dict[str, dict[tuple[str, ...], set[str]]]) -> dict[str, int]:
    return {benchmark: _count_matched_cells(cells) for benchmark, cells in sorted(grouped.items())}


def _cell_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("benchmark", "")),
        str(row.get("axis_id", "")),
        str(row.get("condition_type", "")),
        str(row.get("image_prompt_uid", "")),
        str(row.get("seed", "")),
        str(row.get("eval_uid", "")),
    )


def _seed_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)
