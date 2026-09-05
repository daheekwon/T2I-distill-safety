from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .analysis import behavior_label
from .config import selected_pairs, unique_models
from .io import iter_jsonl, read_json, stable_hash, write_csv, write_json, write_jsonl

MANIFEST_REQUIRED_FIELDS = [
    "job_id",
    "model_id",
    "runner",
    "hf_model",
    "device",
    "benchmark",
    "condition_type",
    "image_prompt_uid",
    "seed",
    "prompt_text",
    "output_path",
]

LABEL_REQUIRED_FIELDS = [
    "label_uid",
    "job_id",
    "eval_uid",
    "model_id",
    "benchmark",
    "axis_id",
    "axis_name",
    "condition_type",
    "complexity_level",
    "image_prompt_uid",
    "seed",
    "prompt_text",
    "image_path",
    "semantic_label",
    "pass_score",
    "quality_score",
    "refusal",
    "unsafe",
    "memorization_score",
]

DISTRIBUTION_CONDITIONS = {"broad", "social_bias_default", "benign_overrefusal"}
PASS_FAIL_CONDITIONS = {
    "explicit",
    "compositional_native",
    "implicit_knowledge",
    "knowledge_native",
    "knowledge_injected",
    "unlearning_explicit",
    "unlearning_implicit",
    "unlearning_retain",
    "unlearning_robustness",
    "unlearning_target",
}
SAFETY_CONDITIONS = {"risky_safety", "safety_native", "unsafe_safety", "safety_counterfactual_benign"}
MEMORIZATION_CONDITIONS = {"memorization_trigger", "memorization_control"}


def split_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    jobs_per_shard: int = 10000,
    index_path: Path | None = None,
    respect_model_boundaries: bool = False,
) -> dict[str, Any]:
    if jobs_per_shard <= 0:
        raise ValueError("jobs_per_shard must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_path or output_dir / "index.json"

    shards: list[dict[str, Any]] = []
    buffer: list[dict[str, Any]] = []
    shard_no = 0
    totals = _empty_job_counters()
    previous_model_id: str | None = None

    for job in iter_jsonl(manifest_path):
        model_id = str(job.get("model_id", ""))
        if respect_model_boundaries and buffer and previous_model_id is not None and model_id != previous_model_id:
            shards.append(_flush_shard(buffer, output_dir, shard_no))
            buffer = []
            shard_no += 1
        buffer.append(job)
        previous_model_id = model_id
        _update_job_counters(totals, job)
        if len(buffer) >= jobs_per_shard:
            shards.append(_flush_shard(buffer, output_dir, shard_no))
            buffer = []
            shard_no += 1
    if buffer:
        shards.append(_flush_shard(buffer, output_dir, shard_no))

    index = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "jobs_per_shard": jobs_per_shard,
        "respect_model_boundaries": respect_model_boundaries,
        "total_jobs": totals["jobs"],
        "jobs_by_model": dict(sorted(totals["by_model"].items())),
        "jobs_by_benchmark": dict(sorted(totals["by_benchmark"].items())),
        "jobs_by_device": dict(sorted(totals["by_device"].items())),
        "shard_count": len(shards),
        "shards": shards,
    }
    write_json(index_path, index)
    return index


def audit_experiment_state(
    *,
    config: dict[str, Any],
    prompt_bank_path: Path,
    manifest_path: Path,
    labels_path: Path | None = None,
    check_images: bool = False,
    image_verify: bool = False,
    image_limit: int | None = None,
) -> dict[str, Any]:
    prompt_summary = audit_prompt_bank(prompt_bank_path)
    manifest_summary = audit_manifest(
        manifest_path,
        expected_device=config.get("default_device"),
        check_images=check_images,
        image_verify=image_verify,
        image_limit=image_limit,
    )
    label_summary = None
    if labels_path and labels_path.exists():
        label_summary = audit_labels(labels_path, config=config)
    gates = _experiment_gates(config, prompt_summary, manifest_summary, label_summary)
    return {
        "prompt_bank": prompt_summary,
        "manifest": manifest_summary,
        "labels": label_summary,
        "gates": gates,
    }


def audit_prompt_bank(path: Path) -> dict[str, Any]:
    counts = _empty_prompt_counters()
    eval_uids: set[str] = set()
    image_uids: set[str] = set()
    duplicate_eval_uids = 0
    geneval2_records = 0
    for row in iter_jsonl(path):
        counts["records"] += 1
        benchmark = str(row.get("benchmark", ""))
        condition = str(row.get("condition_type", ""))
        axis_id = str(row.get("axis_id", ""))
        image_uid = str(row.get("image_prompt_uid", ""))
        eval_uid = str(row.get("eval_uid", ""))
        counts["by_benchmark"][benchmark] += 1
        counts["by_condition"][condition] += 1
        counts["axis_ids"].add(axis_id)
        image_uids.add(image_uid)
        if eval_uid in eval_uids:
            duplicate_eval_uids += 1
        eval_uids.add(eval_uid)
        if benchmark == "geneval2":
            geneval2_records += 1
        if row.get("requires_valid_lineage"):
            counts["requires_valid_lineage_records"] += 1
    return {
        "records": counts["records"],
        "records_by_benchmark": dict(sorted(counts["by_benchmark"].items())),
        "records_by_condition": dict(sorted(counts["by_condition"].items())),
        "axis_count": len(counts["axis_ids"]),
        "unique_image_prompts": len(image_uids),
        "duplicate_eval_uids": duplicate_eval_uids,
        "geneval2_records": geneval2_records,
        "requires_valid_lineage_records": counts["requires_valid_lineage_records"],
    }


def audit_manifest(
    path: Path,
    *,
    expected_device: str | None = None,
    check_images: bool = False,
    image_verify: bool = False,
    image_limit: int | None = None,
) -> dict[str, Any]:
    counters = _empty_job_counters()
    job_ids: set[str] = set()
    output_paths: set[str] = set()
    duplicate_job_ids = 0
    duplicate_output_paths = 0
    missing_fields: Counter[str] = Counter()
    unexpected_devices: Counter[str] = Counter()
    image_status = _empty_image_status()

    for row in iter_jsonl(path):
        _update_job_counters(counters, row)
        for field in MANIFEST_REQUIRED_FIELDS:
            if row.get(field) in (None, ""):
                missing_fields[field] += 1
        job_id = str(row.get("job_id", ""))
        output_path = str(row.get("output_path", ""))
        if job_id in job_ids:
            duplicate_job_ids += 1
        job_ids.add(job_id)
        if output_path in output_paths:
            duplicate_output_paths += 1
        output_paths.add(output_path)
        device = str(row.get("device", ""))
        if expected_device and device != expected_device:
            unexpected_devices[device] += 1
        if check_images:
            _audit_image_path(Path(output_path), image_status, verify=image_verify, limit=image_limit)

    return {
        "jobs": counters["jobs"],
        "jobs_by_model": dict(sorted(counters["by_model"].items())),
        "jobs_by_benchmark": dict(sorted(counters["by_benchmark"].items())),
        "jobs_by_device": dict(sorted(counters["by_device"].items())),
        "duplicate_job_ids": duplicate_job_ids,
        "duplicate_output_paths": duplicate_output_paths,
        "missing_required_fields": dict(sorted(missing_fields.items())),
        "unexpected_devices": dict(sorted(unexpected_devices.items())),
        "image_status": dict(image_status) if check_images else None,
    }


def audit_labels(path: Path, *, config: dict[str, Any], model_scope: str = "primary") -> dict[str, Any]:
    required_header_missing: list[str] = []
    counters = _empty_label_counters()
    label_uids: set[str] = set()
    matched_cells: dict[tuple[str, str, str, str, str, str], set[str]] = defaultdict(set)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required_header_missing = [field for field in LABEL_REQUIRED_FIELDS if field not in fieldnames]
        for row in reader:
            counters["rows"] += 1
            benchmark = str(row.get("benchmark", ""))
            model_id = str(row.get("model_id", ""))
            condition = str(row.get("condition_type", ""))
            counters["by_benchmark"][benchmark] += 1
            counters["by_model"][model_id] += 1
            counters["by_condition"][condition] += 1
            label_uid = str(row.get("label_uid", ""))
            if label_uid in label_uids:
                counters["duplicate_label_uids"] += 1
            label_uids.add(label_uid)
            missing_signals = required_signal_columns(row)
            for column in missing_signals:
                if str(row.get(column, "")).strip() == "":
                    counters["missing_required_signal_by_column"][column] += 1
                    counters["missing_required_signal_by_condition"][condition] += 1
            label = behavior_label(row)
            if label:
                counters["filled_rows"] += 1
                counters["filled_by_benchmark"][benchmark] += 1
                counters["filled_by_model"][model_id] += 1
                key = (
                    benchmark,
                    str(row.get("axis_id", "")),
                    condition,
                    str(row.get("image_prompt_uid", "")),
                    str(row.get("seed", "")),
                    str(row.get("eval_uid", "")),
                )
                matched_cells[key].add(model_id)

    pair_summary = _label_pair_coverage(matched_cells, config, model_scope)
    return {
        "rows": counters["rows"],
        "filled_rows": counters["filled_rows"],
        "filled_fraction": counters["filled_rows"] / counters["rows"] if counters["rows"] else 0.0,
        "rows_by_benchmark": dict(sorted(counters["by_benchmark"].items())),
        "rows_by_model": dict(sorted(counters["by_model"].items())),
        "rows_by_condition": dict(sorted(counters["by_condition"].items())),
        "filled_by_benchmark": dict(sorted(counters["filled_by_benchmark"].items())),
        "filled_by_model": dict(sorted(counters["filled_by_model"].items())),
        "duplicate_label_uids": counters["duplicate_label_uids"],
        "missing_required_header_fields": required_header_missing,
        "missing_required_signal_by_column": dict(sorted(counters["missing_required_signal_by_column"].items())),
        "missing_required_signal_by_condition": dict(sorted(counters["missing_required_signal_by_condition"].items())),
        "pair_coverage": pair_summary,
    }


def write_audit_reports(audit: dict[str, Any], output_json: Path, output_markdown: Path | None = None) -> None:
    write_json(output_json, audit)
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_audit_markdown(audit), encoding="utf-8")


def render_audit_markdown(audit: dict[str, Any]) -> str:
    lines = ["# Experiment State Audit", ""]
    gates = audit.get("gates", {})
    lines.append("## Gates")
    for name, gate in sorted(gates.items()):
        status = "PASS" if gate.get("pass") else "WARN"
        lines.append(f"- {status} `{name}`: {gate.get('detail', '')}")
    lines.append("")
    prompt_bank = audit.get("prompt_bank", {})
    lines.append("## Prompt Bank")
    lines.append(f"- records: {prompt_bank.get('records', 0)}")
    lines.append(f"- unique image prompts: {prompt_bank.get('unique_image_prompts', 0)}")
    lines.append(f"- axes: {prompt_bank.get('axis_count', 0)}")
    lines.append(f"- GenEval2 records: {prompt_bank.get('geneval2_records', 0)}")
    lines.append("")
    manifest = audit.get("manifest", {})
    lines.append("## Manifest")
    lines.append(f"- jobs: {manifest.get('jobs', 0)}")
    lines.append(f"- devices: {manifest.get('jobs_by_device', {})}")
    lines.append(f"- duplicate job ids: {manifest.get('duplicate_job_ids', 0)}")
    lines.append(f"- duplicate output paths: {manifest.get('duplicate_output_paths', 0)}")
    labels = audit.get("labels")
    if labels is not None:
        lines.append("")
        lines.append("## Labels")
        lines.append(f"- rows: {labels.get('rows', 0)}")
        lines.append(f"- filled rows: {labels.get('filled_rows', 0)} ({labels.get('filled_fraction', 0.0):.4f})")
        lines.append(f"- duplicate label ids: {labels.get('duplicate_label_uids', 0)}")
    lines.append("")
    return "\n".join(lines)


def required_signal_columns(row: dict[str, Any]) -> list[str]:
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition_type", ""))
    if condition in SAFETY_CONDITIONS:
        return ["unsafe", "refusal", "quality_score"]
    if benchmark == "overt" or condition == "benign_overrefusal":
        return ["refusal", "unsafe", "quality_score"]
    if condition in MEMORIZATION_CONDITIONS:
        return ["memorization_score", "quality_score"]
    if condition.startswith("unlearning_"):
        return ["pass_score", "semantic_label", "quality_score"]
    if condition in PASS_FAIL_CONDITIONS:
        return ["pass_score", "quality_score"]
    if condition in DISTRIBUTION_CONDITIONS:
        return ["semantic_label", "quality_score"]
    return ["quality_score"]


def _label_pair_coverage(
    matched_cells: dict[tuple[str, str, str, str, str, str], set[str]],
    config: dict[str, Any],
    model_scope: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in selected_pairs(config, model_scope=model_scope):
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        both = teacher_only = student_only = neither = 0
        by_benchmark: Counter[str] = Counter()
        for key, models in matched_cells.items():
            has_teacher = teacher in models
            has_student = student in models
            if has_teacher and has_student:
                both += 1
                by_benchmark[key[0]] += 1
            elif has_teacher:
                teacher_only += 1
            elif has_student:
                student_only += 1
            else:
                neither += 1
        out[pair["comparison_id"]] = {
            "matched_labeled_cells": both,
            "teacher_only_labeled_cells": teacher_only,
            "student_only_labeled_cells": student_only,
            "neither_labeled_cells": neither,
            "matched_labeled_cells_by_benchmark": dict(sorted(by_benchmark.items())),
        }
    return out


def _experiment_gates(
    config: dict[str, Any],
    prompt_summary: dict[str, Any],
    manifest_summary: dict[str, Any],
    label_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_device = config.get("default_device")
    primary_models = [model["model_id"] for model in unique_models(config, "primary")]
    jobs_by_model = manifest_summary.get("jobs_by_model", {})
    per_model_counts = [jobs_by_model.get(model_id, 0) for model_id in primary_models]
    gates = {
        "geneval2_excluded": {
            "pass": prompt_summary.get("geneval2_records", 0) == 0 and "geneval2" not in manifest_summary.get("jobs_by_benchmark", {}),
            "detail": "GenEval2 must remain absent from prompt bank and manifest for the current primary plan.",
        },
        "manifest_device": {
            "pass": manifest_summary.get("unexpected_devices", {}) == {} and list(manifest_summary.get("jobs_by_device", {}).keys()) == [expected_device],
            "detail": f"Expected every generation job on {expected_device}.",
        },
        "manifest_unique_ids": {
            "pass": manifest_summary.get("duplicate_job_ids", 0) == 0 and manifest_summary.get("duplicate_output_paths", 0) == 0,
            "detail": "job_id and output_path must be unique.",
        },
        "primary_model_balance": {
            "pass": len(set(per_model_counts)) == 1 and all(count > 0 for count in per_model_counts),
            "detail": f"Primary model job counts: {dict(zip(primary_models, per_model_counts, strict=True))}",
        },
    }
    if label_summary is not None:
        gates["label_schema"] = {
            "pass": label_summary.get("missing_required_header_fields", []) == [],
            "detail": "Label CSV must expose all analysis columns.",
        }
        gates["label_ready_for_analysis"] = {
            "pass": label_summary.get("filled_rows", 0) > 0 and label_summary.get("missing_required_signal_by_column", {}) == {},
            "detail": "All benchmark-specific required signal columns should be filled before final analysis.",
        }
    return gates


def _flush_shard(rows: list[dict[str, Any]], output_dir: Path, shard_no: int) -> dict[str, Any]:
    path = output_dir / f"shard_{shard_no:05d}.jsonl"
    write_jsonl(path, rows)
    counters = _empty_job_counters()
    for row in rows:
        _update_job_counters(counters, row)
    return {
        "shard_id": f"shard_{shard_no:05d}",
        "path": str(path),
        "jobs": counters["jobs"],
        "first_job_id": rows[0].get("job_id", "") if rows else "",
        "last_job_id": rows[-1].get("job_id", "") if rows else "",
        "jobs_by_model": dict(sorted(counters["by_model"].items())),
        "jobs_by_benchmark": dict(sorted(counters["by_benchmark"].items())),
        "jobs_by_device": dict(sorted(counters["by_device"].items())),
    }


def _audit_image_path(path: Path, status: Counter[str], *, verify: bool, limit: int | None) -> None:
    if limit is not None and status["checked"] >= limit:
        status["skipped_by_limit"] += 1
        return
    status["checked"] += 1
    if not path.exists():
        status["missing"] += 1
        return
    if path.stat().st_size <= 0:
        status["empty"] += 1
        return
    if not verify:
        status["present_nonempty"] += 1
        return
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        status["verified"] += 1
    except Exception:
        status["unreadable"] += 1


def _empty_job_counters() -> dict[str, Any]:
    return {
        "jobs": 0,
        "by_model": Counter(),
        "by_benchmark": Counter(),
        "by_device": Counter(),
    }


def _update_job_counters(counters: dict[str, Any], row: dict[str, Any]) -> None:
    counters["jobs"] += 1
    counters["by_model"][str(row.get("model_id", ""))] += 1
    counters["by_benchmark"][str(row.get("benchmark", ""))] += 1
    counters["by_device"][str(row.get("device", ""))] += 1


def _empty_prompt_counters() -> dict[str, Any]:
    return {
        "records": 0,
        "by_benchmark": Counter(),
        "by_condition": Counter(),
        "axis_ids": set(),
        "requires_valid_lineage_records": 0,
    }


def _empty_label_counters() -> dict[str, Any]:
    return {
        "rows": 0,
        "filled_rows": 0,
        "duplicate_label_uids": 0,
        "by_benchmark": Counter(),
        "by_model": Counter(),
        "by_condition": Counter(),
        "filled_by_benchmark": Counter(),
        "filled_by_model": Counter(),
        "missing_required_signal_by_column": Counter(),
        "missing_required_signal_by_condition": Counter(),
    }


def _empty_image_status() -> Counter[str]:
    return Counter({"checked": 0, "missing": 0, "empty": 0, "present_nonempty": 0, "verified": 0, "unreadable": 0, "skipped_by_limit": 0})
