from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .evaluation import build_evaluator_plan, export_evaluator_batch, plan_evaluator_batches
from .io import iter_jsonl, write_json, write_jsonl
from .manifest import build_label_template
from .execution import split_manifest


def build_preflight_package(
    prompt_bank_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    prompts_per_condition: int = 1,
    seeds: set[int] | None = None,
    jobs_per_shard: int = 200,
) -> dict[str, Any]:
    if prompts_per_condition <= 0:
        raise ValueError("prompts_per_condition must be positive")
    seeds = seeds if seeds is not None else {0}
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_image_prompts_by_condition(prompt_bank_path, prompts_per_condition=prompts_per_condition)
    selected_uids = {uid for uids in selected.values() for uid in uids}
    manifest_out = output_dir / "generation_manifest.jsonl"
    job_summary = _write_selected_manifest(manifest_path, manifest_out, selected_uids=selected_uids, seeds=seeds)

    labels_out = output_dir / "label_template.csv"
    label_summary = build_label_template(prompt_bank_path, manifest_out, labels_out)
    evaluator_summary = build_evaluator_plan(
        labels_out,
        output_dir / "evaluator_task_index.csv",
        output_dir / "evaluator_schema.json",
        output_dir / "evaluator_summary.csv",
    )
    evaluator_batches_dir = output_dir / "evaluator_batches"
    _clear_generated_jsonl(evaluator_batches_dir)
    evaluator_batch_summary = plan_evaluator_batches(
        labels_out,
        evaluator_batches_dir,
        output_dir / "evaluator_batch_index.csv",
        batch_size=jobs_per_shard,
        split_fields=("evaluator_kind", "benchmark", "condition_type"),
        dry_run=False,
    )
    evaluator_preview_summary = export_evaluator_batch(labels_out, output_dir / "evaluator_batch_preview.jsonl", limit=min(20, label_summary["label_rows"]))
    shard_index = split_manifest(manifest_out, output_dir / "shards", jobs_per_shard=jobs_per_shard)
    summary = {
        "output_dir": str(output_dir),
        "prompts_per_condition": prompts_per_condition,
        "seeds": sorted(seeds),
        "selected_condition_count": len(selected),
        "selected_unique_image_prompts": len(selected_uids),
        "selected_by_condition": {f"{bench}/{condition}": len(uids) for (bench, condition), uids in sorted(selected.items())},
        "generation_manifest": job_summary | {"path": str(manifest_out)},
        "label_template": label_summary,
        "evaluator_plan": evaluator_summary,
        "evaluator_batches": evaluator_batch_summary,
        "evaluator_preview": evaluator_preview_summary,
        "shards": {"path": str(output_dir / "shards" / "index.json"), "shard_count": shard_index["shard_count"], "jobs_per_shard": jobs_per_shard},
    }
    write_json(output_dir / "preflight_summary.json", summary)
    return summary


def _clear_generated_jsonl(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.glob("*.jsonl"):
        item.unlink()


def _select_image_prompts_by_condition(prompt_bank_path: Path, *, prompts_per_condition: int) -> dict[tuple[str, str], list[str]]:
    selected: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in iter_jsonl(prompt_bank_path):
        if row.get("requires_valid_lineage"):
            continue
        key = (str(row["benchmark"]), str(row["condition_type"]))
        uid = str(row["image_prompt_uid"])
        if uid in seen[key]:
            continue
        if len(selected[key]) >= prompts_per_condition:
            continue
        selected[key].append(uid)
        seen[key].add(uid)
    return dict(selected)


def _write_selected_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    selected_uids: set[str],
    seeds: set[int],
) -> dict[str, Any]:
    jobs = []
    by_benchmark: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    by_condition: Counter[str] = Counter()
    by_device: Counter[str] = Counter()
    for job in iter_jsonl(manifest_path):
        if str(job.get("image_prompt_uid", "")) not in selected_uids:
            continue
        if int(job.get("seed", -1)) not in seeds:
            continue
        jobs.append(job)
        by_benchmark[str(job.get("benchmark", ""))] += 1
        by_model[str(job.get("model_id", ""))] += 1
        by_condition[str(job.get("condition_type", ""))] += 1
        by_device[str(job.get("device", ""))] += 1
    count = write_jsonl(output_path, jobs)
    return {
        "jobs": count,
        "jobs_by_benchmark": dict(sorted(by_benchmark.items())),
        "jobs_by_model": dict(sorted(by_model.items())),
        "jobs_by_condition": dict(sorted(by_condition.items())),
        "jobs_by_device": dict(sorted(by_device.items())),
    }
