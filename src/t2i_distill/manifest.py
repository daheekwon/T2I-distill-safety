from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import seeds_for_condition, unique_models
from .io import iter_jsonl, slugify, stable_hash, write_csv, write_jsonl

LABEL_TEMPLATE_FIELDS = [
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
    "target_behavior",
    "candidate_behaviors",
    "semantic_label",
    "semantic_label_confidence",
    "pass_score",
    "quality_score",
    "refusal",
    "unsafe",
    "memorization_score",
    "evaluator",
    "notes",
]


def build_generation_manifest(
    config: dict[str, Any],
    prompt_bank_path: Path,
    output_path: Path,
    *,
    model_scope: str = "primary",
    include_gated: bool = False,
    benchmark_filter: set[str] | None = None,
    limit_prompts_per_benchmark: int | None = None,
    job_order: str = "model_major",
) -> dict[str, Any]:
    prompt_records = list(iter_jsonl(prompt_bank_path))
    generation_prompts = dedupe_generation_prompts(
        prompt_records,
        include_gated=include_gated,
        benchmark_filter=benchmark_filter,
        limit_prompts_per_benchmark=limit_prompts_per_benchmark,
    )
    jobs = list(_iter_generation_jobs(config, generation_prompts, model_scope=model_scope, job_order=job_order))
    count = write_jsonl(output_path, jobs)
    return summarize_jobs(jobs) | {"output_path": str(output_path), "jobs": count}


def build_label_template(
    prompt_bank_path: Path,
    generation_manifest_path: Path,
    output_path: Path,
    *,
    include_gated: bool = False,
) -> dict[str, Any]:
    records_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for record in iter_jsonl(prompt_bank_path):
        if record.get("requires_valid_lineage") and not include_gated:
            continue
        records_by_prompt.setdefault(record["image_prompt_uid"], []).append(record)

    rows = []
    for job in iter_jsonl(generation_manifest_path):
        for record in records_by_prompt.get(job["image_prompt_uid"], []):
            rows.append(
                {
                    "label_uid": stable_hash(job["job_id"], record["eval_uid"]),
                    "job_id": job["job_id"],
                    "eval_uid": record["eval_uid"],
                    "model_id": job["model_id"],
                    "benchmark": record["benchmark"],
                    "axis_id": record["axis_id"],
                    "axis_name": record["axis_name"],
                    "condition_type": record["condition_type"],
                    "complexity_level": record["complexity_level"],
                    "image_prompt_uid": record["image_prompt_uid"],
                    "seed": job["seed"],
                    "prompt_text": record["prompt_text"],
                    "image_path": job["output_path"],
                    "target_behavior": record.get("target_behavior", ""),
                    "candidate_behaviors": record.get("candidate_behaviors", []),
                    "semantic_label": "",
                    "semantic_label_confidence": "",
                    "pass_score": "",
                    "quality_score": "",
                    "refusal": "",
                    "unsafe": "",
                    "memorization_score": "",
                    "evaluator": record.get("evaluator", ""),
                    "notes": "",
                }
            )
    count = write_csv(output_path, rows, LABEL_TEMPLATE_FIELDS)
    return {"output_path": str(output_path), "label_rows": count}


def dedupe_generation_prompts(
    prompt_records: Iterable[dict[str, Any]],
    *,
    include_gated: bool,
    benchmark_filter: set[str] | None,
    limit_prompts_per_benchmark: int | None,
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    seen_per_benchmark: dict[str, set[str]] = {}
    for record in prompt_records:
        benchmark = record["benchmark"]
        if benchmark_filter and benchmark not in benchmark_filter:
            continue
        if record.get("requires_valid_lineage") and not include_gated:
            continue
        seen = seen_per_benchmark.setdefault(benchmark, set())
        if limit_prompts_per_benchmark is not None and record["image_prompt_uid"] not in seen:
            if len(seen) >= limit_prompts_per_benchmark:
                continue
        seen.add(record["image_prompt_uid"])
        unique.setdefault(
            record["image_prompt_uid"],
            {
                "image_prompt_uid": record["image_prompt_uid"],
                "benchmark": benchmark,
                "condition_type": record["condition_type"],
                "prompt_text": record["prompt_text"],
                "requires_valid_lineage": bool(record.get("requires_valid_lineage")),
                "eval_uids": [],
            },
        )
        unique[record["image_prompt_uid"]]["eval_uids"].append(record["eval_uid"])
    return list(unique.values())


def summarize_jobs(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, int] = {}
    by_benchmark: dict[str, int] = {}
    by_device: dict[str, int] = {}
    for job in jobs:
        by_model[job["model_id"]] = by_model.get(job["model_id"], 0) + 1
        by_benchmark[job["benchmark"]] = by_benchmark.get(job["benchmark"], 0) + 1
        by_device[job["device"]] = by_device.get(job["device"], 0) + 1
    return {
        "jobs_by_model": dict(sorted(by_model.items())),
        "jobs_by_benchmark": dict(sorted(by_benchmark.items())),
        "jobs_by_device": dict(sorted(by_device.items())),
    }


def _iter_generation_jobs(
    config: dict[str, Any],
    prompts: Iterable[dict[str, Any]],
    *,
    model_scope: str,
    job_order: str = "model_major",
) -> Iterable[dict[str, Any]]:
    if job_order not in {"model_major", "prompt_major"}:
        raise ValueError("job_order must be one of: model_major, prompt_major")
    models = unique_models(config, model_scope=model_scope)
    prompt_list = list(prompts)
    if job_order == "prompt_major":
        for prompt in prompt_list:
            for model in models:
                yield from _jobs_for_model_prompt(config, model, prompt)
        return
    for model in models:
        for prompt in prompt_list:
            yield from _jobs_for_model_prompt(config, model, prompt)


def _jobs_for_model_prompt(
    config: dict[str, Any],
    model: dict[str, Any],
    prompt: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    output_root = Path(config["image_output_root"])
    fmt = config.get("image_format", "png")
    benchmark = prompt["benchmark"]
    seeds = seeds_for_condition(config, benchmark, prompt["condition_type"])
    for seed in seeds:
        output_path = output_root / model["model_id"] / benchmark / f"{slugify(prompt['image_prompt_uid'])}__seed{seed}.{fmt}"
        job_id = stable_hash(model["model_id"], prompt["image_prompt_uid"], seed)
        yield {
            "job_id": job_id,
            "model_id": model["model_id"],
            "display_name": model.get("display_name", model["model_id"]),
            "family": model.get("family", ""),
            "runner": model.get("runner", ""),
            "hf_model": model.get("hf_model", ""),
            "base_model": model.get("base_model", ""),
            "checkpoint": model.get("checkpoint", ""),
            "num_inference_steps": model.get("num_inference_steps"),
            "guidance_scale": model.get("guidance_scale"),
            "device": model.get("device") or config["default_device"],
            "benchmark": benchmark,
            "condition_type": prompt["condition_type"],
            "image_prompt_uid": prompt["image_prompt_uid"],
            "eval_uids": prompt["eval_uids"],
            "seed": seed,
            "prompt_text": prompt["prompt_text"],
            "output_path": str(output_path),
            "requires_valid_lineage": prompt.get("requires_valid_lineage", False),
            "generation_kwargs": _generation_kwargs(model),
        }


def _generation_kwargs(model: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "hf_model",
        "base_model",
        "checkpoint",
        "num_inference_steps",
        "guidance_scale",
        "scheduler",
    ]
    return {key: model[key] for key in keys if key in model and model[key] not in ("", None)}


def manifest_preview(path: Path, n: int = 3) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(iter_jsonl(path)):
        if idx >= n:
            break
        trimmed = dict(row)
        prompt = str(trimmed.get("prompt_text", ""))
        trimmed["prompt_text"] = "<redacted>"
        trimmed["prompt_chars"] = len(prompt)
        rows.append(trimmed)
    return rows
