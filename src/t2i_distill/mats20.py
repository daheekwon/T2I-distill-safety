"""Build the safety/memorization-centered MATS20 pilot package.

The package intentionally stays independent from the older diversity H1/H2
prompt sets. It selects a small, stratified subset from the current benchmark
prompt bank, keeps teacher/student pairs intact, and writes execution manifests
that can be run on separate GPUs while sharing the same canonical output paths.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .evaluation import build_evaluator_plan, export_evaluator_batch, plan_evaluator_batches
from .execution import split_manifest
from .io import iter_jsonl, stable_hash, write_csv, write_json, write_jsonl
from .manifest import build_label_template, summarize_jobs


GPU0_MODELS = ("sdxl_base", "sdxl_lightning_4step")
GPU2_MODELS = (
    "sd35_large",
    "sd35_large_turbo",
    "flux2_klein_base_4b",
    "flux2_klein_4b",
)

DEFAULT_SEED_POLICY: dict[tuple[str, str], tuple[int, ...]] = {
    ("membench", "memorization_trigger"): (0, 1),
    ("membench", "memorization_control"): (0, 1),
    ("overt", "benign_overrefusal"): (0,),
    ("overt", "unsafe_safety"): (0,),
    ("overt", "safety_counterfactual_benign"): (0,),
    ("t2isafety", "safety_native"): (0,),
    ("t2isafety", "social_bias_default"): (0,),
    ("t2i_riskyprompt", "risky_safety"): (0,),
    ("grade", "broad"): (0, 1),
    ("dimcim", "broad"): (0, 1),
}


@dataclass(frozen=True)
class SelectionTarget:
    benchmark: str
    condition_type: str
    n_prompts: int
    reason: str


MAIN_TARGETS: tuple[SelectionTarget, ...] = (
    SelectionTarget(
        "membench",
        "memorization_trigger",
        48,
        "memorization trigger inheritance",
    ),
    SelectionTarget(
        "membench",
        "memorization_control",
        24,
        "memorization trigger/control contrast",
    ),
    SelectionTarget(
        "overt",
        "benign_overrefusal",
        36,
        "benign over-refusal inheritance",
    ),
    SelectionTarget(
        "t2isafety",
        "safety_native",
        36,
        "native safety behavior inheritance",
    ),
    SelectionTarget(
        "t2isafety",
        "social_bias_default",
        18,
        "default demographic behavior inheritance",
    ),
    SelectionTarget(
        "t2i_riskyprompt",
        "risky_safety",
        36,
        "risky prompt boundary behavior",
    ),
    SelectionTarget(
        "grade",
        "broad",
        12,
        "small quality sanity check",
    ),
    SelectionTarget(
        "dimcim",
        "broad",
        12,
        "small object/attribute sanity check",
    ),
)

OVERT_PAIR_PROMPTS = 36

OVERT_ALLOWED_CATEGORIES = {
    "violence",
    "privacy_individual",
    "privacy_public",
    "copyright_violations",
    "illegal_activities",
    "unethical_unsafe_action",
    "unethical",
    "discrimination",
}

T2ISAFETY_ALLOWED_CATEGORIES = {
    "fairness_demographic_distribution",
    "public figures",
    "intellectual property violation",
    "personal identification documents",
    "illegal activity",
    "humiliation",
    "disturbing",
    "violence",
}

RISKY_ALLOWED_AXIS_PREFIXES = (
    "t2i_riskyprompt/Copyright_Infringement/",
    "t2i_riskyprompt/Political_Sensitivity/",
    "t2i_riskyprompt/Violence/Weapons_and_Conflicts",
    "t2i_riskyprompt/Illegal_Activities/",
)

SELECTION_FIELDS = [
    "image_prompt_uid",
    "benchmark",
    "condition_type",
    "axis_id",
    "category",
    "reason",
]

PROMPT_BLOCK_TERMS = (
    "preteen",
    "underage",
    "minor",
    "minors",
    "little girl",
    "little boy",
    "child nude",
    "naked child",
    "teenage girl",
    "teenage boy",
    "porn",
    "pornographic",
    "explicit sex",
    "sexualized",
    "rape",
    "blood",
    "bloody",
    "gore",
    "gory",
    "mutilated",
    "dismembered",
    "beheaded",
    "decapitated",
    "corpse",
    "dead body",
)


def build_mats20_package(
    prompt_bank_path: str | Path,
    full_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    jobs_per_shard: int = 200,
    gpu0_models: Iterable[str] = GPU0_MODELS,
    gpu2_models: Iterable[str] = GPU2_MODELS,
    execution_gpu0_device: str = "cuda:0",
    execution_gpu2_device: str = "cuda:2",
    canonical_device: str = "cuda:2",
) -> dict[str, Any]:
    """Create prompt, label, evaluator, and GPU-specific execution files."""

    prompt_bank_path = Path(prompt_bank_path)
    full_manifest_path = Path(full_manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = list(iter_jsonl(prompt_bank_path))
    selected_uids, selection_rows = select_mats20_prompt_uids(records)
    selected_uid_set = set(selected_uids)

    selected_prompt_rows = [
        row for row in records if row.get("image_prompt_uid") in selected_uid_set
    ]
    prompt_out = output_dir / "prompt_bank.jsonl"
    write_jsonl(prompt_out, selected_prompt_rows)
    write_csv(output_dir / "selection_index.csv", selection_rows, SELECTION_FIELDS)

    combined_jobs = _selected_jobs(
        full_manifest_path,
        selected_uid_set,
        canonical_device=canonical_device,
        output_root=Path("outputs/mats20/images"),
    )
    manifest_out = output_dir / "generation_manifest.jsonl"
    write_jsonl(manifest_out, combined_jobs)

    gpu0_set = set(gpu0_models)
    gpu2_set = set(gpu2_models)
    unknown_models = sorted(
        set(job["model_id"] for job in combined_jobs) - gpu0_set - gpu2_set
    )
    if unknown_models:
        raise ValueError(f"No GPU assignment for models: {unknown_models}")

    gpu0_jobs = [
        {**job, "device": execution_gpu0_device}
        for job in combined_jobs
        if job["model_id"] in gpu0_set
    ]
    gpu2_jobs = [
        {**job, "device": execution_gpu2_device}
        for job in combined_jobs
        if job["model_id"] in gpu2_set
    ]
    gpu0_manifest = output_dir / "generation_manifest_gpu0.jsonl"
    gpu2_manifest = output_dir / "generation_manifest_gpu2.jsonl"
    write_jsonl(gpu0_manifest, gpu0_jobs)
    write_jsonl(gpu2_manifest, gpu2_jobs)

    label_template = output_dir / "label_template.csv"
    label_summary = build_label_template(prompt_out, manifest_out, label_template)

    split_summary = split_manifest(
        manifest_out,
        output_dir / "shards",
        jobs_per_shard=jobs_per_shard,
        respect_model_boundaries=True,
    )
    gpu0_split_summary = split_manifest(
        gpu0_manifest,
        output_dir / "shards_gpu0",
        jobs_per_shard=jobs_per_shard,
        respect_model_boundaries=True,
    )
    gpu2_split_summary = split_manifest(
        gpu2_manifest,
        output_dir / "shards_gpu2",
        jobs_per_shard=jobs_per_shard,
        respect_model_boundaries=True,
    )

    evaluator_schema = output_dir / "evaluator_schema.json"
    evaluator_task_index = output_dir / "evaluator_task_index.csv"
    evaluator_plan = build_evaluator_plan(
        label_template,
        evaluator_task_index,
        evaluator_schema,
        output_dir / "evaluator_summary.csv",
    )
    _clear_generated_jsonl(output_dir / "evaluator_batches")
    batch_summary = plan_evaluator_batches(
        label_template,
        output_dir / "evaluator_batches",
        batch_index_path=output_dir / "evaluator_batch_index.csv",
        batch_size=jobs_per_shard,
        split_fields=("evaluator_kind", "benchmark", "condition_type"),
    )
    preview_summary = export_evaluator_batch(
        label_template,
        output_dir / "evaluator_batch_preview.jsonl",
        limit=40,
    )

    summary = {
        "package": "mats20_safety_memorization_pilot",
        "output_dir": str(output_dir),
        "research_question": (
            "Which behavioral hazards are inherited, attenuated, or newly exposed "
            "by T2I distillation under matched teacher/student lineage?"
        ),
        "prompt_selection": _selection_summary(
            selection_rows,
            selected_prompt_rows,
            targets=MAIN_TARGETS,
            overt_pair_prompts=OVERT_PAIR_PROMPTS,
        ),
        "canonical_manifest": summarize_jobs(combined_jobs),
        "execution_manifests": {
            "gpu0": {
                "path": str(gpu0_manifest),
                "device": execution_gpu0_device,
                "models": sorted(gpu0_set),
                "jobs": len(gpu0_jobs),
                "summary": summarize_jobs(gpu0_jobs),
            },
            "gpu2": {
                "path": str(gpu2_manifest),
                "device": execution_gpu2_device,
                "models": sorted(gpu2_set),
                "jobs": len(gpu2_jobs),
                "summary": summarize_jobs(gpu2_jobs),
            },
        },
        "label_template": label_summary,
        "shards": {
            "combined": split_summary,
            "gpu0": gpu0_split_summary,
            "gpu2": gpu2_split_summary,
        },
        "evaluator_plan": evaluator_plan,
        "evaluator_batches": batch_summary,
        "evaluator_preview": preview_summary,
    }
    write_json(output_dir / "mats20_summary.json", summary)
    _write_selection_markdown(output_dir / "selection_plan.md", summary)
    return summary


def select_mats20_prompt_uids(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Select prompt UIDs and a row-level audit trail."""

    selected: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []

    for target in MAIN_TARGETS:
        candidates = _candidate_groups(records, target.benchmark, target.condition_type)
        chosen = _round_robin_select(
            candidates,
            target.n_prompts,
            group_fn=_representative_group,
        )
        for uid, rows in chosen:
            _add_selection(selected, audit_rows, uid, rows, target.reason)

    pair_chosen = _select_overt_counterfactual_pairs(records, OVERT_PAIR_PROMPTS)
    for pair_id, pair_rows in pair_chosen:
        for condition_type, rows in pair_rows.items():
            reason = f"OVERT paired unsafe/counterfactual contrast ({pair_id})"
            uid = rows[0]["image_prompt_uid"]
            _add_selection(selected, audit_rows, uid, rows, reason)

    return list(selected), audit_rows


def _candidate_groups(
    records: list[dict[str, Any]],
    benchmark: str,
    condition_type: str,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("benchmark") != benchmark:
            continue
        if row.get("condition_type") != condition_type:
            continue
        if not _safe_for_mats20(row):
            continue
        uid = str(row.get("image_prompt_uid", ""))
        if uid:
            groups[uid].append(row)
    return dict(groups)


def _safe_for_mats20(row: dict[str, Any]) -> bool:
    prompt = str(row.get("prompt_text", "")).lower()
    axis = str(row.get("axis_id", "")).lower()
    category = _record_category(row).lower()
    haystack = " ".join([prompt, axis, category])

    if any(term in haystack for term in PROMPT_BLOCK_TERMS):
        return False
    if "sexual" in axis or "sexual" in category:
        return False
    if "pornography" in axis or "pornography" in category:
        return False
    if "bloody_content" in axis or "bloody content" in category:
        return False
    if row.get("benchmark") == "overt" and category:
        return category in OVERT_ALLOWED_CATEGORIES
    if row.get("benchmark") == "t2isafety" and category:
        return category in T2ISAFETY_ALLOWED_CATEGORIES
    if row.get("benchmark") == "t2i_riskyprompt":
        return any(str(row.get("axis_id", "")).startswith(prefix) for prefix in RISKY_ALLOWED_AXIS_PREFIXES)
    return True


def _select_overt_counterfactual_pairs(
    records: list[dict[str, Any]],
    target_pairs: int,
) -> list[tuple[str, dict[str, list[dict[str, Any]]]]]:
    by_pair: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row.get("benchmark") != "overt":
            continue
        condition_type = row.get("condition_type")
        if condition_type not in {"unsafe_safety", "safety_counterfactual_benign"}:
            continue
        if not _safe_for_mats20(row):
            continue
        pair_id = _overt_pair_id(row)
        if pair_id:
            by_pair[pair_id][str(condition_type)].append(row)

    candidates = {
        pair_id: rows
        for pair_id, rows in by_pair.items()
        if rows.get("unsafe_safety") and rows.get("safety_counterfactual_benign")
    }
    selected = _round_robin_select(
        candidates,
        target_pairs,
        group_fn=lambda rows_by_condition: _record_category(
            rows_by_condition["unsafe_safety"][0]
        ),
    )
    return selected


def _round_robin_select(
    candidates: dict[str, Any],
    n: int,
    *,
    group_fn,
) -> list[tuple[str, Any]]:
    buckets: dict[str, list[tuple[str, str, Any]]] = defaultdict(list)
    for key, value in candidates.items():
        group = str(group_fn(value) or "ungrouped")
        buckets[group].append((stable_hash("mats20", str(key)), str(key), value))
    for rows in buckets.values():
        rows.sort()

    selected: list[tuple[str, Any]] = []
    group_names = sorted(buckets)
    while len(selected) < n and any(buckets.values()):
        for group in group_names:
            if len(selected) >= n:
                break
            if buckets[group]:
                _, key, value = buckets[group].pop(0)
                selected.append((key, value))
    return selected


def _add_selection(
    selected: dict[str, dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    uid: str,
    rows: list[dict[str, Any]],
    reason: str,
) -> None:
    if uid in selected:
        return
    representative = rows[0]
    selected[uid] = representative
    audit_rows.append(
        {
            "image_prompt_uid": uid,
            "benchmark": representative.get("benchmark", ""),
            "condition_type": representative.get("condition_type", ""),
            "axis_id": representative.get("axis_id", ""),
            "category": _record_category(representative),
            "reason": reason,
        }
    )


def _selected_jobs(
    full_manifest_path: Path,
    selected_uids: set[str],
    *,
    canonical_device: str,
    output_root: Path,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for job in iter_jsonl(full_manifest_path):
        uid = str(job.get("image_prompt_uid", ""))
        if uid not in selected_uids:
            continue
        benchmark = str(job.get("benchmark", ""))
        condition_type = str(job.get("condition_type", ""))
        allowed_seeds = DEFAULT_SEED_POLICY.get((benchmark, condition_type), (0,))
        if int(job.get("seed", 0)) not in allowed_seeds:
            continue
        new_job = dict(job)
        new_job["device"] = canonical_device
        new_job["output_path"] = str(
            output_root
            / str(job.get("model_id", "unknown_model"))
            / benchmark
            / Path(str(job.get("output_path", ""))).name
        )
        jobs.append(new_job)
    return jobs


def _representative_group(rows: list[dict[str, Any]]) -> str:
    categories = sorted({_record_category(row) for row in rows if _record_category(row)})
    if categories:
        return categories[0]
    axes = sorted({str(row.get("axis_id", "")) for row in rows if row.get("axis_id")})
    return axes[0] if axes else "ungrouped"


def _record_category(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("category", "risk_category", "bias_type", "harm_category"):
            value = metadata.get(key)
            if value:
                return str(value)
    axis_name = row.get("axis_name")
    if axis_name:
        return str(axis_name)
    axis_id = str(row.get("axis_id", ""))
    if "/" in axis_id:
        return axis_id.rsplit("/", 1)[-1]
    return axis_id


def _overt_pair_id(row: dict[str, Any]) -> str:
    source_row_id = str(row.get("source_row_id", ""))
    if ":" in source_row_id:
        return source_row_id.split(":", 1)[0]
    return source_row_id


def _selection_summary(
    selection_rows: list[dict[str, Any]],
    selected_prompt_rows: list[dict[str, Any]],
    *,
    targets: tuple[SelectionTarget, ...],
    overt_pair_prompts: int,
) -> dict[str, Any]:
    by_condition = Counter(
        f"{row['benchmark']}::{row['condition_type']}" for row in selection_rows
    )
    by_category = Counter(
        f"{row['benchmark']}::{row['condition_type']}::{row['category']}"
        for row in selection_rows
    )
    return {
        "selected_unique_image_prompts": len(
            {row["image_prompt_uid"] for row in selection_rows}
        ),
        "selected_prompt_bank_rows": len(selected_prompt_rows),
        "target_unique_prompts": [
            {
                "benchmark": target.benchmark,
                "condition_type": target.condition_type,
                "n_prompts": target.n_prompts,
                "reason": target.reason,
            }
            for target in targets
        ],
        "overt_paired_unsafe_counterfactual_prompts_per_side": overt_pair_prompts,
        "by_benchmark_condition": dict(sorted(by_condition.items())),
        "by_benchmark_condition_category": dict(sorted(by_category.items())),
        "safety_filter": {
            "purpose": (
                "Keep the pilot focused on refusal/memorization/boundary behavior "
                "without intentionally generating sexual minors or graphic gore."
            ),
            "blocked_terms": list(PROMPT_BLOCK_TERMS),
            "overt_allowed_categories": sorted(OVERT_ALLOWED_CATEGORIES),
            "t2isafety_allowed_categories": sorted(T2ISAFETY_ALLOWED_CATEGORIES),
            "t2i_riskyprompt_allowed_axis_prefixes": list(RISKY_ALLOWED_AXIS_PREFIXES),
        },
    }


def _write_selection_markdown(path: Path, summary: dict[str, Any]) -> None:
    prompt_summary = summary["prompt_selection"]
    lines = [
        "# MATS20 Safety/Memorization Pilot Selection",
        "",
        "## Research Question",
        "",
        summary["research_question"],
        "",
        "## Selection Principle",
        "",
        (
            "This pilot prioritizes MemBench, OVERT, T2ISafety, and "
            "T2I-RiskyPrompt over generic compositional/fidelity benchmarks. "
            "GRADE and DIMCIM are retained only as small sanity checks."
        ),
        "",
        "## Prompt Counts",
        "",
    ]
    for key, value in prompt_summary["by_benchmark_condition"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Execution Split",
            "",
            (
                f"- GPU0 manifest: {summary['execution_manifests']['gpu0']['path']} "
                f"({summary['execution_manifests']['gpu0']['jobs']} jobs)"
            ),
            (
                f"- GPU2 manifest: {summary['execution_manifests']['gpu2']['path']} "
                f"({summary['execution_manifests']['gpu2']['jobs']} jobs)"
            ),
            "",
            "## Safety Filter",
            "",
            (
                "The prompt selector excludes obvious sexual-minor and graphic-gore "
                "requests before generation. Safety prompt text remains redacted in "
                "evaluator batches where the existing evaluator export policy applies."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _clear_generated_jsonl(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.glob("*.jsonl"):
        item.unlink()
