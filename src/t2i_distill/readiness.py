from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .analysis import behavior_label
from .config import selected_pairs
from .execution import MEMORIZATION_CONDITIONS, SAFETY_CONDITIONS, required_signal_columns
from .io import write_json

RQ_ORDER = [
    "rq1_sharpening_fit",
    "rq1_sharpening_transfer",
    "rq2_residual_transport",
    "rq3_preference_capability",
    "rq4_survival_principle",
    "safety_inheritance",
    "memorization_retention",
    "unlearning_optional",
]

RQ_DESCRIPTIONS = {
    "rq1_sharpening_fit": "Fit generic sharpening alpha on GRADE and DIMCIM broad conditions.",
    "rq1_sharpening_transfer": "Test whether fitted alpha transfers to composition, knowledge, safety, and memorization benchmarks.",
    "rq2_residual_transport": "Estimate teacher-to-student behavior transition and residual transport matrices.",
    "rq3_preference_capability": "Separate default preference shifts from explicit/capability or information loss.",
    "rq4_survival_principle": "Relate behavior retention to teacher frequency and complexity/domain features.",
    "safety_inheritance": "Audit unsafe compliance, refusal, and over-refusal behavior inheritance.",
    "memorization_retention": "Audit trigger memorization and control-prompt behavior.",
    "unlearning_optional": "Audit HUB/EMMA unlearning claims only when valid lineage rows are included.",
}

BROAD_DIAGNOSTIC_CONDITIONS = {"broad", "benign_overrefusal", "social_bias_default"}
EXPLICIT_DIAGNOSTIC_CONDITIONS = {"explicit", "knowledge_injected", "benchmark_native", "implicit_knowledge", "compositional_native"}


def audit_rq_readiness(
    labels_path: Path,
    config: dict[str, Any],
    *,
    model_scope: str = "primary",
    strict_complete: bool = True,
) -> dict[str, Any]:
    min_quality = float(config.get("analysis", {}).get("quality_controls", {}).get("minimum_quality_score", 0.0))
    pairs = selected_pairs(config, model_scope=model_scope)
    pair_models = {
        pair["comparison_id"]: {
            "teacher": pair["teacher"]["model_id"],
            "student": pair["student"]["model_id"],
            "family": pair["family"],
        }
        for pair in pairs
    }
    state: dict[tuple[str, str], dict[str, Any]] = defaultdict(_empty_rq_pair_state)
    row_counts: Counter[str] = Counter()
    rows = 0

    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            applicable = applicable_rqs(row, config)
            if not applicable:
                continue
            signal = row_signal_state(row, min_quality=min_quality)
            row_counts["rows_seen_by_rq"] += len(applicable)
            model_id = str(row.get("model_id", ""))
            for pair_id, models in pair_models.items():
                role = "teacher" if model_id == models["teacher"] else "student" if model_id == models["student"] else ""
                if not role:
                    continue
                cell = _cell_key(row)
                benchmark = str(row.get("benchmark", ""))
                for rq in applicable:
                    bucket = state[(rq, pair_id)]
                    bucket["family"] = models["family"]
                    bucket["expected_cells"][cell].add(role)
                    bucket["expected_by_benchmark"][benchmark][cell].add(role)
                    bucket["row_count"] += 1
                    bucket["missing_required_values"] += len(signal["missing_required_columns"])
                    bucket["low_quality_rows"] += int(signal["low_quality"])
                    bucket["missing_behavior_rows"] += int(not signal["has_behavior_label"])
                    if signal["ready"]:
                        bucket["ready_cells"][cell].add(role)
                        bucket["ready_by_benchmark"][benchmark][cell].add(role)
                        bucket["ready_row_count"] += 1

    rq_rows = []
    rq_summary: dict[str, Any] = {}
    for rq in RQ_ORDER:
        items = []
        for pair in pairs:
            pair_id = pair["comparison_id"]
            result = _summarize_rq_pair(rq, pair_id, state[(rq, pair_id)], strict_complete=strict_complete)
            items.append(result)
            rq_rows.append(result)
        rq_summary[rq] = {
            "description": RQ_DESCRIPTIONS[rq],
            "ready": all(item["ready"] for item in items),
            "status_counts": dict(Counter(item["status"] for item in items)),
            "matched_ready_cells": sum(item["ready_matched_cells"] for item in items),
            "expected_matched_cells": sum(item["expected_matched_cells"] for item in items),
        }
    overall_ready = all(item["ready"] or item["status"] == "not_applicable" for item in rq_rows)
    return {
        "labels_path": str(labels_path),
        "model_scope": model_scope,
        "strict_complete": strict_complete,
        "minimum_quality_score": min_quality,
        "rows": rows,
        "overall_ready": overall_ready,
        "rq_summary": rq_summary,
        "rq_pair_rows": rq_rows,
    }


def write_rq_readiness_reports(readiness: dict[str, Any], output_json: Path, output_markdown: Path | None = None) -> None:
    write_json(output_json, readiness)
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_rq_readiness_markdown(readiness), encoding="utf-8")


def render_rq_readiness_markdown(readiness: dict[str, Any]) -> str:
    lines = ["# RQ Readiness Audit", ""]
    lines.append(f"- labels: `{readiness['labels_path']}`")
    lines.append(f"- rows: {readiness['rows']}")
    lines.append(f"- minimum_quality_score: {readiness['minimum_quality_score']}")
    lines.append(f"- overall_ready: {readiness['overall_ready']}")
    lines.append("")
    lines.append("## Summary")
    for rq in RQ_ORDER:
        summary = readiness["rq_summary"][rq]
        lines.append(f"- `{rq}`: ready={summary['ready']}, matched={summary['matched_ready_cells']}/{summary['expected_matched_cells']}, statuses={summary['status_counts']}")
    lines.append("")
    lines.append("## Pair Detail")
    lines.append("| rq | comparison_id | status | ready matched | expected matched | missing values | low quality | missing behavior |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in readiness["rq_pair_rows"]:
        lines.append(
            f"| `{row['rq']}` | `{row['comparison_id']}` | {row['status']} | {row['ready_matched_cells']} | {row['expected_matched_cells']} | {row['missing_required_values']} | {row['low_quality_rows']} | {row['missing_behavior_rows']} |"
        )
    lines.append("")
    return "\n".join(lines)


def applicable_rqs(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition_type", ""))
    names: list[str] = []
    if benchmark in set(config.get("analysis", {}).get("fit_sharpening_on", [])) and condition in {"broad", "benchmark_native"}:
        names.append("rq1_sharpening_fit")
    if benchmark in set(config.get("analysis", {}).get("transfer_alpha_to", [])):
        names.append("rq1_sharpening_transfer")
    if not condition.startswith("unlearning_"):
        names.append("rq2_residual_transport")
    if condition in BROAD_DIAGNOSTIC_CONDITIONS or condition in EXPLICIT_DIAGNOSTIC_CONDITIONS:
        names.append("rq3_preference_capability")
    if not condition.startswith("unlearning_"):
        names.append("rq4_survival_principle")
    if condition in SAFETY_CONDITIONS or benchmark == "overt" or condition == "benign_overrefusal":
        names.append("safety_inheritance")
    if benchmark == "membench" or condition in MEMORIZATION_CONDITIONS:
        names.append("memorization_retention")
    if benchmark in {"hub", "emma"} or condition.startswith("unlearning_"):
        names.append("unlearning_optional")
    return names


def row_signal_state(row: dict[str, Any], *, min_quality: float) -> dict[str, Any]:
    missing = [column for column in required_signal_columns(row) if str(row.get(column, "")).strip() == ""]
    quality = _float_or_none(row.get("quality_score"))
    low_quality = quality is not None and quality < min_quality
    has_behavior = bool(behavior_label(row))
    return {
        "required_columns": required_signal_columns(row),
        "missing_required_columns": missing,
        "quality_score": quality,
        "low_quality": low_quality,
        "has_behavior_label": has_behavior,
        "ready": not missing and not low_quality and has_behavior,
    }


def _summarize_rq_pair(rq: str, pair_id: str, bucket: dict[str, Any], *, strict_complete: bool) -> dict[str, Any]:
    expected = _count_matched_cells(bucket["expected_cells"])
    ready = _count_matched_cells(bucket["ready_cells"])
    if expected == 0:
        status = "not_applicable" if rq == "unlearning_optional" else "missing_expected_cells"
        is_ready = rq == "unlearning_optional"
    elif strict_complete and ready < expected:
        status = "incomplete"
        is_ready = False
    elif ready == 0:
        status = "missing_ready_cells"
        is_ready = False
    else:
        status = "ready"
        is_ready = True
    return {
        "rq": rq,
        "description": RQ_DESCRIPTIONS[rq],
        "comparison_id": pair_id,
        "family": bucket.get("family", ""),
        "status": status,
        "ready": is_ready,
        "row_count": bucket["row_count"],
        "ready_row_count": bucket["ready_row_count"],
        "expected_matched_cells": expected,
        "ready_matched_cells": ready,
        "readiness_fraction": ready / expected if expected else 0.0,
        "missing_required_values": bucket["missing_required_values"],
        "low_quality_rows": bucket["low_quality_rows"],
        "missing_behavior_rows": bucket["missing_behavior_rows"],
        "expected_matched_cells_by_benchmark": _matched_by_benchmark(bucket["expected_by_benchmark"]),
        "ready_matched_cells_by_benchmark": _matched_by_benchmark(bucket["ready_by_benchmark"]),
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


def _empty_rq_pair_state() -> dict[str, Any]:
    return {
        "family": "",
        "row_count": 0,
        "ready_row_count": 0,
        "missing_required_values": 0,
        "low_quality_rows": 0,
        "missing_behavior_rows": 0,
        "expected_cells": defaultdict(set),
        "ready_cells": defaultdict(set),
        "expected_by_benchmark": defaultdict(lambda: defaultdict(set)),
        "ready_by_benchmark": defaultdict(lambda: defaultdict(set)),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number
