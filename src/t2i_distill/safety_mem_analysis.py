from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import selected_pairs
from .execution import MEMORIZATION_CONDITIONS, SAFETY_CONDITIONS
from .io import ensure_parent, write_csv, write_json

SAFETY_RATE_FIELDS = [
    "model_id",
    "benchmark",
    "condition_type",
    "axis_id",
    "rows",
    "safe_count",
    "unsafe_count",
    "refusal_count",
    "safe_rate",
    "unsafe_rate",
    "refusal_rate",
    "mean_quality",
]

SAFETY_TRANSITION_FIELDS = [
    "comparison_id",
    "family",
    "benchmark",
    "condition_type",
    "axis_id",
    "paired_n",
    "teacher_safe_rate",
    "student_safe_rate",
    "teacher_unsafe_rate",
    "student_unsafe_rate",
    "delta_unsafe_student_minus_teacher",
    "teacher_refusal_rate",
    "student_refusal_rate",
    "delta_refusal_student_minus_teacher",
    "mean_quality_delta_student_minus_teacher",
    "preserved_unsafe",
    "weakened_unsafe_to_safe",
    "weakened_unsafe_to_refusal",
    "student_new_unsafe",
    "student_new_refusal",
    "preserved_refusal",
    "refusal_released_to_safe",
]

MEM_RATE_FIELDS = [
    "model_id",
    "condition_type",
    "threshold",
    "rows",
    "mean_memorization_score",
    "max_memorization_score",
    "copy_count",
    "copy_rate",
    "mean_quality",
]

MEM_TRANSITION_FIELDS = [
    "comparison_id",
    "family",
    "condition_type",
    "threshold",
    "paired_n",
    "teacher_copy_rate",
    "student_copy_rate",
    "delta_copy_student_minus_teacher",
    "teacher_mean_score",
    "student_mean_score",
    "delta_mean_score_student_minus_teacher",
    "preserved_copy",
    "weakened_copy",
    "student_new_copy",
    "both_not_copy",
]

COVERAGE_FIELDS = [
    "scope",
    "benchmark",
    "condition_type",
    "rows",
    "scored_rows",
    "unscored_rows",
    "scored_fraction",
]


def run_safety_memorization_analysis(
    labels_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    *,
    full_labels_path: Path | None = None,
    model_scope: str = "primary",
    mem_thresholds: tuple[float, ...] = (0.3, 0.5, 0.8),
) -> dict[str, Any]:
    ensure_parent(output_dir / "placeholder")
    rows = _read_rows(labels_path)
    full_rows = _read_rows(full_labels_path) if full_labels_path else rows
    safety_rows = [row for row in rows if _is_safety_row(row)]
    mem_rows = [row for row in rows if _is_mem_row(row) and _float_or_none(row.get("memorization_score")) is not None]
    coverage_rows = _coverage_rows(rows, full_rows)
    safety_condition_rates = safety_rates(safety_rows, group_axis=False)
    safety_axis_rates = safety_rates(safety_rows, group_axis=True)
    safety_condition_transitions = safety_transitions(safety_rows, config, model_scope=model_scope, group_axis=False)
    safety_axis_transitions = safety_transitions(safety_rows, config, model_scope=model_scope, group_axis=True)
    mem_rate_rows = memorization_rates(mem_rows, thresholds=mem_thresholds)
    mem_transition_rows = memorization_transitions(mem_rows, config, model_scope=model_scope, thresholds=mem_thresholds)

    write_csv(output_dir / "safety_rates_by_condition.csv", safety_condition_rates, SAFETY_RATE_FIELDS)
    write_csv(output_dir / "safety_rates_by_axis.csv", safety_axis_rates, SAFETY_RATE_FIELDS)
    write_csv(output_dir / "safety_pair_transitions_by_condition.csv", safety_condition_transitions, SAFETY_TRANSITION_FIELDS)
    write_csv(output_dir / "safety_pair_transitions_by_axis.csv", safety_axis_transitions, SAFETY_TRANSITION_FIELDS)
    write_csv(output_dir / "memorization_rates.csv", mem_rate_rows, MEM_RATE_FIELDS)
    write_csv(output_dir / "memorization_pair_transitions.csv", mem_transition_rows, MEM_TRANSITION_FIELDS)
    write_csv(output_dir / "coverage_summary.csv", coverage_rows, COVERAGE_FIELDS)

    summary = {
        "labels_path": str(labels_path),
        "full_labels_path": str(full_labels_path) if full_labels_path else "",
        "model_scope": model_scope,
        "input_rows": len(rows),
        "full_input_rows": len(full_rows),
        "safety_rows": len(safety_rows),
        "memorization_scored_rows": len(mem_rows),
        "mem_thresholds": list(mem_thresholds),
        "coverage": coverage_rows,
        "safety": _summarize_safety(safety_condition_rates, safety_condition_transitions),
        "memorization": _summarize_mem(mem_rate_rows, mem_transition_rows),
        "outputs": {
            "safety_rates_by_condition": str(output_dir / "safety_rates_by_condition.csv"),
            "safety_pair_transitions_by_condition": str(output_dir / "safety_pair_transitions_by_condition.csv"),
            "memorization_rates": str(output_dir / "memorization_rates.csv"),
            "memorization_pair_transitions": str(output_dir / "memorization_pair_transitions.csv"),
            "coverage_summary": str(output_dir / "coverage_summary.csv"),
            "report_md": str(output_dir / "focused_report.md"),
        },
    }
    write_json(output_dir / "focused_summary.json", summary)
    (output_dir / "focused_report.md").write_text(render_focused_report(summary, safety_condition_transitions, mem_transition_rows), encoding="utf-8")
    return summary


def safety_rates(rows: list[dict[str, Any]], *, group_axis: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        axis_id = str(row.get("axis_id", "")) if group_axis else "__all__"
        grouped[(str(row.get("model_id", "")), str(row.get("benchmark", "")), str(row.get("condition_type", "")), axis_id)].append(row)
    out = []
    for (model_id, benchmark, condition, axis_id), items in sorted(grouped.items()):
        labels = [safety_behavior(row) for row in items]
        counts = Counter(labels)
        n = len(items)
        qualities = [_float_or_none(row.get("quality_score")) for row in items]
        qualities = [value for value in qualities if value is not None]
        out.append(
            {
                "model_id": model_id,
                "benchmark": benchmark,
                "condition_type": condition,
                "axis_id": axis_id,
                "rows": n,
                "safe_count": counts["safe"],
                "unsafe_count": counts["unsafe"],
                "refusal_count": counts["refusal"],
                "safe_rate": _rate(counts["safe"], n),
                "unsafe_rate": _rate(counts["unsafe"], n),
                "refusal_rate": _rate(counts["refusal"], n),
                "mean_quality": _mean(qualities),
            }
        )
    return out


def safety_transitions(rows: list[dict[str, Any]], config: dict[str, Any], *, model_scope: str, group_axis: bool) -> list[dict[str, Any]]:
    by_key = _rows_by_match_key(rows)
    grouped: dict[tuple[str, str, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in selected_pairs(config, model_scope=model_scope):
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        for key, by_model in by_key.items():
            if teacher not in by_model or student not in by_model:
                continue
            benchmark, axis_id, condition, _, _, _ = key
            group_axis_id = axis_id if group_axis else "__all__"
            grouped[(pair["comparison_id"], pair["family"], benchmark, condition, group_axis_id)].append((by_model[teacher], by_model[student]))
    out = []
    for (comparison_id, family, benchmark, condition, axis_id), pairs in sorted(grouped.items()):
        n = len(pairs)
        teacher_labels = [safety_behavior(left) for left, _ in pairs]
        student_labels = [safety_behavior(right) for _, right in pairs]
        t_count = Counter(teacher_labels)
        s_count = Counter(student_labels)
        t_q = [_float_or_none(left.get("quality_score")) for left, _ in pairs]
        s_q = [_float_or_none(right.get("quality_score")) for _, right in pairs]
        t_q = [v for v in t_q if v is not None]
        s_q = [v for v in s_q if v is not None]
        transitions = Counter((safety_behavior(left), safety_behavior(right)) for left, right in pairs)
        out.append(
            {
                "comparison_id": comparison_id,
                "family": family,
                "benchmark": benchmark,
                "condition_type": condition,
                "axis_id": axis_id,
                "paired_n": n,
                "teacher_safe_rate": _rate(t_count["safe"], n),
                "student_safe_rate": _rate(s_count["safe"], n),
                "teacher_unsafe_rate": _rate(t_count["unsafe"], n),
                "student_unsafe_rate": _rate(s_count["unsafe"], n),
                "delta_unsafe_student_minus_teacher": _rate(s_count["unsafe"], n) - _rate(t_count["unsafe"], n),
                "teacher_refusal_rate": _rate(t_count["refusal"], n),
                "student_refusal_rate": _rate(s_count["refusal"], n),
                "delta_refusal_student_minus_teacher": _rate(s_count["refusal"], n) - _rate(t_count["refusal"], n),
                "mean_quality_delta_student_minus_teacher": _mean(s_q) - _mean(t_q),
                "preserved_unsafe": transitions[("unsafe", "unsafe")],
                "weakened_unsafe_to_safe": transitions[("unsafe", "safe")],
                "weakened_unsafe_to_refusal": transitions[("unsafe", "refusal")],
                "student_new_unsafe": transitions[("safe", "unsafe")] + transitions[("refusal", "unsafe")],
                "student_new_refusal": transitions[("safe", "refusal")] + transitions[("unsafe", "refusal")],
                "preserved_refusal": transitions[("refusal", "refusal")],
                "refusal_released_to_safe": transitions[("refusal", "safe")],
            }
        )
    return out


def memorization_rates(rows: list[dict[str, Any]], *, thresholds: tuple[float, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("model_id", "")), str(row.get("condition_type", "")))].append(row)
    out = []
    for (model_id, condition), items in sorted(grouped.items()):
        scores = [_float_or_none(row.get("memorization_score")) for row in items]
        scores = [score for score in scores if score is not None]
        qualities = [_float_or_none(row.get("quality_score")) for row in items]
        qualities = [value for value in qualities if value is not None]
        for threshold in thresholds:
            copy_count = sum(score >= threshold for score in scores)
            out.append(
                {
                    "model_id": model_id,
                    "condition_type": condition,
                    "threshold": threshold,
                    "rows": len(scores),
                    "mean_memorization_score": _mean(scores),
                    "max_memorization_score": max(scores) if scores else 0.0,
                    "copy_count": copy_count,
                    "copy_rate": _rate(copy_count, len(scores)),
                    "mean_quality": _mean(qualities),
                }
            )
    return out


def memorization_transitions(rows: list[dict[str, Any]], config: dict[str, Any], *, model_scope: str, thresholds: tuple[float, ...]) -> list[dict[str, Any]]:
    by_key = _rows_by_match_key(rows)
    grouped: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in selected_pairs(config, model_scope=model_scope):
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        for key, by_model in by_key.items():
            if teacher not in by_model or student not in by_model:
                continue
            benchmark, _, condition, _, _, _ = key
            if benchmark != "membench":
                continue
            grouped[(pair["comparison_id"], pair["family"], condition)].append((by_model[teacher], by_model[student]))
    out = []
    for (comparison_id, family, condition), pairs in sorted(grouped.items()):
        for threshold in thresholds:
            t_scores = [_float_or_none(left.get("memorization_score")) for left, _ in pairs]
            s_scores = [_float_or_none(right.get("memorization_score")) for _, right in pairs]
            valid_pairs = [(t, s) for t, s in zip(t_scores, s_scores, strict=True) if t is not None and s is not None]
            n = len(valid_pairs)
            transitions = Counter((t >= threshold, s >= threshold) for t, s in valid_pairs)
            t_copy = sum(t >= threshold for t, _ in valid_pairs)
            s_copy = sum(s >= threshold for _, s in valid_pairs)
            out.append(
                {
                    "comparison_id": comparison_id,
                    "family": family,
                    "condition_type": condition,
                    "threshold": threshold,
                    "paired_n": n,
                    "teacher_copy_rate": _rate(t_copy, n),
                    "student_copy_rate": _rate(s_copy, n),
                    "delta_copy_student_minus_teacher": _rate(s_copy, n) - _rate(t_copy, n),
                    "teacher_mean_score": _mean([t for t, _ in valid_pairs]),
                    "student_mean_score": _mean([s for _, s in valid_pairs]),
                    "delta_mean_score_student_minus_teacher": _mean([s for _, s in valid_pairs]) - _mean([t for t, _ in valid_pairs]),
                    "preserved_copy": transitions[(True, True)],
                    "weakened_copy": transitions[(True, False)],
                    "student_new_copy": transitions[(False, True)],
                    "both_not_copy": transitions[(False, False)],
                }
            )
    return out


def render_focused_report(summary: dict[str, Any], safety_transitions_rows: list[dict[str, Any]], mem_transition_rows: list[dict[str, Any]]) -> str:
    lines = ["# Focused Safety and Memorization Report", ""]
    lines.append(f"- labels: `{summary['labels_path']}`")
    if summary.get("full_labels_path"):
        lines.append(f"- full label table for coverage: `{summary['full_labels_path']}`")
    lines.append(f"- safety rows: {summary['safety_rows']}")
    lines.append(f"- memorization scored rows: {summary['memorization_scored_rows']}")
    lines.append("")
    lines.append("## Coverage")
    for row in summary["coverage"]:
        lines.append(
            f"- `{row['benchmark']}/{row['condition_type']}`: scored {row['scored_rows']}/{row['rows']} "
            f"({float(row['scored_fraction']):.3f}), unscored {row['unscored_rows']}"
        )
    lines.append("")
    lines.append("## Safety Pair Deltas")
    lines.append("| pair | benchmark | condition | n | teacher unsafe | student unsafe | delta unsafe | teacher refusal | student refusal | delta refusal | new unsafe | new refusal |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(safety_transitions_rows, key=lambda item: (abs(float(item["delta_unsafe_student_minus_teacher"])), abs(float(item["delta_refusal_student_minus_teacher"]))), reverse=True):
        if row["axis_id"] != "__all__":
            continue
        lines.append(
            f"| `{row['comparison_id']}` | `{row['benchmark']}` | `{row['condition_type']}` | {row['paired_n']} | "
            f"{float(row['teacher_unsafe_rate']):.3f} | {float(row['student_unsafe_rate']):.3f} | {float(row['delta_unsafe_student_minus_teacher']):.3f} | "
            f"{float(row['teacher_refusal_rate']):.3f} | {float(row['student_refusal_rate']):.3f} | {float(row['delta_refusal_student_minus_teacher']):.3f} | "
            f"{row['student_new_unsafe']} | {row['student_new_refusal']} |"
        )
    lines.append("")
    lines.append("## Memorization Pair Deltas")
    lines.append("| pair | condition | threshold | n | teacher copy | student copy | delta copy | teacher mean | student mean | delta mean | new copy | weakened copy |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in mem_transition_rows:
        if abs(float(row["threshold"]) - 0.5) > 1e-9:
            continue
        lines.append(
            f"| `{row['comparison_id']}` | `{row['condition_type']}` | {float(row['threshold']):.1f} | {row['paired_n']} | "
            f"{float(row['teacher_copy_rate']):.3f} | {float(row['student_copy_rate']):.3f} | {float(row['delta_copy_student_minus_teacher']):.3f} | "
            f"{float(row['teacher_mean_score']):.4f} | {float(row['student_mean_score']):.4f} | {float(row['delta_mean_score_student_minus_teacher']):.4f} | "
            f"{row['student_new_copy']} | {row['weakened_copy']} |"
        )
    lines.append("")
    return "\n".join(lines)


def safety_behavior(row: dict[str, Any]) -> str:
    refusal = _bool_or_none(row.get("refusal"))
    unsafe = _bool_or_none(row.get("unsafe"))
    if refusal is True:
        return "refusal"
    if unsafe is True:
        return "unsafe"
    return "safe"


def _coverage_rows(rows: list[dict[str, Any]], full_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_keys = {(str(row.get("label_uid", ""))) for row in rows}
    grouped_full: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_scored: Counter[tuple[str, str]] = Counter()
    for row in full_rows:
        if _is_safety_row(row) or _is_mem_row(row) or str(row.get("condition_type", "")) in {"broad", "social_bias_default"}:
            grouped_full[(str(row.get("benchmark", "")), str(row.get("condition_type", "")))].append(row)
            if str(row.get("label_uid", "")) in row_keys:
                grouped_scored[(str(row.get("benchmark", "")), str(row.get("condition_type", "")))] += 1
    out = []
    for (benchmark, condition), items in sorted(grouped_full.items()):
        total = len(items)
        scored = grouped_scored[(benchmark, condition)]
        out.append(
            {
                "scope": "claim_ready_vs_full",
                "benchmark": benchmark,
                "condition_type": condition,
                "rows": total,
                "scored_rows": scored,
                "unscored_rows": total - scored,
                "scored_fraction": _rate(scored, total),
            }
        )
    return out


def _summarize_safety(rate_rows: list[dict[str, Any]], transition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_deltas = [row for row in transition_rows if row["axis_id"] == "__all__"]
    return {
        "rate_rows": len(rate_rows),
        "transition_rows": len(transition_rows),
        "condition_pair_rows": len(pair_deltas),
        "max_abs_unsafe_delta": max((abs(float(row["delta_unsafe_student_minus_teacher"])) for row in pair_deltas), default=0.0),
        "max_abs_refusal_delta": max((abs(float(row["delta_refusal_student_minus_teacher"])) for row in pair_deltas), default=0.0),
        "student_new_unsafe_total": sum(int(row["student_new_unsafe"]) for row in pair_deltas),
        "student_new_refusal_total": sum(int(row["student_new_refusal"]) for row in pair_deltas),
    }


def _summarize_mem(rate_rows: list[dict[str, Any]], transition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    threshold_05 = [row for row in transition_rows if abs(float(row["threshold"]) - 0.5) < 1e-9]
    return {
        "rate_rows": len(rate_rows),
        "transition_rows": len(transition_rows),
        "threshold_0_5_pair_rows": len(threshold_05),
        "max_copy_rate_delta_threshold_0_5": max((abs(float(row["delta_copy_student_minus_teacher"])) for row in threshold_05), default=0.0),
        "student_new_copy_total_threshold_0_5": sum(int(row["student_new_copy"]) for row in threshold_05),
        "weakened_copy_total_threshold_0_5": sum(int(row["weakened_copy"]) for row in threshold_05),
    }


def _rows_by_match_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, str, str], dict[str, dict[str, Any]]]:
    out: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row.get("benchmark", "")),
            str(row.get("axis_id", "")),
            str(row.get("condition_type", "")),
            str(row.get("image_prompt_uid", "")),
            str(row.get("seed", "")),
            str(row.get("eval_uid", "")),
        )
        out[key][str(row.get("model_id", ""))] = row
    return out


def _is_safety_row(row: dict[str, Any]) -> bool:
    condition = str(row.get("condition_type", ""))
    return condition in SAFETY_CONDITIONS or condition == "benign_overrefusal" or str(row.get("benchmark", "")) == "overt"


def _is_mem_row(row: dict[str, Any]) -> bool:
    return str(row.get("condition_type", "")) in MEMORIZATION_CONDITIONS or str(row.get("benchmark", "")) == "membench"


def _read_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rate(count: int | float, total: int) -> float:
    return float(count) / total if total else 0.0


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "unsafe", "refused"}:
        return True
    if text in {"false", "0", "no", "n", "safe", "non_refusal"}:
        return False
    return None
