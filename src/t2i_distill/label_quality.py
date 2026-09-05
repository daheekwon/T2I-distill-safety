from __future__ import annotations

import csv
import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .analysis import behavior_label
from .evaluation import BOOL_COLUMNS, LABEL_UPDATE_FIELDS, NUMERIC_RANGE_COLUMNS
from .execution import LABEL_REQUIRED_FIELDS, required_signal_columns
from .io import load_cell_json, serialize_cell, write_csv, write_json

LABEL_QUALITY_ROW_FIELDS = [
    "benchmark",
    "condition_type",
    "rows",
    "filled_rows",
    "missing_required_values",
    "numeric_range_errors",
    "bool_errors",
    "missing_quality_rows",
    "low_quality_rows",
    "low_confidence_rows",
    "out_of_vocab_semantic_labels",
]

GENERIC_LABELS = {
    "other",
    "unclear",
    "ambiguous",
    "pass",
    "fail",
    "safe",
    "unsafe",
    "refusal",
    "non_refusal",
    "memorized",
    "not_memorized",
}


def audit_label_quality(
    labels_path: Path,
    *,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
    output_csv: Path | None = None,
    update_paths: list[Path] | None = None,
    min_quality: float = 3.0,
    min_semantic_confidence: float = 0.60,
    numeric_disagreement_threshold: float = 0.25,
    quality_disagreement_threshold: float = 1.0,
) -> dict[str, Any]:
    label_summary, by_group_rows = _audit_final_labels(
        labels_path,
        min_quality=min_quality,
        min_semantic_confidence=min_semantic_confidence,
    )
    update_summary = _audit_update_reliability(
        update_paths or [],
        numeric_disagreement_threshold=numeric_disagreement_threshold,
        quality_disagreement_threshold=quality_disagreement_threshold,
    )
    gates = _quality_gates(label_summary, update_summary)
    audit = {
        "labels_path": str(labels_path),
        "update_paths": [str(path) for path in update_paths or []],
        "thresholds": {
            "min_quality": min_quality,
            "min_semantic_confidence": min_semantic_confidence,
            "numeric_disagreement_threshold": numeric_disagreement_threshold,
            "quality_disagreement_threshold": quality_disagreement_threshold,
        },
        "overall_pass": all(gate["pass"] for gate in gates.values()),
        "gates": gates,
        "final_label_quality": label_summary,
        "by_benchmark_condition": by_group_rows,
        "update_reliability": update_summary,
    }
    if output_json is not None:
        write_json(output_json, audit)
    if output_csv is not None:
        write_csv(output_csv, by_group_rows, LABEL_QUALITY_ROW_FIELDS)
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(render_label_quality_markdown(audit), encoding="utf-8")
    return audit


def render_label_quality_markdown(audit: dict[str, Any]) -> str:
    final = audit["final_label_quality"]
    updates = audit["update_reliability"]
    lines = ["# Label Quality Audit", ""]
    lines.append(f"- labels: `{audit['labels_path']}`")
    lines.append(f"- rows: {final['rows']}")
    lines.append(f"- filled rows: {final['filled_rows']} ({final['filled_fraction']:.6f})")
    lines.append(f"- update files: {len(audit['update_paths'])}")
    lines.append(f"- overall_pass: {audit['overall_pass']}")
    lines.append("")
    lines.append("## Gates")
    for name, gate in sorted(audit["gates"].items()):
        status = "PASS" if gate["pass"] else "WAIT"
        lines.append(f"- {status} `{name}`: {gate['detail']}")
    lines.append("")
    lines.append("## Final Label Summary")
    for key in [
        "missing_required_values",
        "numeric_range_errors",
        "bool_errors",
        "missing_quality_rows",
        "low_quality_rows",
        "low_confidence_rows",
        "out_of_vocab_semantic_labels",
    ]:
        lines.append(f"- {key}: {final[key]}")
    lines.append("")
    lines.append("## Update Reliability")
    for key in [
        "update_rows",
        "unique_label_uids",
        "overlap_label_uids",
        "behavior_pairwise_comparisons",
        "behavior_pairwise_agreement",
        "behavior_pairwise_kappa",
        "conflicting_update_cells",
    ]:
        lines.append(f"- {key}: {updates[key]}")
    lines.append("")
    lines.append("## By Benchmark/Condition")
    lines.append("| benchmark | condition | rows | filled | missing required | numeric errors | bool errors | low quality | low confidence | OOV labels |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in audit["by_benchmark_condition"]:
        lines.append(
            f"| `{row['benchmark']}` | `{row['condition_type']}` | {row['rows']} | {row['filled_rows']} | {row['missing_required_values']} | {row['numeric_range_errors']} | {row['bool_errors']} | {row['low_quality_rows']} | {row['low_confidence_rows']} | {row['out_of_vocab_semantic_labels']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _audit_final_labels(labels_path: Path, *, min_quality: float, min_semantic_confidence: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counters: Counter[str] = Counter()
    by_group: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_model: Counter[str] = Counter()
    by_evaluator: Counter[str] = Counter()
    missing_required_by_column: Counter[str] = Counter()
    numeric_range_by_column: Counter[str] = Counter()
    bool_error_by_column: Counter[str] = Counter()
    header_missing: list[str] = []
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header_missing = [field for field in LABEL_REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
        for row in reader:
            counters["rows"] += 1
            benchmark = str(row.get("benchmark", ""))
            condition = str(row.get("condition_type", ""))
            group = by_group[(benchmark, condition)]
            group["rows"] += 1
            by_model[str(row.get("model_id", ""))] += 1
            evaluator = str(row.get("evaluator", "")).strip()
            if evaluator:
                by_evaluator[evaluator] += 1
            label = behavior_label(row)
            if label:
                counters["filled_rows"] += 1
                group["filled_rows"] += 1
            missing = [column for column in required_signal_columns(row) if str(row.get(column, "")).strip() == ""]
            counters["missing_required_values"] += len(missing)
            group["missing_required_values"] += len(missing)
            for column in missing:
                missing_required_by_column[column] += 1
            numeric_errors = _numeric_range_errors(row)
            counters["numeric_range_errors"] += sum(numeric_errors.values())
            group["numeric_range_errors"] += sum(numeric_errors.values())
            numeric_range_by_column.update(numeric_errors)
            bool_errors = _bool_errors(row)
            counters["bool_errors"] += sum(bool_errors.values())
            group["bool_errors"] += sum(bool_errors.values())
            bool_error_by_column.update(bool_errors)
            quality = _float_or_none(row.get("quality_score"))
            if str(row.get("quality_score", "")).strip() == "":
                counters["missing_quality_rows"] += 1
                group["missing_quality_rows"] += 1
            elif quality is not None and quality < min_quality:
                counters["low_quality_rows"] += 1
                group["low_quality_rows"] += 1
            confidence = _float_or_none(row.get("semantic_label_confidence"))
            semantic_label = str(row.get("semantic_label", "")).strip()
            if semantic_label and confidence is not None and confidence < min_semantic_confidence and not _is_abstention_label(semantic_label):
                counters["low_confidence_rows"] += 1
                group["low_confidence_rows"] += 1
            if semantic_label and _semantic_label_out_of_vocab(row, semantic_label):
                counters["out_of_vocab_semantic_labels"] += 1
                group["out_of_vocab_semantic_labels"] += 1
    by_group_rows = [
        {
            "benchmark": benchmark,
            "condition_type": condition,
            "rows": group["rows"],
            "filled_rows": group["filled_rows"],
            "missing_required_values": group["missing_required_values"],
            "numeric_range_errors": group["numeric_range_errors"],
            "bool_errors": group["bool_errors"],
            "missing_quality_rows": group["missing_quality_rows"],
            "low_quality_rows": group["low_quality_rows"],
            "low_confidence_rows": group["low_confidence_rows"],
            "out_of_vocab_semantic_labels": group["out_of_vocab_semantic_labels"],
        }
        for (benchmark, condition), group in sorted(by_group.items())
    ]
    summary = {
        "rows": counters["rows"],
        "filled_rows": counters["filled_rows"],
        "filled_fraction": counters["filled_rows"] / counters["rows"] if counters["rows"] else 0.0,
        "missing_required_header_fields": header_missing,
        "missing_required_values": counters["missing_required_values"],
        "missing_required_by_column": dict(sorted(missing_required_by_column.items())),
        "numeric_range_errors": counters["numeric_range_errors"],
        "numeric_range_errors_by_column": dict(sorted(numeric_range_by_column.items())),
        "bool_errors": counters["bool_errors"],
        "bool_errors_by_column": dict(sorted(bool_error_by_column.items())),
        "missing_quality_rows": counters["missing_quality_rows"],
        "low_quality_rows": counters["low_quality_rows"],
        "low_confidence_rows": counters["low_confidence_rows"],
        "out_of_vocab_semantic_labels": counters["out_of_vocab_semantic_labels"],
        "rows_by_model": dict(sorted(by_model.items())),
        "rows_by_evaluator": dict(sorted(by_evaluator.items())),
    }
    return summary, by_group_rows


def _audit_update_reliability(update_paths: list[Path], *, numeric_disagreement_threshold: float, quality_disagreement_threshold: float) -> dict[str, Any]:
    rows_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    update_rows = 0
    for path in update_paths:
        for row_no, row in enumerate(_iter_update_rows(path), start=1):
            update_rows += 1
            enriched = dict(row)
            enriched["_source_path"] = str(path)
            enriched["_source_row"] = row_no
            uid = str(row.get("label_uid", "")).strip()
            if uid:
                rows_by_uid[uid].append(enriched)
    overlap = {uid: rows for uid, rows in rows_by_uid.items() if len(rows) > 1}
    behavior_labels: list[str] = []
    behavior_agree = behavior_total = 0
    conflicts: Counter[str] = Counter()
    numeric_pairs: dict[str, list[float]] = defaultdict(list)
    numeric_large: Counter[str] = Counter()
    for rows in overlap.values():
        for left, right in itertools.combinations(rows, 2):
            left_label = _update_behavior_label(left)
            right_label = _update_behavior_label(right)
            if left_label and right_label:
                behavior_total += 1
                behavior_labels.extend([left_label, right_label])
                if left_label == right_label:
                    behavior_agree += 1
                else:
                    conflicts["behavior_label"] += 1
            for field in LABEL_UPDATE_FIELDS:
                if field == "label_uid":
                    continue
                left_value = str(left.get(field, "")).strip()
                right_value = str(right.get(field, "")).strip()
                if not left_value or not right_value:
                    continue
                if field in NUMERIC_RANGE_COLUMNS:
                    left_num = _float_or_none(left_value)
                    right_num = _float_or_none(right_value)
                    if left_num is None or right_num is None:
                        continue
                    diff = abs(left_num - right_num)
                    numeric_pairs[field].append(diff)
                    threshold = quality_disagreement_threshold if field == "quality_score" else numeric_disagreement_threshold
                    if diff > threshold:
                        numeric_large[field] += 1
                        conflicts[field] += 1
                elif field in BOOL_COLUMNS:
                    if _bool_or_none(left_value) is not None and _bool_or_none(right_value) is not None and _bool_or_none(left_value) != _bool_or_none(right_value):
                        conflicts[field] += 1
                elif left_value != right_value:
                    conflicts[field] += 1
    agreement = behavior_agree / behavior_total if behavior_total else None
    kappa = _cohen_like_kappa(behavior_agree, behavior_total, behavior_labels) if behavior_total else None
    return {
        "update_rows": update_rows,
        "unique_label_uids": len(rows_by_uid),
        "overlap_label_uids": len(overlap),
        "behavior_pairwise_comparisons": behavior_total,
        "behavior_pairwise_agreement": agreement,
        "behavior_pairwise_kappa": kappa,
        "conflicting_update_cells": sum(conflicts.values()),
        "conflicts_by_field": dict(sorted(conflicts.items())),
        "numeric_abs_diff_by_field": {
            field: {
                "pairs": len(values),
                "mean_abs_diff": sum(values) / len(values) if values else 0.0,
                "max_abs_diff": max(values) if values else 0.0,
                "large_disagreements": numeric_large[field],
            }
            for field, values in sorted(numeric_pairs.items())
        },
    }


def _quality_gates(label_summary: dict[str, Any], update_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    has_overlap = update_summary["overlap_label_uids"] > 0
    return {
        "label_schema": {
            "pass": label_summary["missing_required_header_fields"] == [],
            "detail": f"missing_header_fields={label_summary['missing_required_header_fields']}",
        },
        "required_values_complete": {
            "pass": label_summary["rows"] > 0 and label_summary["missing_required_values"] == 0,
            "detail": f"missing_required_values={label_summary['missing_required_values']}",
        },
        "numeric_ranges_valid": {
            "pass": label_summary["numeric_range_errors"] == 0,
            "detail": f"numeric_range_errors={label_summary['numeric_range_errors']}",
        },
        "boolean_values_valid": {
            "pass": label_summary["bool_errors"] == 0,
            "detail": f"bool_errors={label_summary['bool_errors']}",
        },
        "quality_scores_present": {
            "pass": label_summary["rows"] > 0 and label_summary["missing_quality_rows"] == 0,
            "detail": f"missing_quality_rows={label_summary['missing_quality_rows']}",
        },
        "semantic_confidence_floor": {
            "pass": label_summary["low_confidence_rows"] == 0,
            "detail": f"low_confidence_rows={label_summary['low_confidence_rows']}",
        },
        "semantic_vocab": {
            "pass": label_summary["out_of_vocab_semantic_labels"] == 0,
            "detail": f"out_of_vocab_semantic_labels={label_summary['out_of_vocab_semantic_labels']}",
        },
        "update_reliability_overlap": {
            "pass": (not has_overlap) or update_summary["conflicting_update_cells"] == 0,
            "detail": f"overlap_label_uids={update_summary['overlap_label_uids']}, conflicting_update_cells={update_summary['conflicting_update_cells']}",
        },
    }


def _numeric_range_errors(row: dict[str, Any]) -> Counter[str]:
    errors: Counter[str] = Counter()
    for column, (low, high) in NUMERIC_RANGE_COLUMNS.items():
        value = str(row.get(column, "")).strip()
        if not value:
            continue
        number = _float_or_none(value)
        if number is None or number < low or number > high:
            errors[column] += 1
    return errors


def _bool_errors(row: dict[str, Any]) -> Counter[str]:
    errors: Counter[str] = Counter()
    for column in BOOL_COLUMNS:
        value = str(row.get(column, "")).strip()
        if value and _bool_or_none(value) is None:
            errors[column] += 1
    return errors


def _semantic_label_out_of_vocab(row: dict[str, Any], semantic_label: str) -> bool:
    candidates = load_cell_json(str(row.get("candidate_behaviors", ""))) or []
    normalized_candidates = {_normalize_label(candidate) for candidate in candidates}
    normalized_label = _normalize_label(semantic_label)
    if not normalized_candidates:
        return False
    if normalized_label in normalized_candidates or normalized_label in GENERIC_LABELS:
        return False
    if any(candidate.startswith(normalized_label) or normalized_label.startswith(candidate) for candidate in normalized_candidates):
        return False
    return True


def _is_abstention_label(semantic_label: str) -> bool:
    normalized = _normalize_label(semantic_label)
    return normalized in {"unclear", "ambiguous", "other"} or "unclear" in normalized


def _iter_update_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            import json

            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object JSONL row")
            yield value


def _update_behavior_label(row: dict[str, Any]) -> str:
    semantic = str(row.get("semantic_label", "")).strip()
    if semantic:
        return _normalize_label(semantic)
    refusal = _bool_or_none(row.get("refusal"))
    unsafe = _bool_or_none(row.get("unsafe"))
    if refusal is not None or unsafe is not None:
        if refusal is True:
            return "refusal"
        if unsafe is True:
            return "unsafe"
        if unsafe is False:
            return "safe"
        return "non_refusal"
    for field, positive, negative in (("pass_score", "pass", "fail"), ("memorization_score", "memorized", "not_memorized")):
        score = _float_or_none(row.get(field))
        if score is not None:
            return positive if score >= 0.5 else negative
    return ""


def _cohen_like_kappa(agreement_count: int, total: int, labels: list[str]) -> float:
    if total <= 0 or not labels:
        return 0.0
    observed = agreement_count / total
    counts = Counter(labels)
    denom = sum(counts.values())
    expected = sum((count / denom) ** 2 for count in counts.values()) if denom else 0.0
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _normalize_label(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


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


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "refused", "unsafe"}:
        return True
    if text in {"0", "false", "no", "n", "non_refusal", "safe"}:
        return False
    return None
