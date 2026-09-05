from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from statistics import NormalDist
from pathlib import Path
from typing import Any, Iterable

from .config import selected_pairs
from .io import ensure_parent, load_cell_json, stable_hash, write_csv, write_json
from .metrics import (
    fit_alpha,
    hellinger,
    js_divergence,
    normalized_entropy,
    probability_distribution,
    sharpen_distribution,
    spearman,
    total_variation,
    wilson_interval,
)


def run_full_analysis(
    labels_path: Path,
    config: dict[str, Any],
    output_dir: Path,
    *,
    model_scope: str = "primary",
) -> dict[str, Any]:
    ensure_parent(output_dir / "placeholder")
    rows = read_label_csv(labels_path)
    labeled_rows = [row for row in rows if behavior_label(row)]
    analysis_rows, quality_summary, quality_by_model_rows, quality_pair_rows = apply_quality_controls(
        labeled_rows,
        config,
        model_scope=model_scope,
    )

    distribution_rows = distribution_summary(analysis_rows)
    pairwise_rows = pairwise_distribution_shift(analysis_rows, config, model_scope=model_scope)
    paired_outcome_rows = paired_outcome_tests(analysis_rows, config, model_scope=model_scope)
    alpha_rows, alpha_by_pair = sharpening_fits(analysis_rows, config, model_scope=model_scope)
    transfer_rows = sharpening_transfer(analysis_rows, config, alpha_by_pair, model_scope=model_scope)
    residual_rows = residual_transport(analysis_rows, config, model_scope=model_scope)
    dissociation_rows = preference_capability_dissociation(analysis_rows, config, model_scope=model_scope)
    survival_rows, survival_summary = survival_features(analysis_rows, config, model_scope=model_scope)
    survival_regression_rows, survival_regression_summary = survival_regression(survival_rows)

    write_csv(output_dir / "distribution_summary.csv", distribution_rows, DISTRIBUTION_FIELDS)
    write_csv(output_dir / "pairwise_shift.csv", pairwise_rows, PAIRWISE_FIELDS)
    write_csv(output_dir / "paired_outcome_tests.csv", paired_outcome_rows, PAIRED_OUTCOME_FIELDS)
    write_csv(output_dir / "sharpening_fits.csv", alpha_rows, SHARPENING_FIELDS)
    write_csv(output_dir / "sharpening_transfer.csv", transfer_rows, SHARPENING_TRANSFER_FIELDS)
    write_csv(output_dir / "residual_transport.csv", residual_rows, RESIDUAL_FIELDS)
    write_csv(output_dir / "preference_capability.csv", dissociation_rows, DISSOCIATION_FIELDS)
    write_csv(output_dir / "survival_features.csv", survival_rows, SURVIVAL_FIELDS)
    write_csv(output_dir / "survival_regression.csv", survival_regression_rows, SURVIVAL_REGRESSION_FIELDS)
    write_csv(output_dir / "quality_by_model_benchmark.csv", quality_by_model_rows, QUALITY_BY_MODEL_FIELDS)
    write_csv(output_dir / "quality_pair_balance.csv", quality_pair_rows, QUALITY_PAIR_FIELDS)

    evidence = {
        "labels_path": str(labels_path),
        "model_scope": model_scope,
        "input_label_rows": len(rows),
        "usable_label_rows": len(analysis_rows),
        "usable_label_rows_before_quality": len(labeled_rows),
        "quality_control": quality_summary,
        "paired_outcome_tests": {
            "rows": len(paired_outcome_rows),
            "nominal_significant_p_lt_0_05": sum(row["significant_nominal_0_05"] == "true" for row in paired_outcome_rows),
            "bonferroni_significant_0_05": sum(row["significant_bonferroni_0_05"] == "true" for row in paired_outcome_rows),
            "bh_fdr_significant_0_05": sum(row["significant_bh_fdr_0_05"] == "true" for row in paired_outcome_rows),
        },
        "rq1_generic_sharpening": {
            "fit_rows": len(alpha_rows),
            "alpha_by_pair": alpha_by_pair,
            "transfer_rows": len(transfer_rows),
        },
        "rq2_structured_remapping": {
            "residual_rows": len(residual_rows),
            "large_residual_rows_abs_ge_0_20": sum(abs(float(row["residual"])) >= 0.20 for row in residual_rows),
            "nominal_significant_residual_p_lt_0_05": sum(row["significant_nominal_0_05"] == "true" for row in residual_rows),
            "bonferroni_significant_residual_0_05": sum(row["significant_bonferroni_0_05"] == "true" for row in residual_rows),
            "bh_fdr_significant_residual_0_05": sum(row["significant_bh_fdr_0_05"] == "true" for row in residual_rows),
        },
        "rq3_what_is_lost": {
            "dissociation_rows": len(dissociation_rows),
            "preference_shift": sum(row["classification"] == "preference_shift" for row in dissociation_rows),
            "capability_or_information_loss": sum(row["classification"] == "capability_or_information_loss" for row in dissociation_rows),
        },
        "rq4_survival_predictors": survival_summary,
        "rq4_survival_regression": survival_regression_summary,
    }
    write_json(output_dir / "claim_evidence_matrix.json", evidence)
    return evidence


QUALITY_BY_MODEL_FIELDS = [
    "model_id",
    "benchmark",
    "condition_type",
    "rows",
    "kept_rows",
    "missing_quality_rows",
    "low_quality_rows",
    "mean_quality",
]

QUALITY_PAIR_FIELDS = [
    "comparison_id",
    "family",
    "benchmark",
    "condition_type",
    "matched_quality_cells",
    "teacher_mean_quality",
    "student_mean_quality",
    "mean_abs_quality_delta",
    "max_abs_quality_delta",
    "exceeds_config_delta",
]


def apply_quality_controls(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model_scope: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    qc = config.get("analysis", {}).get("quality_controls", {})
    score_column = str(qc.get("quality_score_column", "quality_score"))
    minimum = float(qc.get("minimum_quality_score", 0.0))
    match_delta = float(qc.get("quality_match_delta", 1.0))
    retain_refusal_rows = bool(qc.get("retain_refusal_rows", False))

    kept: list[dict[str, Any]] = []
    stats: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"rows": 0, "kept": 0, "missing": 0, "low": 0, "qualities": []})
    for row in rows:
        key = (row.get("model_id", ""), row.get("benchmark", ""), row.get("condition_type", ""))
        bucket = stats[key]
        bucket["rows"] += 1
        quality = _float_or_none(row.get(score_column))
        if quality is None:
            bucket["missing"] += 1
            continue
        bucket["qualities"].append(quality)
        if quality < minimum:
            bucket["low"] += 1
            if not (retain_refusal_rows and behavior_label(row) == "refusal"):
                continue
        bucket["kept"] += 1
        kept.append(row)

    by_model_rows = []
    for (model_id, benchmark, condition), bucket in sorted(stats.items()):
        by_model_rows.append(
            {
                "model_id": model_id,
                "benchmark": benchmark,
                "condition_type": condition,
                "rows": bucket["rows"],
                "kept_rows": bucket["kept"],
                "missing_quality_rows": bucket["missing"],
                "low_quality_rows": bucket["low"],
                "mean_quality": _mean(bucket["qualities"]),
            }
        )

    quality_pair_rows = _quality_pair_balance(rows, config, score_column, match_delta, model_scope=model_scope)
    summary = {
        "score_column": score_column,
        "minimum_quality_score": minimum,
        "quality_match_delta": match_delta,
        "retain_refusal_rows": retain_refusal_rows,
        "input_labeled_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(rows) - len(kept),
        "missing_quality_rows": sum(row["missing_quality_rows"] for row in by_model_rows),
        "low_quality_rows": sum(row["low_quality_rows"] for row in by_model_rows),
        "pair_balance_rows": len(quality_pair_rows),
        "pair_balance_exceeding_delta": sum(row["exceeds_config_delta"] == "true" for row in quality_pair_rows),
    }
    return kept, summary, by_model_rows, quality_pair_rows


def _quality_pair_balance(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    score_column: str,
    match_delta: float,
    *,
    model_scope: str,
) -> list[dict[str, Any]]:
    by_match: dict[tuple[str, str, str, str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        quality = _float_or_none(row.get(score_column))
        if quality is None:
            continue
        key = (
            row["benchmark"],
            row["axis_id"],
            row["condition_type"],
            row["image_prompt_uid"],
            str(row["seed"]),
            row.get("eval_uid", ""),
        )
        by_match[key][row["model_id"]] = quality

    grouped: dict[tuple[str, str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for pair in selected_pairs(config, model_scope=model_scope):
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        for key, quality_by_model in by_match.items():
            if teacher in quality_by_model and student in quality_by_model:
                benchmark, _, condition, _, _, _ = key
                grouped[(pair["comparison_id"], pair["family"], benchmark, condition)].append((quality_by_model[teacher], quality_by_model[student]))

    out = []
    for (comparison_id, family, benchmark, condition), pairs in sorted(grouped.items()):
        deltas = [abs(t - s) for t, s in pairs]
        out.append(
            {
                "comparison_id": comparison_id,
                "family": family,
                "benchmark": benchmark,
                "condition_type": condition,
                "matched_quality_cells": len(pairs),
                "teacher_mean_quality": _mean([t for t, _ in pairs]),
                "student_mean_quality": _mean([s for _, s in pairs]),
                "mean_abs_quality_delta": _mean(deltas),
                "max_abs_quality_delta": max(deltas) if deltas else 0.0,
                "exceeds_config_delta": "true" if deltas and _mean(deltas) > match_delta else "false",
            }
        )
    return out


def read_label_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed = dict(row)
            parsed["candidate_behaviors"] = load_cell_json(str(row.get("candidate_behaviors", ""))) or []
            rows.append(parsed)
        return rows


def behavior_label(row: dict[str, Any]) -> str:
    label = str(row.get("semantic_label", "")).strip()
    if label:
        return label
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
    score = _float_or_none(row.get("pass_score"))
    if score is not None:
        return "pass" if score >= 0.5 else "fail"
    mem = _float_or_none(row.get("memorization_score"))
    if mem is not None:
        return "memorized" if mem >= 0.5 else "not_memorized"
    return ""


def success_for_target(row: dict[str, Any], target: str) -> bool | None:
    score = _float_or_none(row.get("pass_score"))
    if score is not None:
        return score >= 0.5
    label = behavior_label(row)
    if not label:
        return None
    if target:
        return label == target or label == "pass"
    return label == "pass"


DISTRIBUTION_FIELDS = [
    "model_id",
    "benchmark",
    "axis_id",
    "condition_type",
    "label",
    "count",
    "probability",
    "n",
    "normalized_entropy",
]


def distribution_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["benchmark"], row["axis_id"], row["condition_type"])].append(behavior_label(row))
    out: list[dict[str, Any]] = []
    for (model_id, benchmark, axis_id, condition), labels in sorted(grouped.items()):
        counts = Counter(labels)
        support = sorted(counts)
        dist = probability_distribution(labels, support)
        entropy = normalized_entropy(dist)
        n = sum(counts.values())
        for label in support:
            out.append(
                {
                    "model_id": model_id,
                    "benchmark": benchmark,
                    "axis_id": axis_id,
                    "condition_type": condition,
                    "label": label,
                    "count": counts[label],
                    "probability": dist[label],
                    "n": n,
                    "normalized_entropy": entropy,
                }
            )
    return out


PAIRWISE_FIELDS = [
    "comparison_id",
    "family",
    "benchmark",
    "axis_id",
    "condition_type",
    "teacher_n",
    "student_n",
    "teacher_entropy",
    "student_entropy",
    "entropy_delta_student_minus_teacher",
    "total_variation",
    "js_divergence",
    "hellinger",
    "total_variation_ci_low",
    "total_variation_ci_high",
    "js_divergence_ci_low",
    "js_divergence_ci_high",
    "hellinger_ci_low",
    "hellinger_ci_high",
    "entropy_delta_ci_low",
    "entropy_delta_ci_high",
]


def pairwise_distribution_shift(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model_scope: str,
) -> list[dict[str, Any]]:
    grouped = _labels_by_model_axis(rows)
    out: list[dict[str, Any]] = []
    for pair in selected_pairs(config, model_scope=model_scope):
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        keys = sorted({key[1:] for key in grouped if key[0] in {teacher, student}})
        for key in keys:
            t_labels = grouped.get((teacher, *key), [])
            s_labels = grouped.get((student, *key), [])
            if not t_labels or not s_labels:
                continue
            support = sorted(set(t_labels) | set(s_labels))
            p_t = probability_distribution(t_labels, support)
            p_s = probability_distribution(s_labels, support)
            h_t = normalized_entropy(p_t)
            h_s = normalized_entropy(p_s)
            benchmark, axis_id, condition = key
            cis = _bootstrap_pairwise_metric_cis(
                t_labels,
                s_labels,
                support,
                config,
                salt=f"{pair['comparison_id']}:{benchmark}:{axis_id}:{condition}",
            )
            out.append(
                {
                    "comparison_id": pair["comparison_id"],
                    "family": pair["family"],
                    "benchmark": benchmark,
                    "axis_id": axis_id,
                    "condition_type": condition,
                    "teacher_n": len(t_labels),
                    "student_n": len(s_labels),
                    "teacher_entropy": h_t,
                    "student_entropy": h_s,
                    "entropy_delta_student_minus_teacher": h_s - h_t,
                    "total_variation": total_variation(p_t, p_s),
                    "js_divergence": js_divergence(p_t, p_s),
                    "hellinger": hellinger(p_t, p_s),
                    **cis,
                }
            )
    return out


PAIRED_OUTCOME_FIELDS = [
    "comparison_id",
    "family",
    "benchmark",
    "axis_id",
    "condition_type",
    "outcome_label",
    "paired_n",
    "teacher_successes",
    "student_successes",
    "teacher_rate",
    "student_rate",
    "delta_student_minus_teacher",
    "delta_ci_low",
    "delta_ci_high",
    "both_positive",
    "both_negative",
    "teacher_only",
    "student_only",
    "agreement_rate",
    "discordant_n",
    "exact_sign_p_value",
    "p_value_bonferroni",
    "p_value_bh_fdr",
    "significant_nominal_0_05",
    "significant_bonferroni_0_05",
    "significant_bh_fdr_0_05",
    "effect_direction",
]


def paired_outcome_tests(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model_scope: str,
) -> list[dict[str, Any]]:
    boot = config.get("analysis", {}).get("bootstrap", {})
    confidence = float(boot.get("confidence", 0.95))
    by_match: dict[tuple[str, str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        label = behavior_label(row)
        if not label:
            continue
        key = (
            row["benchmark"],
            row["axis_id"],
            row["condition_type"],
            row["image_prompt_uid"],
            str(row["seed"]),
            row.get("eval_uid", ""),
        )
        by_match[key][row["model_id"]] = row

    out: list[dict[str, Any]] = []
    for pair in selected_pairs(config, model_scope=model_scope):
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        grouped_pairs: dict[tuple[str, str, str], list[tuple[str, str, set[str]]]] = defaultdict(list)
        for key, rows_by_model in by_match.items():
            if teacher not in rows_by_model or student not in rows_by_model:
                continue
            benchmark, axis_id, condition, _, _, _ = key
            teacher_row = rows_by_model[teacher]
            student_row = rows_by_model[student]
            teacher_label = behavior_label(teacher_row)
            student_label = behavior_label(student_row)
            support = _paired_outcome_support(teacher_row) | _paired_outcome_support(student_row) | {teacher_label, student_label}
            grouped_pairs[(benchmark, axis_id, condition)].append((teacher_label, student_label, support))

        for (benchmark, axis_id, condition), label_pairs in sorted(grouped_pairs.items()):
            support = sorted({label for _, _, labels in label_pairs for label in labels if label})
            for outcome in support:
                paired_n = len(label_pairs)
                both_positive = sum(t == outcome and s == outcome for t, s, _ in label_pairs)
                both_negative = sum(t != outcome and s != outcome for t, s, _ in label_pairs)
                teacher_only = sum(t == outcome and s != outcome for t, s, _ in label_pairs)
                student_only = sum(t != outcome and s == outcome for t, s, _ in label_pairs)
                teacher_successes = both_positive + teacher_only
                student_successes = both_positive + student_only
                delta = (student_successes - teacher_successes) / paired_n if paired_n else 0.0
                ci_low, ci_high = _paired_delta_ci(
                    both_positive=both_positive,
                    both_negative=both_negative,
                    teacher_only=teacher_only,
                    student_only=student_only,
                    confidence=confidence,
                )
                if delta > 0:
                    direction = "student_higher"
                elif delta < 0:
                    direction = "teacher_higher"
                else:
                    direction = "no_change"
                out.append(
                    {
                        "comparison_id": pair["comparison_id"],
                        "family": pair["family"],
                        "benchmark": benchmark,
                        "axis_id": axis_id,
                        "condition_type": condition,
                        "outcome_label": outcome,
                        "paired_n": paired_n,
                        "teacher_successes": teacher_successes,
                        "student_successes": student_successes,
                        "teacher_rate": teacher_successes / paired_n if paired_n else 0.0,
                        "student_rate": student_successes / paired_n if paired_n else 0.0,
                        "delta_student_minus_teacher": delta,
                        "delta_ci_low": ci_low,
                        "delta_ci_high": ci_high,
                        "both_positive": both_positive,
                        "both_negative": both_negative,
                        "teacher_only": teacher_only,
                        "student_only": student_only,
                        "agreement_rate": (both_positive + both_negative) / paired_n if paired_n else 0.0,
                        "discordant_n": teacher_only + student_only,
                        "exact_sign_p_value": _two_sided_sign_p(student_only, teacher_only),
                        "effect_direction": direction,
                    }
                )
    _annotate_multiple_testing(out, p_field="exact_sign_p_value")
    return out


def _annotate_multiple_testing(rows: list[dict[str, Any]], *, p_field: str) -> None:
    m = len(rows)
    if m == 0:
        return
    indexed = []
    for idx, row in enumerate(rows):
        p_value = float(row.get(p_field, 1.0))
        if math.isnan(p_value):
            p_value = 1.0
        p_value = min(1.0, max(0.0, p_value))
        row[p_field] = p_value
        row["p_value_bonferroni"] = min(1.0, p_value * m)
        row["significant_nominal_0_05"] = "true" if p_value < 0.05 else "false"
        indexed.append((idx, p_value))

    adjusted = [1.0] * m
    running_min = 1.0
    for rank_from_zero, (idx, p_value) in reversed(list(enumerate(sorted(indexed, key=lambda item: item[1])))):
        rank = rank_from_zero + 1
        running_min = min(running_min, p_value * m / rank)
        adjusted[idx] = min(1.0, running_min)
    for idx, q_value in enumerate(adjusted):
        rows[idx]["p_value_bh_fdr"] = q_value
        rows[idx]["significant_bonferroni_0_05"] = "true" if rows[idx]["p_value_bonferroni"] < 0.05 else "false"
        rows[idx]["significant_bh_fdr_0_05"] = "true" if q_value < 0.05 else "false"


def _paired_outcome_support(row: dict[str, Any]) -> set[str]:
    support = set()
    candidates = row.get("candidate_behaviors", [])
    if isinstance(candidates, list):
        support.update(str(item).strip() for item in candidates if str(item).strip())
    target = str(row.get("target_behavior", "")).strip()
    if target:
        support.add(target)
    label = behavior_label(row)
    if label:
        support.add(label)
    return support


def _paired_delta_ci(
    *,
    both_positive: int,
    both_negative: int,
    teacher_only: int,
    student_only: int,
    confidence: float,
) -> tuple[float, float]:
    n = both_positive + both_negative + teacher_only + student_only
    if n <= 1:
        delta = (student_only - teacher_only) / n if n else 0.0
        return delta, delta
    delta = (student_only - teacher_only) / n
    zero_count = both_positive + both_negative
    variance = (
        student_only * (1.0 - delta) ** 2
        + teacher_only * (-1.0 - delta) ** 2
        + zero_count * (0.0 - delta) ** 2
    ) / (n - 1)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    margin = z * math.sqrt(variance / n)
    return max(-1.0, delta - margin), min(1.0, delta + margin)


def _two_sided_sign_p(student_only: int, teacher_only: int) -> float:
    discordant = student_only + teacher_only
    if discordant == 0:
        return 1.0
    lower_tail_k = min(student_only, teacher_only)
    if discordant <= 2000:
        log_half = math.log(0.5)
        log_terms = [
            math.lgamma(discordant + 1) - math.lgamma(k + 1) - math.lgamma(discordant - k + 1) + discordant * log_half
            for k in range(lower_tail_k + 1)
        ]
        max_log = max(log_terms)
        cdf = math.exp(max_log) * sum(math.exp(term - max_log) for term in log_terms)
    else:
        mean = discordant / 2.0
        sd = math.sqrt(discordant / 4.0)
        z = (lower_tail_k + 0.5 - mean) / sd
        cdf = _normal_cdf(z)
    return min(1.0, max(0.0, 2.0 * cdf))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


SHARPENING_FIELDS = [
    "comparison_id",
    "family",
    "benchmark",
    "axis_id",
    "condition_type",
    "teacher_n",
    "student_n",
    "support_size",
    "alpha",
    "kl_student_to_pred",
    "tv_student_to_pred",
    "js_student_to_pred",
    "hellinger_student_to_pred",
]


def sharpening_fits(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model_scope: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    fit_benchmarks = set(config["analysis"]["fit_sharpening_on"])
    grouped = _labels_by_model_axis(rows)
    out: list[dict[str, Any]] = []
    alphas_by_pair: dict[str, list[float]] = defaultdict(list)
    for pair in selected_pairs(config, model_scope=model_scope):
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        keys = sorted({key[1:] for key in grouped if key[0] in {teacher, student}})
        for key in keys:
            benchmark, axis_id, condition = key
            if benchmark not in fit_benchmarks:
                continue
            if condition not in {"broad", "benchmark_native"}:
                continue
            t_labels = grouped.get((teacher, *key), [])
            s_labels = grouped.get((student, *key), [])
            support = sorted(set(t_labels) | set(s_labels))
            if len(support) < 2 or len(t_labels) < 4 or len(s_labels) < 4:
                continue
            p_t = probability_distribution(t_labels, support)
            p_s = probability_distribution(s_labels, support)
            fit = fit_alpha(p_t, p_s)
            alphas_by_pair[pair["comparison_id"]].append(fit["alpha"])
            out.append(
                {
                    "comparison_id": pair["comparison_id"],
                    "family": pair["family"],
                    "benchmark": benchmark,
                    "axis_id": axis_id,
                    "condition_type": condition,
                    "teacher_n": len(t_labels),
                    "student_n": len(s_labels),
                    "support_size": len(support),
                    **fit,
                }
            )
    alpha_by_pair = {
        comparison_id: _median(values)
        for comparison_id, values in sorted(alphas_by_pair.items())
        if values
    }
    return out, alpha_by_pair


SHARPENING_TRANSFER_FIELDS = [
    "comparison_id",
    "family",
    "benchmark",
    "axis_id",
    "condition_type",
    "alpha_from_grade_dimcim",
    "teacher_n",
    "student_n",
    "support_size",
    "tv_student_to_pred",
    "js_student_to_pred",
    "hellinger_student_to_pred",
]


def sharpening_transfer(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    alpha_by_pair: dict[str, float],
    *,
    model_scope: str,
) -> list[dict[str, Any]]:
    transfer_benchmarks = set(config["analysis"]["transfer_alpha_to"])
    grouped = _labels_by_model_axis(rows)
    out: list[dict[str, Any]] = []
    for pair in selected_pairs(config, model_scope=model_scope):
        comparison_id = pair["comparison_id"]
        if comparison_id not in alpha_by_pair:
            continue
        alpha = alpha_by_pair[comparison_id]
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        keys = sorted({key[1:] for key in grouped if key[0] in {teacher, student}})
        for key in keys:
            benchmark, axis_id, condition = key
            if benchmark not in transfer_benchmarks:
                continue
            t_labels = grouped.get((teacher, *key), [])
            s_labels = grouped.get((student, *key), [])
            support = sorted(set(t_labels) | set(s_labels))
            if len(support) < 2 or not t_labels or not s_labels:
                continue
            p_t = probability_distribution(t_labels, support)
            p_s = probability_distribution(s_labels, support)
            pred = sharpen_distribution(p_t, alpha)
            out.append(
                {
                    "comparison_id": comparison_id,
                    "family": pair["family"],
                    "benchmark": benchmark,
                    "axis_id": axis_id,
                    "condition_type": condition,
                    "alpha_from_grade_dimcim": alpha,
                    "teacher_n": len(t_labels),
                    "student_n": len(s_labels),
                    "support_size": len(support),
                    "tv_student_to_pred": total_variation(p_s, pred),
                    "js_student_to_pred": js_divergence(p_s, pred),
                    "hellinger_student_to_pred": hellinger(p_s, pred),
                }
            )
    return out


RESIDUAL_FIELDS = [
    "comparison_id",
    "family",
    "benchmark",
    "axis_id",
    "condition_type",
    "teacher_label",
    "student_label",
    "transition_count",
    "teacher_label_count",
    "m_ij",
    "student_marginal",
    "residual",
    "residual_se",
    "residual_ci_low",
    "residual_ci_high",
    "residual_z",
    "residual_p_value",
    "p_value_bonferroni",
    "p_value_bh_fdr",
    "significant_nominal_0_05",
    "significant_bonferroni_0_05",
    "significant_bh_fdr_0_05",
    "teacher_label_wilson_low",
    "teacher_label_wilson_high",
]


def residual_transport(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model_scope: str,
) -> list[dict[str, Any]]:
    by_match: dict[tuple[str, str, str, str, str, str], dict[str, str]] = defaultdict(dict)
    for row in rows:
        label = behavior_label(row)
        if not label:
            continue
        key = (
            row["benchmark"],
            row["axis_id"],
            row["condition_type"],
            row["image_prompt_uid"],
            str(row["seed"]),
            row.get("eval_uid", ""),
        )
        by_match[key][row["model_id"]] = label

    out: list[dict[str, Any]] = []
    for pair in selected_pairs(config, model_scope=model_scope):
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        grouped_pairs: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
        for key, labels_by_model in by_match.items():
            if teacher in labels_by_model and student in labels_by_model:
                benchmark, axis_id, condition, _, _, _ = key
                grouped_pairs[(benchmark, axis_id, condition)].append((labels_by_model[teacher], labels_by_model[student]))
        for (benchmark, axis_id, condition), transitions in sorted(grouped_pairs.items()):
            transition_counts = Counter(transitions)
            teacher_counts = Counter(t for t, _ in transitions)
            student_counts = Counter(s for _, s in transitions)
            total = len(transitions)
            student_marginal = {label: count / total for label, count in student_counts.items()}
            confidence = float(config.get("analysis", {}).get("bootstrap", {}).get("confidence", 0.95))
            z_crit = NormalDist().inv_cdf(0.5 + confidence / 2.0)
            for (teacher_label, student_label), count in sorted(transition_counts.items()):
                denom = teacher_counts[teacher_label]
                m_ij = count / denom
                marginal = student_marginal.get(student_label, 0.0)
                residual = m_ij - marginal
                residual_se = math.sqrt(max(0.0, (m_ij * (1.0 - m_ij) / denom) + (marginal * (1.0 - marginal) / total)))
                residual_z = residual / residual_se if residual_se > 0 else 0.0
                residual_p = 2.0 * (1.0 - _normal_cdf(abs(residual_z))) if residual_se > 0 else 1.0
                lo, hi = wilson_interval(count, denom)
                out.append(
                    {
                        "comparison_id": pair["comparison_id"],
                        "family": pair["family"],
                        "benchmark": benchmark,
                        "axis_id": axis_id,
                        "condition_type": condition,
                        "teacher_label": teacher_label,
                        "student_label": student_label,
                        "transition_count": count,
                        "teacher_label_count": denom,
                        "m_ij": m_ij,
                        "student_marginal": marginal,
                        "residual": residual,
                        "residual_se": residual_se,
                        "residual_ci_low": max(-1.0, residual - z_crit * residual_se),
                        "residual_ci_high": min(1.0, residual + z_crit * residual_se),
                        "residual_z": residual_z,
                        "residual_p_value": residual_p,
                        "teacher_label_wilson_low": lo,
                        "teacher_label_wilson_high": hi,
                    }
                )
    _annotate_multiple_testing(out, p_field="residual_p_value")
    return out


DISSOCIATION_FIELDS = [
    "comparison_id",
    "family",
    "benchmark",
    "axis_id",
    "target_behavior",
    "teacher_broad_rate",
    "student_broad_rate",
    "teacher_explicit_rate",
    "student_explicit_rate",
    "broad_drop",
    "explicit_drop",
    "teacher_broad_n",
    "student_broad_n",
    "teacher_explicit_n",
    "student_explicit_n",
    "classification",
]


def preference_capability_dissociation(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model_scope: str,
) -> list[dict[str, Any]]:
    thresholds = config["analysis"]["dissociation_thresholds"]
    broad_conditions = {"broad", "benign_overrefusal", "social_bias_default"}
    explicit_conditions = {
        "explicit",
        "knowledge_injected",
        "benchmark_native",
        "implicit_knowledge",
        "compositional_native",
    }

    by_model_axis_condition: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    targets_by_axis: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_model_axis_condition[(row["model_id"], row["benchmark"], row["axis_id"], row["condition_type"])].append(row)
        target = str(row.get("target_behavior", "")).strip()
        if target:
            targets_by_axis[(row["benchmark"], row["axis_id"])].add(target)

    out: list[dict[str, Any]] = []
    for pair in selected_pairs(config, model_scope=model_scope):
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        for (benchmark, axis_id), targets in sorted(targets_by_axis.items()):
            for target in sorted(targets):
                teacher_broad = _rows_for_conditions(by_model_axis_condition, teacher, benchmark, axis_id, broad_conditions)
                student_broad = _rows_for_conditions(by_model_axis_condition, student, benchmark, axis_id, broad_conditions)
                teacher_explicit = _target_rows(_rows_for_conditions(by_model_axis_condition, teacher, benchmark, axis_id, explicit_conditions), target)
                student_explicit = _target_rows(_rows_for_conditions(by_model_axis_condition, student, benchmark, axis_id, explicit_conditions), target)
                if not teacher_broad or not student_broad or not teacher_explicit or not student_explicit:
                    continue
                p_t = _rate_label_equals(teacher_broad, target)
                p_s = _rate_label_equals(student_broad, target)
                q_t = _success_rate(teacher_explicit, target)
                q_s = _success_rate(student_explicit, target)
                broad_drop = p_t - p_s
                explicit_drop = q_t - q_s
                classification = "stable_or_ambiguous"
                if broad_drop >= thresholds["broad_drop_min"]:
                    if explicit_drop <= thresholds["explicit_recovery_tolerance"]:
                        classification = "preference_shift"
                    elif explicit_drop >= thresholds["explicit_loss_min"]:
                        classification = "capability_or_information_loss"
                out.append(
                    {
                        "comparison_id": pair["comparison_id"],
                        "family": pair["family"],
                        "benchmark": benchmark,
                        "axis_id": axis_id,
                        "target_behavior": target,
                        "teacher_broad_rate": p_t,
                        "student_broad_rate": p_s,
                        "teacher_explicit_rate": q_t,
                        "student_explicit_rate": q_s,
                        "broad_drop": broad_drop,
                        "explicit_drop": explicit_drop,
                        "teacher_broad_n": len(teacher_broad),
                        "student_broad_n": len(student_broad),
                        "teacher_explicit_n": len(teacher_explicit),
                        "student_explicit_n": len(student_explicit),
                        "classification": classification,
                    }
                )
    return out


SURVIVAL_FIELDS = [
    "comparison_id",
    "family",
    "benchmark",
    "axis_id",
    "condition_type",
    "complexity_level",
    "behavior",
    "teacher_frequency",
    "student_frequency",
    "retention_ratio",
    "teacher_n",
    "student_n",
]


def survival_features(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    model_scope: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped = _labels_by_model_axis(rows)
    complexity_by_axis = _complexity_by_axis(rows)
    out: list[dict[str, Any]] = []
    for pair in selected_pairs(config, model_scope=model_scope):
        teacher = pair["teacher"]["model_id"]
        student = pair["student"]["model_id"]
        keys = sorted({key[1:] for key in grouped if key[0] in {teacher, student}})
        for key in keys:
            t_labels = grouped.get((teacher, *key), [])
            s_labels = grouped.get((student, *key), [])
            if not t_labels or not s_labels:
                continue
            support = sorted(set(t_labels) | set(s_labels))
            p_t = probability_distribution(t_labels, support)
            p_s = probability_distribution(s_labels, support)
            benchmark, axis_id, condition = key
            complexity = complexity_by_axis.get((benchmark, axis_id, condition), "unknown")
            for behavior in support:
                teacher_frequency = p_t.get(behavior, 0.0)
                student_frequency = p_s.get(behavior, 0.0)
                ratio = student_frequency / teacher_frequency if teacher_frequency > 0 else math.nan
                out.append(
                    {
                        "comparison_id": pair["comparison_id"],
                        "family": pair["family"],
                        "benchmark": benchmark,
                        "axis_id": axis_id,
                        "condition_type": condition,
                        "complexity_level": complexity,
                        "behavior": behavior,
                        "teacher_frequency": teacher_frequency,
                        "student_frequency": student_frequency,
                        "retention_ratio": ratio,
                        "teacher_n": len(t_labels),
                        "student_n": len(s_labels),
                    }
                )
    summary = _survival_summary(out)
    return out, summary



SURVIVAL_REGRESSION_FIELDS = [
    "comparison_id",
    "family",
    "formula",
    "status",
    "term",
    "coefficient",
    "std_error",
    "t_stat",
    "p_value",
    "n",
    "df",
    "r_squared",
    "reference_benchmark",
    "interpretation",
]


def survival_regression(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        teacher_frequency = _float_or_none(row.get("teacher_frequency"))
        student_frequency = _float_or_none(row.get("student_frequency"))
        if teacher_frequency is None or student_frequency is None or teacher_frequency <= 0:
            continue
        retention_ratio = _float_or_none(row.get("retention_ratio"))
        if retention_ratio is None or math.isnan(retention_ratio):
            continue
        by_pair[(str(row.get("comparison_id", "")), str(row.get("family", "")))].append(row)

    coefficient_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for (comparison_id, family), items in sorted(by_pair.items()):
        benchmarks = sorted({str(row.get("benchmark", "")) for row in items})
        reference = benchmarks[0] if benchmarks else ""
        terms = ["intercept", "teacher_frequency", "complexity_rank"] + [f"benchmark:{name}" for name in benchmarks[1:]]
        formula = "log_retention_ratio ~ teacher_frequency + complexity_rank + C(benchmark)"
        fit = _fit_survival_wls(items, terms)
        while fit is None and any(term.startswith("benchmark:") for term in terms):
            terms = [term for term in terms if not term.startswith("benchmark:") or term != terms[-1]]
            formula = "log_retention_ratio ~ teacher_frequency + complexity_rank" if not any(term.startswith("benchmark:") for term in terms) else formula
            fit = _fit_survival_wls(items, terms)
        if fit is None:
            summary[comparison_id] = {
                "status": "insufficient_variation",
                "n": len(items),
                "formula": formula,
                "reference_benchmark": reference,
                "complexity_rank_coefficient": None,
                "complexity_rank_p_value": None,
                "frequency_controlled_hierarchy_supported": False,
            }
            coefficient_rows.append(
                {
                    "comparison_id": comparison_id,
                    "family": family,
                    "formula": formula,
                    "status": "insufficient_variation",
                    "term": "model",
                    "coefficient": "",
                    "std_error": "",
                    "t_stat": "",
                    "p_value": "",
                    "n": len(items),
                    "df": 0,
                    "r_squared": "",
                    "reference_benchmark": reference,
                    "interpretation": "Not enough independent variation to estimate frequency-controlled survival regression.",
                }
            )
            continue
        term_rows = []
        for idx, term in enumerate(fit["terms"]):
            interpretation = _survival_term_interpretation(term, fit["coefficients"][idx], fit["p_values"][idx])
            row = {
                "comparison_id": comparison_id,
                "family": family,
                "formula": formula,
                "status": "estimated",
                "term": term,
                "coefficient": fit["coefficients"][idx],
                "std_error": fit["std_errors"][idx],
                "t_stat": fit["t_stats"][idx],
                "p_value": fit["p_values"][idx],
                "n": fit["n"],
                "df": fit["df"],
                "r_squared": fit["r_squared"],
                "reference_benchmark": reference,
                "interpretation": interpretation,
            }
            coefficient_rows.append(row)
            term_rows.append(row)
        complexity = next((row for row in term_rows if row["term"] == "complexity_rank"), None)
        teacher_freq = next((row for row in term_rows if row["term"] == "teacher_frequency"), None)
        supported = bool(complexity and float(complexity["coefficient"]) < 0 and float(complexity["p_value"]) < 0.05)
        summary[comparison_id] = {
            "status": "estimated",
            "n": fit["n"],
            "df": fit["df"],
            "formula": formula,
            "reference_benchmark": reference,
            "r_squared": fit["r_squared"],
            "complexity_rank_coefficient": complexity["coefficient"] if complexity else None,
            "complexity_rank_p_value": complexity["p_value"] if complexity else None,
            "teacher_frequency_coefficient": teacher_freq["coefficient"] if teacher_freq else None,
            "teacher_frequency_p_value": teacher_freq["p_value"] if teacher_freq else None,
            "frequency_controlled_hierarchy_supported": supported,
        }
    return coefficient_rows, summary


def _fit_survival_wls(items: list[dict[str, Any]], terms: list[str]) -> dict[str, Any] | None:
    x_rows: list[list[float]] = []
    y_values: list[float] = []
    weights: list[float] = []
    for row in items:
        teacher_frequency = _float_or_none(row.get("teacher_frequency"))
        student_frequency = _float_or_none(row.get("student_frequency"))
        if teacher_frequency is None or student_frequency is None or teacher_frequency <= 0:
            continue
        x_rows.append([_survival_feature_value(row, term) for term in terms])
        epsilon = 1e-12
        y_values.append(math.log(max(student_frequency, epsilon) / max(teacher_frequency, epsilon)))
        teacher_n = _float_or_none(row.get("teacher_n")) or 1.0
        student_n = _float_or_none(row.get("student_n")) or 1.0
        weights.append(max(1.0, min(teacher_n, student_n) * teacher_frequency))
    n = len(y_values)
    p = len(terms)
    if n <= p or p == 0:
        return None
    xtwx = [[0.0 for _ in range(p)] for _ in range(p)]
    xtwy = [0.0 for _ in range(p)]
    for x, y, weight in zip(x_rows, y_values, weights):
        for j in range(p):
            xtwy[j] += weight * x[j] * y
            for k in range(p):
                xtwx[j][k] += weight * x[j] * x[k]
    inverse = _invert_matrix(xtwx)
    if inverse is None:
        return None
    coefficients = [_dot(inverse_row, xtwy) for inverse_row in inverse]
    y_hat = [_dot(x, coefficients) for x in x_rows]
    weight_total = sum(weights)
    y_mean = sum(weight * y for weight, y in zip(weights, y_values)) / weight_total
    sse = sum(weight * (y - pred) ** 2 for y, pred, weight in zip(y_values, y_hat, weights))
    sst = sum(weight * (y - y_mean) ** 2 for y, weight in zip(y_values, weights))
    df = n - p
    if df <= 0:
        return None
    sigma2 = sse / df
    std_errors = [math.sqrt(max(0.0, sigma2 * inverse[idx][idx])) for idx in range(p)]
    t_stats = [coef / se if se > 0 else 0.0 for coef, se in zip(coefficients, std_errors)]
    p_values = [2.0 * (1.0 - NormalDist().cdf(abs(t_stat))) for t_stat in t_stats]
    return {
        "terms": terms,
        "coefficients": coefficients,
        "std_errors": std_errors,
        "t_stats": t_stats,
        "p_values": p_values,
        "n": n,
        "df": df,
        "r_squared": 1.0 - sse / sst if sst > 0 else 0.0,
    }


def _survival_feature_value(row: dict[str, Any], term: str) -> float:
    if term == "intercept":
        return 1.0
    if term == "teacher_frequency":
        return float(_float_or_none(row.get("teacher_frequency")) or 0.0)
    if term == "complexity_rank":
        return float(_complexity_rank(str(row.get("complexity_level", "unknown"))))
    if term.startswith("benchmark:"):
        return 1.0 if str(row.get("benchmark", "")) == term.split(":", 1)[1] else 0.0
    raise ValueError(f"unknown survival regression term: {term}")


def _complexity_rank(level: str) -> int:
    rank = {
        "atomic": 0,
        "relational": 1,
        "safety_benign": 1,
        "bias": 1,
        "compositional": 2,
        "safety_unsafe": 2,
        "memorization": 2,
        "implicit": 3,
        "knowledge": 3,
        "unlearning": 3,
        "unknown": 3,
    }
    return rank.get(level, 3)


def _survival_term_interpretation(term: str, coefficient: float, p_value: float) -> str:
    significance = "nominally significant" if p_value < 0.05 else "not nominally significant"
    if term == "complexity_rank":
        direction = "lower retention at higher complexity" if coefficient < 0 else "higher retention at higher complexity"
        return f"{direction}; {significance} after teacher-frequency and benchmark controls."
    if term == "teacher_frequency":
        direction = "more common teacher behaviors retain more strongly" if coefficient > 0 else "more common teacher behaviors retain less strongly"
        return f"{direction}; {significance} after complexity and benchmark controls."
    if term.startswith("benchmark:"):
        return f"benchmark fixed-effect relative to the reference benchmark; {significance}."
    return f"model intercept; {significance}."


def _invert_matrix(matrix: list[list[float]]) -> list[list[float]] | None:
    n = len(matrix)
    augmented = [[float(matrix[i][j]) for j in range(n)] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-10:
            return None
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            augmented[row] = [value - factor * pivot_value for value, pivot_value in zip(augmented[row], augmented[col])]
    return [row[n:] for row in augmented]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _labels_by_model_axis(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[str]]:
    grouped: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for row in rows:
        label = behavior_label(row)
        if label:
            grouped[(row["model_id"], row["benchmark"], row["axis_id"], row["condition_type"])].append(label)
    return grouped


def _complexity_by_axis(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[(row["benchmark"], row["axis_id"], row["condition_type"])][str(row.get("complexity_level", "unknown"))] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in counts.items()}


def _rows_for_conditions(
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    model_id: str,
    benchmark: str,
    axis_id: str,
    conditions: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        rows.extend(grouped.get((model_id, benchmark, axis_id, condition), []))
    return rows


def _target_rows(rows: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    exact = [row for row in rows if str(row.get("target_behavior", "")).strip() == target]
    return exact or rows


def _rate_label_equals(rows: list[dict[str, Any]], target: str) -> float:
    labels = [behavior_label(row) for row in rows if behavior_label(row)]
    if not labels:
        return 0.0
    return sum(label == target for label in labels) / len(labels)


def _success_rate(rows: list[dict[str, Any]], target: str) -> float:
    values = [success_for_target(row, target) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    return sum(bool(value) for value in values) / len(values)


def _survival_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not math.isnan(float(row["retention_ratio"])):
            by_pair[row["comparison_id"]].append(row)
    summary: dict[str, Any] = {}
    complexity_rank = {
        "atomic": 0,
        "relational": 1,
        "compositional": 2,
        "implicit": 3,
        "knowledge": 3,
        "safety_benign": 1,
        "safety_unsafe": 2,
        "bias": 1,
        "memorization": 2,
        "unlearning": 3,
        "unknown": 3,
    }
    for comparison_id, items in by_pair.items():
        teacher_frequency = [float(row["teacher_frequency"]) for row in items]
        retention = [float(row["retention_ratio"]) for row in items]
        complexity = [complexity_rank.get(str(row["complexity_level"]), 3) for row in items]
        means: dict[str, float] = {}
        for level in sorted({str(row["complexity_level"]) for row in items}):
            vals = [float(row["retention_ratio"]) for row in items if row["complexity_level"] == level]
            if vals:
                means[level] = sum(vals) / len(vals)
        summary[comparison_id] = {
            "n_behavior_cells": len(items),
            "spearman_teacher_frequency_vs_retention": spearman(teacher_frequency, retention),
            "spearman_complexity_vs_retention": spearman(complexity, retention),
            "mean_retention_by_complexity": means,
        }
    return summary


def _bootstrap_pairwise_metric_cis(
    t_labels: list[str],
    s_labels: list[str],
    support: list[str],
    config: dict[str, Any],
    *,
    salt: str,
) -> dict[str, float]:
    boot = config.get("analysis", {}).get("bootstrap", {})
    iterations = int(boot.get("iterations", 0))
    confidence = float(boot.get("confidence", 0.95))
    if iterations <= 1 or not t_labels or not s_labels:
        return {
            "total_variation_ci_low": 0.0,
            "total_variation_ci_high": 0.0,
            "js_divergence_ci_low": 0.0,
            "js_divergence_ci_high": 0.0,
            "hellinger_ci_low": 0.0,
            "hellinger_ci_high": 0.0,
            "entropy_delta_ci_low": 0.0,
            "entropy_delta_ci_high": 0.0,
        }
    seed_base = int(stable_hash(boot.get("seed", 0), salt, length=8), 16)
    rng = random.Random(seed_base)
    tv_values: list[float] = []
    js_values: list[float] = []
    hell_values: list[float] = []
    entropy_delta_values: list[float] = []
    for _ in range(iterations):
        t_sample = [t_labels[rng.randrange(len(t_labels))] for _ in range(len(t_labels))]
        s_sample = [s_labels[rng.randrange(len(s_labels))] for _ in range(len(s_labels))]
        p_t = probability_distribution(t_sample, support)
        p_s = probability_distribution(s_sample, support)
        tv_values.append(total_variation(p_t, p_s))
        js_values.append(js_divergence(p_t, p_s))
        hell_values.append(hellinger(p_t, p_s))
        entropy_delta_values.append(normalized_entropy(p_s) - normalized_entropy(p_t))
    lo_q = (1.0 - confidence) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "total_variation_ci_low": _quantile(tv_values, lo_q),
        "total_variation_ci_high": _quantile(tv_values, hi_q),
        "js_divergence_ci_low": _quantile(js_values, lo_q),
        "js_divergence_ci_high": _quantile(js_values, hi_q),
        "hellinger_ci_low": _quantile(hell_values, lo_q),
        "hellinger_ci_high": _quantile(hell_values, hi_q),
        "entropy_delta_ci_low": _quantile(entropy_delta_values, lo_q),
        "entropy_delta_ci_high": _quantile(entropy_delta_values, hi_q),
    }


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


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
