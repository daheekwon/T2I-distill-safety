from __future__ import annotations

from pathlib import Path

from t2i_distill.analysis import (
    DISTRIBUTION_FIELDS,
    DISSOCIATION_FIELDS,
    PAIRED_OUTCOME_FIELDS,
    RESIDUAL_FIELDS,
    SHARPENING_FIELDS,
    SHARPENING_TRANSFER_FIELDS,
    SURVIVAL_FIELDS,
    SURVIVAL_REGRESSION_FIELDS,
)
from t2i_distill.io import write_csv, write_json
from t2i_distill.reporting import build_claim_report


def test_claim_report_marks_template_state_pending(tmp_path: Path) -> None:
    result_dir = tmp_path / "post"
    analysis = result_dir / "analysis"
    analysis.mkdir(parents=True)
    write_json(result_dir / "design_matrix.json", {"overall_pass": True})
    write_json(result_dir / "statistical_power.json", {"overall_pass": True})
    write_json(result_dir / "label_quality.json", {"overall_pass": False, "gates": {}})
    write_json(result_dir / "rq_readiness.json", {"overall_ready": False})
    write_json(result_dir / "experiment_state.json", {"gates": {"label_ready_for_analysis": {"pass": False, "detail": "missing labels"}}})
    write_json(analysis / "claim_evidence_matrix.json", {"usable_label_rows": 0})

    report = build_claim_report(result_dir=result_dir, config=_config(), output_json=result_dir / "claim_report.json", output_markdown=result_dir / "claim_report.md")

    assert report["overall_status"] == "pending_labels"
    assert report["rq_reports"]["rq1_generic_sharpening"]["status"] == "pending_labels"
    assert (result_dir / "claim_report.md").exists()


def test_claim_report_summarizes_interpretable_results(tmp_path: Path) -> None:
    result_dir = tmp_path / "post"
    analysis = result_dir / "analysis"
    analysis.mkdir(parents=True)
    write_json(result_dir / "design_matrix.json", {"overall_pass": True})
    write_json(result_dir / "statistical_power.json", {"overall_pass": True})
    write_json(result_dir / "label_quality.json", {"overall_pass": True, "gates": {}})
    write_json(result_dir / "rq_readiness.json", {"overall_ready": True})
    write_json(result_dir / "experiment_state.json", {"gates": {"label_ready_for_analysis": {"pass": True, "detail": "ready"}}})
    write_json(
        analysis / "claim_evidence_matrix.json",
        {
            "usable_label_rows": 100,
            "rq4_survival_predictors": {
                "teacher__student": {
                    "n_behavior_cells": 3,
                    "spearman_teacher_frequency_vs_retention": 0.4,
                    "spearman_complexity_vs_retention": -0.2,
                    "mean_retention_by_complexity": {"atomic": 0.9},
                }
            },
            "rq4_survival_regression": {
                "teacher__student": {
                    "status": "estimated",
                    "n": 20,
                    "df": 15,
                    "formula": "log_retention_ratio ~ teacher_frequency + complexity_rank + C(benchmark)",
                    "reference_benchmark": "grade",
                    "r_squared": 0.7,
                    "complexity_rank_coefficient": -0.3,
                    "complexity_rank_p_value": 0.01,
                    "teacher_frequency_coefficient": 0.2,
                    "teacher_frequency_p_value": 0.04,
                    "frequency_controlled_hierarchy_supported": True,
                }
            },
        },
    )
    write_csv(
        analysis / "sharpening_fits.csv",
        [{"comparison_id": "teacher__student", "family": "unit", "benchmark": "grade", "axis_id": "a", "condition_type": "broad", "teacher_n": 50, "student_n": 50, "support_size": 2, "alpha": 1.5, "kl_student_to_pred": 0.1, "tv_student_to_pred": 0.02, "js_student_to_pred": 0.01, "hellinger_student_to_pred": 0.03}],
        SHARPENING_FIELDS,
    )
    write_csv(
        analysis / "sharpening_transfer.csv",
        [{"comparison_id": "teacher__student", "family": "unit", "benchmark": "t2i_compbench", "axis_id": "b", "condition_type": "compositional_native", "alpha_from_grade_dimcim": 1.5, "teacher_n": 50, "student_n": 50, "support_size": 2, "tv_student_to_pred": 0.08, "js_student_to_pred": 0.02, "hellinger_student_to_pred": 0.04}],
        SHARPENING_TRANSFER_FIELDS,
    )
    write_csv(
        analysis / "residual_transport.csv",
        [{"comparison_id": "teacher__student", "family": "unit", "benchmark": "grade", "axis_id": "a", "condition_type": "broad", "teacher_label": "red", "student_label": "blue", "transition_count": 20, "teacher_label_count": 50, "m_ij": 0.4, "student_marginal": 0.1, "residual": 0.3, "teacher_label_wilson_low": 0.2, "teacher_label_wilson_high": 0.6}],
        RESIDUAL_FIELDS,
    )
    write_csv(
        analysis / "preference_capability.csv",
        [{"comparison_id": "teacher__student", "family": "unit", "benchmark": "dimcim", "axis_id": "color", "target_behavior": "red", "teacher_broad_rate": 0.5, "student_broad_rate": 0.2, "teacher_explicit_rate": 0.9, "student_explicit_rate": 0.88, "broad_drop": 0.3, "explicit_drop": 0.02, "teacher_broad_n": 50, "student_broad_n": 50, "teacher_explicit_n": 20, "student_explicit_n": 20, "classification": "preference_shift"}],
        DISSOCIATION_FIELDS,
    )
    write_csv(
        analysis / "survival_features.csv",
        [{"comparison_id": "teacher__student", "family": "unit", "benchmark": "grade", "axis_id": "a", "condition_type": "broad", "complexity_level": "atomic", "behavior": "red", "teacher_frequency": 0.5, "student_frequency": 0.45, "retention_ratio": 0.9, "teacher_n": 50, "student_n": 50}],
        SURVIVAL_FIELDS,
    )
    write_csv(
        analysis / "survival_regression.csv",
        [{"comparison_id": "teacher__student", "family": "unit", "formula": "log_retention_ratio ~ teacher_frequency + complexity_rank + C(benchmark)", "status": "estimated", "term": "complexity_rank", "coefficient": -0.3, "std_error": 0.1, "t_stat": -3.0, "p_value": 0.01, "n": 20, "df": 15, "r_squared": 0.7, "reference_benchmark": "grade", "interpretation": "lower retention at higher complexity; nominally significant after teacher-frequency and benchmark controls."}],
        SURVIVAL_REGRESSION_FIELDS,
    )
    write_csv(
        analysis / "distribution_summary.csv",
        [
            _dist("teacher", "overt", "unsafe_safety", "unsafe", 20, 100),
            _dist("teacher", "overt", "unsafe_safety", "safe", 80, 100),
            _dist("student", "overt", "unsafe_safety", "unsafe", 30, 100),
            _dist("student", "overt", "unsafe_safety", "safe", 70, 100),
            _dist("teacher", "membench", "memorization_trigger", "memorized", 10, 100),
            _dist("teacher", "membench", "memorization_trigger", "not_memorized", 90, 100),
            _dist("student", "membench", "memorization_trigger", "memorized", 20, 100),
            _dist("student", "membench", "memorization_trigger", "not_memorized", 80, 100),
        ],
        DISTRIBUTION_FIELDS,
    )
    write_csv(
        analysis / "paired_outcome_tests.csv",
        [
            _paired("overt", "unsafe_safety", "unsafe", 100, 5, 15, 0.10, 0.12),
            _paired("membench", "memorization_trigger", "memorized", 100, 4, 14, 0.10, 0.18),
        ],
        PAIRED_OUTCOME_FIELDS,
    )

    report = build_claim_report(result_dir=result_dir, config=_config(), output_json=result_dir / "claim_report.json")

    assert report["overall_status"] == "interpretable_results_available"
    assert report["rq_reports"]["rq1_generic_sharpening"]["status"] == "selective_deviation_from_simple_sharpening"
    assert report["rq_reports"]["rq2_structured_remapping"]["summary"]["large_abs_residual_rows"] == 1
    assert report["rq_reports"]["rq4_survival_principle"]["status"] == "frequency_controlled_hierarchy_supported"
    assert report["rq_reports"]["rq4_survival_principle"]["summary"]["pairs_supporting_frequency_controlled_hierarchy"] == 1
    assert report["safety_report"]["status"] == "safety_shift_detected"
    assert report["safety_report"]["summary"]["paired_test_rows"] == 1
    assert report["memorization_report"]["status"] == "memorization_shift_detected"
    assert report["memorization_report"]["summary"]["paired_test_rows"] == 1


def _dist(model_id: str, benchmark: str, condition: str, label: str, count: int, n: int) -> dict[str, object]:
    return {
        "model_id": model_id,
        "benchmark": benchmark,
        "axis_id": "axis",
        "condition_type": condition,
        "label": label,
        "count": count,
        "probability": count / n,
        "n": n,
        "normalized_entropy": 0.5,
    }


def _paired(benchmark: str, condition: str, outcome: str, n: int, teacher_only: int, student_only: int, delta: float, p_value: float) -> dict[str, object]:
    return {
        "comparison_id": "teacher__student",
        "family": "unit",
        "benchmark": benchmark,
        "axis_id": "axis",
        "condition_type": condition,
        "outcome_label": outcome,
        "paired_n": n,
        "teacher_successes": 20,
        "student_successes": 20 + student_only - teacher_only,
        "teacher_rate": 0.2,
        "student_rate": 0.2 + delta,
        "delta_student_minus_teacher": delta,
        "delta_ci_low": delta - 0.05,
        "delta_ci_high": delta + 0.05,
        "both_positive": 15,
        "both_negative": n - 15 - teacher_only - student_only,
        "teacher_only": teacher_only,
        "student_only": student_only,
        "agreement_rate": 0.9,
        "discordant_n": teacher_only + student_only,
        "exact_sign_p_value": p_value,
        "p_value_bonferroni": min(1.0, p_value * 2),
        "p_value_bh_fdr": min(1.0, p_value * 2),
        "significant_nominal_0_05": "true" if p_value < 0.05 else "false",
        "significant_bonferroni_0_05": "false",
        "significant_bh_fdr_0_05": "false",
        "effect_direction": "student_higher" if delta > 0 else "teacher_higher",
    }


def _config() -> dict[str, object]:
    return {
        "default_device": "cuda:2",
        "benchmark_plan": {},
        "model_pairs": [
            {
                "comparison_id": "teacher__student",
                "family": "unit",
                "priority": "primary",
                "teacher": {"model_id": "teacher"},
                "student": {"model_id": "student"},
            }
        ],
    }
