from __future__ import annotations

import csv
from pathlib import Path

from t2i_distill.analysis import behavior_label, run_full_analysis, survival_regression
from t2i_distill.config import load_config


FIELDS = [
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


def test_analysis_writes_core_rq_outputs(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    rows = []
    for seed in range(8):
        teacher_label = "red" if seed < 6 else "blue"
        student_label = "red" if seed < 7 else "blue"
        rows.extend(
            [
                _row("sdxl_base", "grade", "grade/object/color", "broad", "prompt-a", seed, teacher_label),
                _row("sdxl_lightning_4step", "grade", "grade/object/color", "broad", "prompt-a", seed, student_label),
            ]
        )
    for seed in range(4):
        rows.extend(
            [
                _row("sdxl_base", "dimcim", "dimcim/chair/material", "broad", f"broad-{seed}", seed, "wood"),
                _row("sdxl_lightning_4step", "dimcim", "dimcim/chair/material", "broad", f"broad-{seed}", seed, "plastic"),
                _row("sdxl_base", "dimcim", "dimcim/chair/material", "explicit", f"explicit-{seed}", seed, "wood", "wood", "1.0"),
                _row("sdxl_lightning_4step", "dimcim", "dimcim/chair/material", "explicit", f"explicit-{seed}", seed, "wood", "wood", "1.0"),
            ]
        )
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    out = tmp_path / "analysis"
    summary = run_full_analysis(labels, load_config(Path("config/experiment.json")), out)
    assert summary["usable_label_rows"] == len(rows)
    assert (out / "distribution_summary.csv").exists()
    assert (out / "paired_outcome_tests.csv").exists()
    assert (out / "residual_transport.csv").exists()
    assert (out / "survival_regression.csv").exists()
    assert (out / "quality_by_model_benchmark.csv").exists()
    assert (out / "quality_pair_balance.csv").exists()
    assert summary["quality_control"]["kept_rows"] == len(rows)
    with (out / "pairwise_shift.csv").open(encoding="utf-8") as handle:
        pairwise_header = handle.readline()
    with (out / "paired_outcome_tests.csv").open(encoding="utf-8") as handle:
        paired_header = handle.readline()
        paired_body = handle.read()
    with (out / "survival_regression.csv").open(encoding="utf-8") as handle:
        survival_regression_header = handle.readline()
    with (out / "residual_transport.csv").open(encoding="utf-8") as handle:
        residual_header = handle.readline()
    assert "total_variation_ci_low" in pairwise_header
    assert "exact_sign_p_value" in paired_header
    assert "p_value_bh_fdr" in paired_header
    assert "significant_bonferroni_0_05" in paired_header
    assert "student_only" in paired_header
    assert "red" in paired_body
    assert "complexity_rank" in survival_regression_header or "term" in survival_regression_header
    assert "residual_p_value" in residual_header
    assert "p_value_bh_fdr" in residual_header
    assert summary["paired_outcome_tests"]["rows"] > 0
    assert "bh_fdr_significant_residual_0_05" in summary["rq2_structured_remapping"]
    assert summary["rq3_what_is_lost"]["preference_shift"] >= 1




def test_quality_controls_can_retain_low_quality_refusal_rows(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    rows = [
        {
            **_row("sdxl_base", "overt", "overt/safety", "unsafe_safety", "prompt-a", 0, ""),
            "quality_score": "1",
            "refusal": "true",
            "unsafe": "false",
        },
        {
            **_row("sdxl_lightning_4step", "overt", "overt/safety", "unsafe_safety", "prompt-a", 0, ""),
            "quality_score": "1",
            "refusal": "true",
            "unsafe": "false",
        },
    ]
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    config = load_config(Path("config/experiment.json"))
    config["analysis"]["quality_controls"]["retain_refusal_rows"] = True

    summary = run_full_analysis(labels, config, tmp_path / "analysis")

    assert summary["usable_label_rows"] == 2
    assert summary["quality_control"]["low_quality_rows"] == 2
    assert summary["quality_control"]["kept_rows"] == 2


def test_survival_regression_controls_frequency_and_complexity() -> None:
    rows = []
    for idx, (complexity, rank, retention) in enumerate(
        [
            ("atomic", 0, 0.98),
            ("atomic", 0, 0.95),
            ("relational", 1, 0.82),
            ("relational", 1, 0.78),
            ("compositional", 2, 0.58),
            ("compositional", 2, 0.54),
            ("implicit", 3, 0.35),
            ("implicit", 3, 0.30),
        ]
    ):
        teacher_frequency = 0.2 + 0.03 * (idx % 2)
        student_frequency = teacher_frequency * retention
        rows.append(
            {
                "comparison_id": "teacher__student",
                "family": "unit",
                "benchmark": "grade" if idx % 2 == 0 else "dimcim",
                "axis_id": f"axis-{idx}",
                "condition_type": "broad",
                "complexity_level": complexity,
                "behavior": f"behavior-{idx}",
                "teacher_frequency": teacher_frequency,
                "student_frequency": student_frequency,
                "retention_ratio": retention,
                "teacher_n": 100,
                "student_n": 100,
            }
        )

    coef_rows, summary = survival_regression(rows)

    complexity = next(row for row in coef_rows if row["term"] == "complexity_rank")
    assert complexity["coefficient"] < 0
    assert summary["teacher__student"]["status"] == "estimated"
    assert summary["teacher__student"]["complexity_rank_coefficient"] < 0


def _row(
    model_id: str,
    benchmark: str,
    axis_id: str,
    condition_type: str,
    prompt_uid: str,
    seed: int,
    semantic_label: str,
    target_behavior: str = "",
    pass_score: str = "",
) -> dict[str, str]:
    return {
        "label_uid": f"{model_id}-{benchmark}-{condition_type}-{prompt_uid}-{seed}",
        "job_id": f"job-{model_id}-{prompt_uid}-{seed}",
        "eval_uid": f"eval-{benchmark}-{condition_type}-{prompt_uid}",
        "model_id": model_id,
        "benchmark": benchmark,
        "axis_id": axis_id,
        "axis_name": axis_id,
        "condition_type": condition_type,
        "complexity_level": "atomic",
        "image_prompt_uid": prompt_uid,
        "seed": str(seed),
        "prompt_text": "test prompt",
        "image_path": "",
        "target_behavior": target_behavior,
        "candidate_behaviors": "[\"red\", \"blue\", \"wood\", \"plastic\"]",
        "semantic_label": semantic_label,
        "semantic_label_confidence": "1.0",
        "pass_score": pass_score,
        "quality_score": "4.0",
        "refusal": "",
        "unsafe": "",
        "memorization_score": "",
        "evaluator": "test",
        "notes": "",
    }



def test_behavior_label_prioritizes_safety_outcome() -> None:
    assert behavior_label({"refusal": "false", "unsafe": "true"}) == "unsafe"
    assert behavior_label({"refusal": "true", "unsafe": "true"}) == "refusal"
    assert behavior_label({"refusal": "false", "unsafe": "false"}) == "safe"
