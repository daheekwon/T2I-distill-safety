from __future__ import annotations

from pathlib import Path

from t2i_distill.io import read_csv, write_csv
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS
from t2i_distill.safety_mem_analysis import run_safety_memorization_analysis


def test_focused_safety_memorization_analysis_outputs_transitions(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    full = tmp_path / "full.csv"
    rows = [
        _safety_row("ta", "teacher", "unsafe_safety", "p1", unsafe="true"),
        _safety_row("sa", "student", "unsafe_safety", "p1", unsafe="false"),
        _safety_row("tb", "teacher", "unsafe_safety", "p2", unsafe="false"),
        _safety_row("sb", "student", "unsafe_safety", "p2", unsafe="true"),
        _mem_row("tm", "teacher", "memorization_trigger", "m1", score="0.8"),
        _mem_row("sm", "student", "memorization_trigger", "m1", score="0.1"),
    ]
    write_csv(labels, rows, LABEL_TEMPLATE_FIELDS)
    write_csv(full, rows + [_mem_row("missing", "teacher", "memorization_trigger", "m2", score="")], LABEL_TEMPLATE_FIELDS)
    out = tmp_path / "focused"

    summary = run_safety_memorization_analysis(labels, _config(), out, full_labels_path=full)

    assert summary["safety_rows"] == 4
    assert summary["memorization_scored_rows"] == 2
    safety = read_csv(out / "safety_pair_transitions_by_condition.csv")[0]
    assert safety["paired_n"] == "2"
    assert safety["weakened_unsafe_to_safe"] == "1"
    assert safety["student_new_unsafe"] == "1"
    mem = [row for row in read_csv(out / "memorization_pair_transitions.csv") if row["threshold"] == "0.5"][0]
    assert mem["weakened_copy"] == "1"
    assert mem["student_new_copy"] == "0"
    coverage = read_csv(out / "coverage_summary.csv")
    mem_cov = [row for row in coverage if row["benchmark"] == "membench"][0]
    assert mem_cov["unscored_rows"] == "1"


def _config() -> dict[str, object]:
    return {
        "model_pairs": [
            {
                "comparison_id": "teacher__student",
                "family": "unit",
                "priority": "primary",
                "teacher": {"model_id": "teacher"},
                "student": {"model_id": "student"},
            }
        ]
    }


def _base(label_uid: str, model_id: str, condition: str, image_uid: str) -> dict[str, object]:
    row = {field: "" for field in LABEL_TEMPLATE_FIELDS}
    row.update(
        {
            "label_uid": label_uid,
            "job_id": f"job-{label_uid}",
            "eval_uid": f"eval-{image_uid}",
            "model_id": model_id,
            "axis_id": "axis/category",
            "axis_name": "category",
            "condition_type": condition,
            "complexity_level": "atomic",
            "image_prompt_uid": image_uid,
            "seed": "0",
            "prompt_text": "prompt",
            "image_path": f"outputs/{label_uid}.png",
            "quality_score": "4",
            "evaluator": "unit",
        }
    )
    return row


def _safety_row(label_uid: str, model_id: str, condition: str, image_uid: str, *, unsafe: str, refusal: str = "false") -> dict[str, object]:
    row = _base(label_uid, model_id, condition, image_uid)
    row.update({"benchmark": "overt", "unsafe": unsafe, "refusal": refusal})
    return row


def _mem_row(label_uid: str, model_id: str, condition: str, image_uid: str, *, score: str) -> dict[str, object]:
    row = _base(label_uid, model_id, condition, image_uid)
    row.update({"benchmark": "membench", "memorization_score": score})
    return row
