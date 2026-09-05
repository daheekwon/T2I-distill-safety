from __future__ import annotations

from pathlib import Path

from t2i_distill.config import load_config
from t2i_distill.io import write_csv
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS
from t2i_distill.readiness import audit_rq_readiness, row_signal_state


def test_row_signal_state_requires_condition_specific_columns() -> None:
    assert row_signal_state({"benchmark": "grade", "condition_type": "broad", "semantic_label": "red", "quality_score": "4"}, min_quality=3)["ready"] is True
    assert row_signal_state({"benchmark": "grade", "condition_type": "broad", "semantic_label": "", "quality_score": "4"}, min_quality=3)["ready"] is False
    assert row_signal_state({"benchmark": "overt", "condition_type": "unsafe_safety", "refusal": "false", "unsafe": "true", "quality_score": "4"}, min_quality=3)["ready"] is True
    assert row_signal_state({"benchmark": "membench", "condition_type": "memorization_trigger", "memorization_score": "0.2", "quality_score": "2"}, min_quality=3)["ready"] is False


def test_audit_rq_readiness_counts_matched_cells(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    rows = [
        _row("a", "sdxl_base", "grade", "broad", "grade/color", "prompt-a", semantic_label="red", quality_score="4"),
        _row("b", "sdxl_lightning_4step", "grade", "broad", "grade/color", "prompt-a", semantic_label="red", quality_score="4"),
        _row("c", "sd35_large", "grade", "broad", "grade/color", "prompt-a", semantic_label="red", quality_score="4"),
        _row("d", "sd35_large_turbo", "grade", "broad", "grade/color", "prompt-a", semantic_label="", quality_score="4"),
    ]
    write_csv(labels, rows, LABEL_TEMPLATE_FIELDS)
    readiness = audit_rq_readiness(labels, load_config(Path("config/experiment.json")))
    sdxl = next(row for row in readiness["rq_pair_rows"] if row["rq"] == "rq1_sharpening_fit" and row["comparison_id"] == "sdxl_base__sdxl_lightning_4step")
    sd35 = next(row for row in readiness["rq_pair_rows"] if row["rq"] == "rq1_sharpening_fit" and row["comparison_id"] == "sd35_large__sd35_large_turbo")
    assert sdxl["expected_matched_cells"] == 1
    assert sdxl["ready_matched_cells"] == 1
    assert sdxl["status"] == "ready"
    assert sd35["expected_matched_cells"] == 1
    assert sd35["ready_matched_cells"] == 0
    assert sd35["status"] == "incomplete"


def _row(
    label_uid: str,
    model_id: str,
    benchmark: str,
    condition: str,
    axis_id: str,
    image_uid: str,
    *,
    semantic_label: str = "",
    pass_score: str = "",
    quality_score: str = "4",
    refusal: str = "",
    unsafe: str = "",
    memorization_score: str = "",
) -> dict[str, object]:
    row = {field: "" for field in LABEL_TEMPLATE_FIELDS}
    row.update(
        {
            "label_uid": label_uid,
            "job_id": f"job-{label_uid}",
            "eval_uid": f"eval-{axis_id}",
            "model_id": model_id,
            "benchmark": benchmark,
            "axis_id": axis_id,
            "axis_name": axis_id,
            "condition_type": condition,
            "complexity_level": "atomic",
            "image_prompt_uid": image_uid,
            "seed": "0",
            "prompt_text": "a prompt",
            "image_path": "outputs/image.png",
            "target_behavior": "red",
            "candidate_behaviors": ["red", "blue"],
            "semantic_label": semantic_label,
            "pass_score": pass_score,
            "quality_score": quality_score,
            "refusal": refusal,
            "unsafe": unsafe,
            "memorization_score": memorization_score,
            "evaluator": "unit",
        }
    )
    return row
