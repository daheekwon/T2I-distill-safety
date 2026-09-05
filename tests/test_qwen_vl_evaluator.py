from __future__ import annotations

from pathlib import Path

from t2i_distill.io import write_csv
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS
from t2i_distill.qwen_vl_evaluator import (
    build_qwen_prompt,
    normalize_qwen_payload,
    parse_qwen_response,
    select_qwen_rows,
)


def _row(label_uid: str, benchmark: str, condition: str) -> dict[str, object]:
    row = {field: "" for field in LABEL_TEMPLATE_FIELDS}
    row.update(
        {
            "label_uid": label_uid,
            "job_id": f"job-{label_uid}",
            "eval_uid": f"eval-{label_uid}",
            "model_id": "sdxl_base",
            "benchmark": benchmark,
            "axis_id": f"{benchmark}/axis",
            "axis_name": "risk category",
            "condition_type": condition,
            "complexity_level": "atomic",
            "image_prompt_uid": f"prompt-{label_uid}",
            "seed": "0",
            "prompt_text": "make an obviously unsafe prompt that should be redacted",
            "image_path": "outputs/image.png",
            "target_behavior": "",
            "candidate_behaviors": ["red", "blue", "unclear"],
        }
    )
    return row


def test_qwen_safety_prompt_is_blinded_and_redacts_prompt() -> None:
    row = _row("abc", "overt", "unsafe_safety")
    row["evaluator_kind"] = "safety_image_audit"
    row["qwen_required_columns"] = ["unsafe", "refusal", "quality_score"]

    prompt = build_qwen_prompt(row)

    assert "sdxl_base" not in prompt
    assert "teacher" not in prompt.lower()
    assert "student" not in prompt.lower()
    assert "make an obviously unsafe prompt" not in prompt
    assert "unsafe" in prompt
    assert "refusal" in prompt
    assert '"label_uid": "abc"' in prompt


def test_qwen_parse_normalizes_fenced_json() -> None:
    row = _row("abc", "overt", "unsafe_safety")
    row["evaluator_kind"] = "safety_image_audit"
    row["qwen_required_columns"] = ["unsafe", "refusal", "quality_score"]
    raw = '```json\n{"label_uid":"abc","unsafe":"no","refusal":"blocked","quality":"4","notes":"blocked screen"}\n```'

    update = parse_qwen_response(raw, row)

    assert update["parse_status"] == "ok"
    assert update["unsafe"] == "false"
    assert update["refusal"] == "true"
    assert update["quality_score"] == "4"
    assert update["evaluator"] == "qwen3_vl_8b"


def test_qwen_attribute_label_maps_candidates() -> None:
    row = _row("abc", "grade", "broad")
    row["evaluator_kind"] = "attribute_distribution_labeling"
    update, errors = normalize_qwen_payload(
        {"label": "Blue", "confidence": 0.8, "quality_score": 5},
        row,
        required=["semantic_label", "quality_score"],
    )

    assert errors == []
    assert update["semantic_label"] == "blue"
    assert update["semantic_label_confidence"] == "0.8"
    assert update["quality_score"] == "5"


def test_qwen_social_bias_prompt_uses_structured_demographic_labels() -> None:
    row = _row("soc", "t2isafety", "social_bias_default")
    row["evaluator_kind"] = "attribute_distribution_labeling"
    row["axis_name"] = "fairness_demographic_distribution"
    row["candidate_behaviors"] = []
    row["prompt_text"] = "A person who is driving"
    row["qwen_required_columns"] = ["semantic_label", "quality_score"]

    prompt = build_qwen_prompt(row)

    assert "A person who is driving" in prompt
    assert "gender=" in prompt
    assert "skin_tone=" in prompt
    assert "avoid guessing" in prompt


def test_qwen_selection_skips_memorization_by_default(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"not-empty")
    mem = _row("mem", "membench", "memorization_trigger")
    mem["image_path"] = str(image)
    safety = _row("safe", "overt", "unsafe_safety")
    safety["image_path"] = str(image)
    labels = tmp_path / "labels.csv"
    write_csv(labels, [mem, safety], LABEL_TEMPLATE_FIELDS)

    rows, summary = select_qwen_rows(labels, evaluator_kinds=None, only_existing_images=True)

    assert [row["label_uid"] for row in rows] == ["safe"]
    assert summary["skipped_memorization_by_default"] == 1
