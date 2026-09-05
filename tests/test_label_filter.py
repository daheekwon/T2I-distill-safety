from __future__ import annotations

from pathlib import Path

from t2i_distill.io import read_csv, read_json, write_csv
from t2i_distill.label_filter import filter_labels_by_existing_images, filter_labels_by_required_signals
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS


def test_filter_labels_by_existing_images_keeps_only_present_nonempty(tmp_path: Path) -> None:
    present = tmp_path / "present.png"
    present.write_bytes(b"png")
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    missing = tmp_path / "missing.png"

    labels = tmp_path / "labels.csv"
    write_csv(
        labels,
        [
            _row("a", "sdxl_base", present),
            _row("b", "sdxl_base", empty),
            _row("c", "sd35_large", missing),
        ],
        LABEL_TEMPLATE_FIELDS,
    )

    summary = filter_labels_by_existing_images(
        labels,
        tmp_path / "existing.csv",
        summary_path=tmp_path / "summary.json",
    )

    assert summary["rows_seen"] == 3
    assert summary["rows_written"] == 1
    assert summary["rows_empty_image"] == 1
    assert summary["rows_missing_image"] == 1
    assert summary["rows_written_by_model"] == {"sdxl_base": 1}
    assert read_csv(tmp_path / "existing.csv")[0]["label_uid"] == "a"
    assert read_json(tmp_path / "summary.json")["rows_written"] == 1


def test_filter_labels_by_required_signals_removes_unscored_membench(tmp_path: Path) -> None:
    image = tmp_path / "present.png"
    image.write_bytes(b"png")
    scored = _row("a", "sdxl_base", image)
    scored["quality_score"] = "4"
    scored["memorization_score"] = "0.2"
    unscored = _row("b", "sdxl_base", image)
    unscored["quality_score"] = "4"
    grade = _row("c", "sdxl_base", image)
    grade["benchmark"] = "grade"
    grade["condition_type"] = "broad"
    grade["semantic_label"] = "red"
    grade["quality_score"] = "5"
    labels = tmp_path / "labels.csv"
    write_csv(labels, [scored, unscored, grade], LABEL_TEMPLATE_FIELDS)

    summary = filter_labels_by_required_signals(labels, tmp_path / "claim_ready.csv", summary_path=tmp_path / "required_summary.json")

    assert summary["rows_seen"] == 3
    assert summary["rows_written"] == 2
    assert summary["rows_removed_missing_required"] == 1
    assert summary["missing_required_by_column"] == {"memorization_score": 1}
    assert [row["label_uid"] for row in read_csv(tmp_path / "claim_ready.csv")] == ["a", "c"]
    assert read_json(tmp_path / "required_summary.json")["rows_removed_by_benchmark"] == {"membench": 1}


def _row(label_uid: str, model_id: str, image_path: Path) -> dict[str, object]:
    return {
        "label_uid": label_uid,
        "job_id": f"job-{label_uid}",
        "eval_uid": f"eval-{label_uid}",
        "model_id": model_id,
        "benchmark": "membench",
        "axis_id": "membench/test",
        "axis_name": "test",
        "condition_type": "memorization_trigger",
        "complexity_level": "atomic",
        "image_prompt_uid": f"prompt-{label_uid}",
        "seed": 0,
        "prompt_text": "test prompt",
        "image_path": str(image_path),
        "target_behavior": "",
        "candidate_behaviors": [],
        "semantic_label": "",
        "semantic_label_confidence": "",
        "pass_score": "",
        "quality_score": "",
        "refusal": "",
        "unsafe": "",
        "memorization_score": "",
        "evaluator": "test",
        "notes": "",
    }
