from __future__ import annotations

import json
from pathlib import Path

from t2i_distill.io import write_csv, write_jsonl
from t2i_distill.label_quality import audit_label_quality
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS


def test_label_quality_passes_valid_filled_rows(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    write_csv(
        labels,
        [
            _row("a", "grade", "broad", semantic_label="red", quality_score="4", confidence="0.9"),
            _row("b", "overt", "unsafe_safety", refusal="false", unsafe="true", quality_score="5"),
            _row("c", "membench", "memorization_trigger", memorization_score="0.2", quality_score="4"),
        ],
        LABEL_TEMPLATE_FIELDS,
    )

    audit = audit_label_quality(labels)

    assert audit["overall_pass"] is True
    assert audit["final_label_quality"]["filled_rows"] == 3
    assert audit["final_label_quality"]["missing_required_values"] == 0
    assert audit["final_label_quality"]["out_of_vocab_semantic_labels"] == 0


def test_label_quality_flags_missing_invalid_and_update_conflicts(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    update_a = tmp_path / "a.jsonl"
    update_b = tmp_path / "b.jsonl"
    write_csv(
        labels,
        [
            _row("a", "grade", "broad", semantic_label="chartreuse", quality_score="9", confidence="0.2"),
            _row("b", "overt", "unsafe_safety", refusal="maybe", unsafe="true", quality_score=""),
        ],
        LABEL_TEMPLATE_FIELDS,
    )
    write_jsonl(update_a, [{"label_uid": "a", "semantic_label": "red", "quality_score": 4, "required_columns": ["semantic_label", "quality_score"]}])
    write_jsonl(update_b, [{"label_uid": "a", "semantic_label": "blue", "quality_score": 2, "required_columns": ["semantic_label", "quality_score"]}])

    audit = audit_label_quality(labels, update_paths=[update_a, update_b])

    assert audit["overall_pass"] is False
    assert audit["final_label_quality"]["missing_required_values"] > 0
    assert audit["final_label_quality"]["numeric_range_errors"] == 1
    assert audit["final_label_quality"]["bool_errors"] == 1
    assert audit["final_label_quality"]["low_confidence_rows"] == 1
    assert audit["final_label_quality"]["out_of_vocab_semantic_labels"] == 1
    assert audit["update_reliability"]["overlap_label_uids"] == 1
    assert audit["update_reliability"]["conflicts_by_field"]["behavior_label"] == 1


def test_label_quality_allows_unclear_low_confidence_abstention(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    write_csv(
        labels,
        [_row("a", "grade", "broad", semantic_label="unclear", quality_score="4", confidence="0.2")],
        LABEL_TEMPLATE_FIELDS,
    )

    audit = audit_label_quality(labels)

    assert audit["overall_pass"] is True
    assert audit["final_label_quality"]["low_confidence_rows"] == 0


def test_label_quality_allows_candidate_prefix_match(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    row = _row("a", "dimcim", "broad", semantic_label="on a bed", quality_score="4", confidence="0.9")
    row["candidate_behaviors"] = ["on the floor", "on a bed frame"]
    write_csv(labels, [row], LABEL_TEMPLATE_FIELDS)

    audit = audit_label_quality(labels)

    assert audit["overall_pass"] is True
    assert audit["final_label_quality"]["out_of_vocab_semantic_labels"] == 0


def _row(
    label_uid: str,
    benchmark: str,
    condition: str,
    *,
    semantic_label: str = "",
    quality_score: str = "4",
    confidence: str = "",
    refusal: str = "",
    unsafe: str = "",
    memorization_score: str = "",
) -> dict[str, object]:
    row = {field: "" for field in LABEL_TEMPLATE_FIELDS}
    row.update(
        {
            "label_uid": label_uid,
            "job_id": f"job-{label_uid}",
            "eval_uid": f"eval-{label_uid}",
            "model_id": "m0",
            "benchmark": benchmark,
            "axis_id": f"{benchmark}/axis",
            "axis_name": "axis",
            "condition_type": condition,
            "complexity_level": "atomic",
            "image_prompt_uid": f"image-{label_uid}",
            "seed": "0",
            "prompt_text": "prompt",
            "image_path": f"outputs/{label_uid}.png",
            "target_behavior": "red",
            "candidate_behaviors": ["red", "blue"],
            "semantic_label": semantic_label,
            "semantic_label_confidence": confidence,
            "quality_score": quality_score,
            "refusal": refusal,
            "unsafe": unsafe,
            "memorization_score": memorization_score,
            "evaluator": "unit",
        }
    )
    return row
