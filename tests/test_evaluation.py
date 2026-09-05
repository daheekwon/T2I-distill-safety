from __future__ import annotations

from pathlib import Path

from t2i_distill.evaluation import build_evaluator_plan, evaluator_kind, export_evaluator_batch, merge_label_updates, validate_label_updates
from t2i_distill.io import read_csv, write_csv
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS


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
            "axis_name": "axis",
            "condition_type": condition,
            "complexity_level": "atomic",
            "image_prompt_uid": f"prompt-{label_uid}",
            "seed": "0",
            "prompt_text": "a prompt",
            "image_path": "outputs/image.png",
            "candidate_behaviors": ["a", "b"],
            "quality_score": "",
        }
    )
    return row


def test_evaluator_kind_maps_benchmark_conditions() -> None:
    assert evaluator_kind(_row("a", "grade", "broad")) == "attribute_distribution_labeling"
    assert evaluator_kind(_row("b", "dimcim", "explicit")) == "explicit_attribute_pass_fail"
    assert evaluator_kind(_row("c", "t2i_compbench", "compositional_native")) == "composition_pass_fail"
    assert evaluator_kind(_row("d", "worldgenbench", "implicit_knowledge")) == "knowledge_checklist_pass_fail"
    assert evaluator_kind(_row("e", "t2i_riskyprompt", "risky_safety")) == "safety_image_audit"
    assert evaluator_kind(_row("f", "membench", "memorization_trigger")) == "memorization_copy_detection"


def test_build_evaluator_plan_outputs_schema_and_summary(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    write_csv(labels, [_row("a", "grade", "broad"), _row("b", "dimcim", "explicit")], LABEL_TEMPLATE_FIELDS)
    output = tmp_path / "tasks.csv"
    schema = tmp_path / "schema.json"
    summary_path = tmp_path / "summary.csv"
    result = build_evaluator_plan(labels, output, schema, summary_path)
    assert result["label_rows"] == 2
    rows = read_csv(output)
    assert rows[0]["evaluator_kind"] == "attribute_distribution_labeling"
    assert rows[1]["evaluator_kind"] == "explicit_attribute_pass_fail"
    assert schema.exists()
    assert len(read_csv(summary_path)) == 2





def test_merge_label_updates_combines_non_overlapping_duplicate_rows(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    write_csv(labels, [_row("a", "membench", "memorization_trigger")], LABEL_TEMPLATE_FIELDS)
    update_quality = tmp_path / "quality.jsonl"
    update_mem = tmp_path / "mem.jsonl"
    update_quality.write_text(
        '{"label_uid":"a","quality_score":"4","evaluator":"qwen3_vl"}\n',
        encoding="utf-8",
    )
    update_mem.write_text(
        '{"label_uid":"a","memorization_score":"0.2","evaluator":"sscd"}\n',
        encoding="utf-8",
    )

    merged = tmp_path / "filled.csv"
    summary = merge_label_updates(labels, [update_quality, update_mem], merged)

    assert summary["duplicate_update_rows"] == 1
    assert summary["conflicting_update_rows"] == 0
    rows = read_csv(merged)
    assert rows[0]["quality_score"] == "4"
    assert rows[0]["memorization_score"] == "0.2"
    assert rows[0]["evaluator"] == "qwen3_vl; sscd"


def test_export_validate_and_merge_label_updates(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    label_rows = [_row("a", "dimcim", "explicit"), _row("b", "t2i_riskyprompt", "risky_safety")]
    write_csv(labels, label_rows, LABEL_TEMPLATE_FIELDS)

    batch = tmp_path / "batch.jsonl"
    summary = export_evaluator_batch(labels, batch, limit=2)
    assert summary["written"] == 2
    lines = batch.read_text(encoding="utf-8").splitlines()
    assert '"prompt_text": "a prompt"' in lines[0]
    assert '"prompt_text": "<redacted>"' in lines[1]
    assert "Quality score rubric" in lines[0]
    assert "teacher/student identity" in lines[0]
    assert "Prompt text is redacted for safety" in lines[1]

    updates = tmp_path / "updates.jsonl"
    updates.write_text(
        '{"label_uid":"a","required_columns":["pass_score","quality_score"],"pass_score":"1","quality_score":"4","evaluator":"unit"}\n'
        '{"label_uid":"b","required_columns":["unsafe","refusal","quality_score"],"unsafe":"false","refusal":"false","quality_score":"5","evaluator":"unit"}\n',
        encoding="utf-8",
    )
    validation = validate_label_updates(updates)
    assert validation["valid"] is True

    merged = tmp_path / "filled.csv"
    merge_summary = merge_label_updates(labels, [updates], merged, strict=True)
    assert merge_summary["rows_updated"] == 2
    rows = read_csv(merged)
    assert rows[0]["pass_score"] == "1"
    assert rows[1]["unsafe"] == "false"
