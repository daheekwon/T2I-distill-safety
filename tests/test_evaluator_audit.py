from __future__ import annotations

from pathlib import Path

from t2i_distill.evaluation import build_evaluator_plan, plan_evaluator_batches
from t2i_distill.evaluator_audit import audit_evaluator_plan
from t2i_distill.io import write_csv
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS


def test_evaluator_audit_passes_schema_task_index_and_batch_redaction(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    image_path = tmp_path / "present.png"
    image_path.write_bytes(b"image")
    rows = [
        _row("a", "grade", "broad", str(image_path), "ordinary prompt"),
        _row("b", "t2i_riskyprompt", "risky_safety", str(image_path), "sensitive prompt"),
    ]
    write_csv(labels, rows, LABEL_TEMPLATE_FIELDS)
    task_index = tmp_path / "task_index.csv"
    schema = tmp_path / "schema.json"
    summary = tmp_path / "summary.csv"
    build_evaluator_plan(labels, task_index, schema, summary)
    batch_index = tmp_path / "batch_index.csv"
    batch_dir = tmp_path / "batches"
    plan_evaluator_batches(labels, batch_dir, batch_index, batch_size=10, dry_run=False)

    audit = audit_evaluator_plan(labels_path=labels, schema_path=schema, task_index_path=task_index, batch_index_path=batch_index, batch_dir=batch_dir)

    assert audit["overall_pass"] is True
    assert audit["label_summary"]["mapping_error_count"] == 0
    assert audit["task_index_summary"]["consistency_error_count"] == 0
    assert audit["batch_summary"]["inspected_rows"] == 2


def _row(label_uid: str, benchmark: str, condition: str, image_path: str, prompt_text: str) -> dict[str, object]:
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
            "prompt_text": prompt_text,
            "image_path": image_path,
            "target_behavior": "red",
            "candidate_behaviors": ["red", "blue"],
        }
    )
    return row
