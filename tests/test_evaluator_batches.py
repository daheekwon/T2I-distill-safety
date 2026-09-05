from __future__ import annotations

import json
from pathlib import Path

from t2i_distill.evaluation import plan_evaluator_batches
from t2i_distill.io import read_csv, write_csv
from t2i_distill.manifest import LABEL_TEMPLATE_FIELDS


def test_plan_evaluator_batches_writes_sharded_jsonl_and_redacts_safety(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    rows = [
        _row("a", "grade", "broad", "m0", "outputs/a.png", prompt_text="a red helmet"),
        _row("b", "grade", "broad", "m1", "outputs/b.png", prompt_text="a blue helmet"),
        _row("c", "grade", "broad", "m0", "outputs/c.png", prompt_text="a green helmet"),
        _row("d", "t2i_riskyprompt", "risky_safety", "m0", "outputs/d.png", prompt_text="sensitive prompt"),
    ]
    write_csv(labels, rows, LABEL_TEMPLATE_FIELDS)
    out_dir = tmp_path / "batches"
    index_path = tmp_path / "batch_index.csv"

    summary = plan_evaluator_batches(labels, out_dir, index_path, batch_size=2, split_fields=("evaluator_kind", "benchmark", "condition_type"))

    assert summary["rows_selected"] == 4
    assert summary["batch_count"] == 3
    index_rows = read_csv(index_path)
    assert [int(row["rows"]) for row in index_rows] == [2, 1, 1]
    safety_path = next(Path(row["path"]) for row in index_rows if row["benchmark"] == "t2i_riskyprompt")
    safety_payload = json.loads(safety_path.read_text(encoding="utf-8").splitlines()[0])
    assert safety_payload["prompt_text"] == "<redacted>"
    assert safety_payload["prompt_chars"] == len("sensitive prompt")


def test_plan_evaluator_batches_only_existing_images(tmp_path: Path) -> None:
    image_path = tmp_path / "present.png"
    image_path.write_bytes(b"not-empty")
    labels = tmp_path / "labels.csv"
    write_csv(
        labels,
        [
            _row("a", "grade", "broad", "m0", str(image_path)),
            _row("b", "grade", "broad", "m1", str(tmp_path / "missing.png")),
        ],
        LABEL_TEMPLATE_FIELDS,
    )

    summary = plan_evaluator_batches(labels, tmp_path / "batches", tmp_path / "index.csv", batch_size=10, only_existing_images=True)

    assert summary["rows_selected"] == 1
    assert summary["skipped_missing_image"] == 1


def _row(label_uid: str, benchmark: str, condition: str, model_id: str, image_path: str, *, prompt_text: str = "prompt") -> dict[str, object]:
    row = {field: "" for field in LABEL_TEMPLATE_FIELDS}
    row.update(
        {
            "label_uid": label_uid,
            "job_id": f"job-{label_uid}",
            "eval_uid": f"eval-{label_uid}",
            "model_id": model_id,
            "benchmark": benchmark,
            "axis_id": f"{benchmark}/axis",
            "axis_name": "axis",
            "condition_type": condition,
            "complexity_level": "atomic",
            "image_prompt_uid": f"image-{label_uid}",
            "seed": "0",
            "prompt_text": prompt_text,
            "image_path": image_path,
            "target_behavior": "red",
            "candidate_behaviors": ["red", "blue"],
            "evaluator": "unit",
        }
    )
    return row
